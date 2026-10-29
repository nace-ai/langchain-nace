import base64
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_nace import NaceClassifier, NaceOverloadedError, NoulCriteria
from langchain_nace.experimental.middleware import (
    Allow,
    AskHuman,
    Block,
    GateQuestions,
    GateVerdict,
    NaceAutoModeMiddleware,
    ThresholdPolicy,
)
from tests.agents import noul, score, scripted, tool_call
from tests.fakes import FakeDrex, error

RED = base64.b64encode((Path(__file__).parents[3] / "fixtures" / "red.png").read_bytes()).decode()

calls: list[str] = []


@tool
def read_ticket(ticket_id: int) -> str:
    """Read a support ticket."""
    calls.append(f"read:{ticket_id}")
    return f"Ticket {ticket_id}: printer broken"


@tool
def delete_ticket(ticket_id: int) -> str:
    """Delete a support ticket."""
    calls.append(f"delete:{ticket_id}")
    return "deleted"


@pytest.fixture(autouse=True)
def _reset() -> None:
    calls.clear()


def answers(drex: FakeDrex, risk: float, destructive: float, matches: float) -> None:
    drex.overrides.update(
        {"risk": score(risk), "destructive": noul(destructive), "matches_request": noul(matches)}
    )


def agent(gate: NaceAutoModeMiddleware, *script: AIMessage, checkpointer: object = None):
    return create_agent(
        scripted(*script, AIMessage("done")),
        tools=[read_ticket, delete_ticket],
        middleware=[gate],
        checkpointer=checkpointer,
    )  # type: ignore[arg-type]


def last_tool_message(result: dict) -> ToolMessage:
    return next(m for m in reversed(result["messages"]) if isinstance(m, ToolMessage))


def test_allow_runs_the_tool(classifier: NaceClassifier, drex: FakeDrex) -> None:
    answers(drex, risk=0, destructive=0.05, matches=0.95)
    agent(
        NaceAutoModeMiddleware(model=classifier), tool_call("read_ticket", {"ticket_id": 7}, "c1")
    ).invoke({"messages": [HumanMessage("Summarize ticket 7")]})
    assert calls == ["read:7"]
    assert len(drex.requests) == 1
    assert set(drex.bodies[0]["questions"]) == {"risk", "destructive", "matches_request"}


def test_block_returns_an_error_and_skips_the_tool(
    classifier: NaceClassifier, drex: FakeDrex
) -> None:
    answers(drex, risk=4, destructive=0.97, matches=0.03)
    result = agent(
        NaceAutoModeMiddleware(model=classifier), tool_call("delete_ticket", {"ticket_id": 7}, "c1")
    ).invoke({"messages": [HumanMessage("Summarize ticket 7")]})
    assert calls == []
    message = last_tool_message(result)
    assert message.status == "error"
    assert "blocked" in message.content
    assert message.artifact["nace_verdict"] == {
        "risk": 1.0,
        "p_destructive": 0.97,
        "p_matches_request": 0.03,
    }


def test_destructive_and_not_asked_for_is_blocked_even_at_low_risk(classifier, drex) -> None:
    answers(drex, risk=1, destructive=0.9, matches=0.1)
    agent(
        NaceAutoModeMiddleware(model=classifier), tool_call("delete_ticket", {"ticket_id": 7}, "c1")
    ).invoke({"messages": [HumanMessage("Summarize ticket 7")]})
    assert calls == []


def ask_setup(classifier: NaceClassifier, drex: FakeDrex):
    answers(drex, risk=2, destructive=0.8, matches=0.9)
    graph = agent(
        NaceAutoModeMiddleware(model=classifier),
        tool_call("delete_ticket", {"ticket_id": 7}, "c1"),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}
    paused = graph.invoke({"messages": [HumanMessage("Delete ticket 7")]}, config)
    return graph, config, paused


def test_ask_interrupts_and_approve_runs(classifier: NaceClassifier, drex: FakeDrex) -> None:
    graph, config, paused = ask_setup(classifier, drex)
    request = paused["__interrupt__"][0].value
    assert request["type"] == "nace_tool_approval"
    assert request["name"] == "delete_ticket"
    assert request["args"] == {"ticket_id": 7}
    assert calls == []
    graph.invoke(Command(resume={"type": "approve"}), config)
    assert calls == ["delete:7"]


def test_ask_reject(classifier: NaceClassifier, drex: FakeDrex) -> None:
    graph, config, _ = ask_setup(classifier, drex)
    result = graph.invoke(Command(resume={"type": "reject", "message": "not today"}), config)
    assert calls == []
    assert "not today" in last_tool_message(result).content


def test_ask_edit_runs_with_new_args(classifier: NaceClassifier, drex: FakeDrex) -> None:
    graph, config, _ = ask_setup(classifier, drex)
    graph.invoke(Command(resume={"type": "edit", "args": {"ticket_id": 8}}), config)
    assert calls == ["delete:8"]


def test_ask_human_off_blocks(classifier: NaceClassifier, drex: FakeDrex) -> None:
    answers(drex, risk=2, destructive=0.8, matches=0.9)
    gate = NaceAutoModeMiddleware(model=classifier, policy=ThresholdPolicy(ask_human=False))
    agent(gate, tool_call("delete_ticket", {"ticket_id": 7}, "c1")).invoke(
        {"messages": [HumanMessage("Delete 7")]}
    )
    assert calls == []


def test_fails_closed_when_drex_is_down(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.replies.extend([error(529, "overloaded", kind="overloaded_error")] * 3)
    result = agent(
        NaceAutoModeMiddleware(model=classifier), tool_call("read_ticket", {"ticket_id": 7}, "c1")
    ).invoke({"messages": [HumanMessage("Summarize ticket 7")]})
    assert calls == []
    assert "could not be checked" in last_tool_message(result).content


def test_on_error_raise(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.replies.extend([error(529, "overloaded", kind="overloaded_error")] * 3)
    with pytest.raises(NaceOverloadedError):
        agent(
            NaceAutoModeMiddleware(model=classifier, on_error="raise"),
            tool_call("read_ticket", {"ticket_id": 7}, "c1"),
        ).invoke({"messages": [HumanMessage("x")]})


def test_unlisted_tools_are_not_checked(classifier: NaceClassifier, drex: FakeDrex) -> None:
    agent(
        NaceAutoModeMiddleware(model=classifier, tools=[delete_ticket]),
        tool_call("read_ticket", {"ticket_id": 7}, "c1"),
    ).invoke({"messages": [HumanMessage("x")]})
    assert calls == ["read:7"]
    assert drex.requests == []


def test_original_request_survives_40_tool_calls(
    classifier: NaceClassifier, drex: FakeDrex
) -> None:
    answers(drex, risk=0, destructive=0.05, matches=0.95)
    script = [tool_call("read_ticket", {"ticket_id": i}, f"c{i}") for i in range(40)]
    agent(NaceAutoModeMiddleware(model=classifier), *script).invoke(
        {"messages": [HumanMessage("Read tickets 0-39 and summarize. Do not delete anything.")]}
    )
    assert len(drex.requests) == 40
    last = drex.bodies[-1]["state"]
    assert last["original_request"] == "Read tickets 0-39 and summarize. Do not delete anything."
    assert len(last["recent_messages"]) == 20
    assert all(
        m["role"] != "user" for m in last["recent_messages"]
    )  # the request fell out of the window
    assert last["tool_call"] == {"id": "c39", "name": "read_ticket", "args": {"ticket_id": 39}}
    assert last["tool_description"] == "Read a support ticket."


def test_screenshots_go_to_drex_max(make_classifier, drex: FakeDrex) -> None:
    answers(drex, risk=0, destructive=0.05, matches=0.95)
    gate = NaceAutoModeMiddleware(model=make_classifier(model="drex-max"), screenshots=True)
    agent(gate, tool_call("read_ticket", {"ticket_id": 7}, "c1")).invoke(
        {
            "messages": [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Here is my screen"},
                        {"type": "image", "base64": RED, "mime_type": "image/png"},
                    ]
                )
            ]
        }
    )
    body = drex.bodies[0]
    assert body["images"] == [f"data:image/png;base64,{RED}"]
    assert "Image 1" in body["state"]["screenshots"]
    assert body["state"]["recent_messages"][0]["content"] == "Here is my screen\n[image omitted]"


def test_screenshot_function(make_classifier, drex: FakeDrex) -> None:
    answers(drex, risk=0, destructive=0.05, matches=0.95)
    gate = NaceAutoModeMiddleware(
        model=make_classifier(model="drex-max"), screenshots=lambda request: [RED]
    )
    agent(gate, tool_call("read_ticket", {"ticket_id": 7}, "c1")).invoke(
        {"messages": [HumanMessage("x")]}
    )
    assert drex.bodies[0]["images"] == [RED]


def test_rejected_screenshot_is_retried_without_it(make_classifier, drex: FakeDrex) -> None:
    drex.replies.append(error(422, "bad", issues=[{"path": "images.0", "message": "unreadable"}]))
    answers(drex, risk=0, destructive=0.05, matches=0.95)
    gate = NaceAutoModeMiddleware(
        model=make_classifier(model="drex-max"), screenshots=lambda request: [RED]
    )
    agent(gate, tool_call("read_ticket", {"ticket_id": 7}, "c1")).invoke(
        {"messages": [HumanMessage("x")]}
    )
    assert len(drex.requests) == 2
    assert "images" not in drex.bodies[1]
    assert calls == ["read:7"]


def test_screenshots_need_an_image_model(make_classifier) -> None:
    with pytest.raises(ValueError, match="drex-max"):
        NaceAutoModeMiddleware(model=make_classifier(model="drex-v2.0"), screenshots=True)


def test_default_gate_takes_screenshots(classifier: NaceClassifier, drex: FakeDrex) -> None:
    answers(drex, risk=0, destructive=0.05, matches=0.95)
    gate = NaceAutoModeMiddleware(model=classifier, screenshots=lambda request: [RED])
    agent(gate, tool_call("read_ticket", {"ticket_id": 7}, "c1")).invoke(
        {"messages": [HumanMessage("x")]}
    )
    assert drex.bodies[0]["model"] == "drex-max"
    assert drex.bodies[0]["images"] == [RED]


async def test_async_block_and_allow(classifier: NaceClassifier, drex: FakeDrex) -> None:
    answers(drex, risk=4, destructive=0.97, matches=0.03)
    graph = agent(
        NaceAutoModeMiddleware(model=classifier), tool_call("delete_ticket", {"ticket_id": 1}, "c1")
    )
    await graph.ainvoke({"messages": [HumanMessage("Summarize")]})
    assert calls == []
    answers(drex, risk=0, destructive=0.05, matches=0.95)
    graph = agent(
        NaceAutoModeMiddleware(model=classifier), tool_call("read_ticket", {"ticket_id": 1}, "c1")
    )
    await graph.ainvoke({"messages": [HumanMessage("Summarize")]})
    assert calls == ["read:1"]


async def test_async_ask_then_approve(classifier: NaceClassifier, drex: FakeDrex) -> None:
    answers(drex, risk=2, destructive=0.8, matches=0.9)
    graph = agent(
        NaceAutoModeMiddleware(model=classifier),
        tool_call("delete_ticket", {"ticket_id": 3}, "c1"),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "a"}}
    await graph.ainvoke({"messages": [HumanMessage("Delete 3")]}, config)
    await graph.ainvoke(Command(resume={"type": "approve"}), config)
    assert calls == ["delete:3"]


@pytest.mark.parametrize(
    ("risk", "destructive", "matches", "expected"),
    [
        (0.0, 0.1, 0.9, Allow),
        (0.75, 0.1, 0.9, Block),
        (0.74, 0.1, 0.9, AskHuman),
        (0.4, 0.1, 0.9, AskHuman),
        (0.39, 0.1, 0.9, Allow),
        (0.1, 0.5, 0.5, AskHuman),
        (0.1, 0.5, 0.49, Block),
        (0.1, 0.49, 0.0, Allow),
    ],
)
def test_threshold_policy(risk: float, destructive: float, matches: float, expected: type) -> None:
    verdict = GateVerdict(
        risk=risk, p_destructive=destructive, p_matches_request=matches, response=None
    )  # type: ignore[arg-type]
    assert type(ThresholdPolicy()(verdict)) is expected


def test_risk_scales_with_any_rubric_length(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.overrides.update(
        {"risk": score(2, levels=3), "destructive": noul(0.0), "matches_request": noul(1.0)}
    )
    verdict = GateVerdict.from_response(
        classifier.invoke(
            {
                "state": "x",
                "questions": {
                    "risk": {"type": "score", "criteria": ["a", "b", "c"]},
                    "destructive": {"type": "noul", "instructions": "?"},
                    "matches_request": {"type": "noul", "instructions": "?"},
                },
            }
        )
    )
    assert verdict.risk == 1.0


def test_typesafe_instructions_and_criteria(classifier: NaceClassifier, drex: FakeDrex) -> None:
    # The arguments `langchain-typesafe`'s AutoModeMiddleware takes.
    answers(drex, risk=4, destructive=0.97, matches=0.03)
    gate = NaceAutoModeMiddleware(
        model=classifier,
        tools=[delete_ticket],
        instructions="Would running `tool_call` be risky?",
        criteria=NoulCriteria(true="Writes or deletes data.", false="Only reads data."),
    )
    result = agent(gate, tool_call("delete_ticket", {"ticket_id": 7}, "c1")).invoke(
        {"messages": [HumanMessage("Summarize ticket 7")]}
    )
    assert calls == []
    assert "blocked" in last_tool_message(result).content
    questions = drex.bodies[0]["questions"]
    assert questions["risk"]["instructions"] == "Would running `tool_call` be risky?"
    assert questions["risk"]["criteria"] == GateQuestions().risk.criteria
    assert questions["destructive"]["criteria"] == {
        "true": "Writes or deletes data.",
        "false": "Only reads data.",
    }
    assert questions["destructive"]["instructions"] == GateQuestions().destructive.instructions


def test_questions_with_instructions_or_criteria_is_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        NaceAutoModeMiddleware(questions=GateQuestions(), instructions="Risky?")
    with pytest.raises(ValueError, match="not both"):
        NaceAutoModeMiddleware(questions=GateQuestions(), criteria=NoulCriteria(true="x"))


def test_typesafe_readme_gate_runs_with_only_the_import_changed(
    classifier: NaceClassifier, drex: FakeDrex, monkeypatch: pytest.MonkeyPatch
) -> None:
    # langchain-typesafe README, with `langchain_typesafe` replaced by `langchain_nace`.
    from langchain_nace import NoulCriteria as Criteria

    monkeypatch.setenv("NACE_API_KEY", "nace_sk_test")
    from langchain_nace.experimental.middleware import AutoModeMiddleware

    answers(drex, risk=4, destructive=0.97, matches=0.03)
    auto_mode = AutoModeMiddleware(
        tools=[delete_ticket],
        criteria=Criteria(
            true="The call writes, deletes, publishes, or changes access.",
            false="The call only reads public or user-provided data.",
        ),
    )
    auto_mode.classifier = classifier
    agent(auto_mode, tool_call("delete_ticket", {"ticket_id": 7}, "c1")).invoke(
        {"messages": [HumanMessage("Summarize ticket 7")]}
    )
    assert calls == []
