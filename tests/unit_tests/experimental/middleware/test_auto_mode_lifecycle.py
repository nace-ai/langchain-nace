"""Approval lifecycle across checkpoints, workers, and changed inputs."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_nace import NaceClassifier
from langchain_nace.experimental.middleware import NaceAutoModeMiddleware
from tests.agents import noul, score, scripted, tool_call
from tests.fakes import FakeDrex, ok_body

FIXTURES = Path(__file__).parents[3] / "fixtures"
RED = base64.b64encode((FIXTURES / "red.png").read_bytes()).decode()
BLUE = base64.b64encode((FIXTURES / "blue.png").read_bytes()).decode()

ASK = {"risk": score(2), "destructive": noul(0.8), "matches_request": noul(0.9)}  # AskHuman
ALLOW = {"risk": score(0), "destructive": noul(0.05), "matches_request": noul(0.95)}  # -> Allow

calls: list[str] = []


@tool
def delete_ticket(ticket_id: int) -> str:
    """Delete a support ticket."""
    calls.append(f"delete:{ticket_id}")
    return "deleted"


@pytest.fixture(autouse=True)
def _reset() -> None:
    calls.clear()


def reply(overrides: dict):
    return lambda body: (200, {}, ok_body(body, overrides))


def classifier_for(drex: FakeDrex) -> NaceClassifier:
    return NaceClassifier(
        model="drex-max",
        api_key="nace_sk_test",
        base_url="https://drex.test",
        http_client=httpx.Client(transport=drex.transport()),
        http_async_client=httpx.AsyncClient(transport=drex.async_transport()),
    )


def build(drex: FakeDrex, saver: InMemorySaver, screen: dict, *script: AIMessage):
    """A fresh agent + fresh middleware (= a worker) on a shared checkpointer."""
    gate = NaceAutoModeMiddleware(
        model=classifier_for(drex), screenshots=lambda request: [screen["png"]]
    )
    return create_agent(
        scripted(*script, AIMessage("done")),
        tools=[delete_ticket],
        middleware=[gate],
        checkpointer=saver,
    )


CONFIG = {"configurable": {"thread_id": "t1"}}


def pause(second: dict, *, rebuild: bool, resume: dict):
    """Pause on delete_ticket (verdict 1 asks), change the screen, resume. Returns drex."""
    drex = FakeDrex(replies=[reply(ASK), reply(second)])
    saver = InMemorySaver()
    screen = {"png": RED}
    graph = build(drex, saver, screen, tool_call("delete_ticket", {"ticket_id": 7}, "c1"))
    paused = graph.invoke({"messages": [HumanMessage("Delete ticket 7")]}, CONFIG)
    assert paused["__interrupt__"][0].value["type"] == "nace_tool_approval"
    assert calls == []
    screen["png"] = BLUE  # the context the approval was given for is gone
    if rebuild:
        graph = build(drex, saver, screen)  # another worker: new agent, new middleware
    graph.invoke(Command(resume=resume), CONFIG)
    assert len(drex.bodies) == 1  # Resume replays the saved decision.
    assert drex.bodies[0]["images"] == [RED]
    return drex


# Approval is bound to the exact context the person reviewed.
@pytest.mark.parametrize("rebuild", [True, False], ids=["rebuilt", "same_agent"])
def test_changed_screenshot_cannot_reuse_approval_for_ask(rebuild: bool) -> None:
    pause(ASK, rebuild=rebuild, resume={"type": "approve"})
    assert calls == [], f"tool ran on a stale approval: {calls}"


@pytest.mark.parametrize("rebuild", [True, False], ids=["rebuilt", "same_agent"])
def test_changed_screenshot_cannot_reuse_approval_for_allow(rebuild: bool) -> None:
    pause(ALLOW, rebuild=rebuild, resume={"type": "approve"})
    assert calls == [], f"tool ran on a stale approval: {calls}"


# A lower-risk response must never bypass a pending rejection.
@pytest.mark.parametrize("rebuild", [True, False], ids=["rebuilt", "same_agent"])
def test_rejection_replays_original_decision(rebuild: bool) -> None:
    drex = FakeDrex(replies=[reply(ASK), reply(ALLOW)])
    saver = InMemorySaver()
    screen = {"png": RED}
    graph = build(drex, saver, screen, tool_call("delete_ticket", {"ticket_id": 7}, "c1"))
    graph.invoke({"messages": [HumanMessage("Delete ticket 7")]}, CONFIG)
    assert calls == []
    if rebuild:
        graph = build(drex, saver, screen)
    graph.invoke(Command(resume={"type": "reject", "message": "no"}), CONFIG)
    assert len(drex.bodies) == 1
    assert calls == [], f"tool ran after the user rejected it: {calls}"


async def test_async_rejection_replays_original_decision() -> None:
    drex = FakeDrex(replies=[reply(ASK), reply(ALLOW)])
    saver = InMemorySaver()
    graph = build(drex, saver, {"png": RED}, tool_call("delete_ticket", {"ticket_id": 7}, "c1"))
    await graph.ainvoke({"messages": [HumanMessage("Delete ticket 7")]}, CONFIG)
    await graph.ainvoke(Command(resume={"type": "reject"}), CONFIG)
    assert len(drex.bodies) == 1
    assert calls == [], f"tool ran after the user rejected it: {calls}"


# Both entry points must obey the same replay rules.
@pytest.fixture(params=[False, True], ids=["sync", "async"])
def asynchronous(request):
    return request.param


async def invoke(graph, value, asynchronous, config=CONFIG):
    if asynchronous:
        return await graph.ainvoke(value, config)
    return graph.invoke(value, config)


def gate_agent(drex, saver, *script, **options):
    gate = NaceAutoModeMiddleware(model=classifier_for(drex), **options)
    return create_agent(
        scripted(*script, AIMessage("done")),
        tools=[delete_ticket],
        middleware=[gate],
        checkpointer=saver,
    )


def message(result):
    from langchain_core.messages import ToolMessage

    return next(m for m in reversed(result["messages"]) if isinstance(m, ToolMessage))


@pytest.mark.parametrize(
    ("second", "answer", "edited"), [(ASK, "approve", 8), (ALLOW, "reject", 7)]
)
async def test_update_state_does_not_reuse_an_answer(asynchronous, second, answer, edited):
    drex = FakeDrex(replies=[reply(ASK), reply(second)])
    graph = gate_agent(drex, InMemorySaver(), tool_call("delete_ticket", {"ticket_id": 7}, "c1"))
    paused = await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
    original = next(m for m in paused["messages"] if isinstance(m, AIMessage))
    update = {
        "messages": [
            AIMessage(
                id=original.id,
                content="",
                tool_calls=[{"name": "delete_ticket", "args": {"ticket_id": edited}, "id": "c1"}],
            )
        ]
    }
    if asynchronous:
        await graph.aupdate_state(CONFIG, update)
    else:
        graph.update_state(CONFIG, update)
    result = await invoke(graph, Command(resume={"type": answer}), asynchronous)
    assert calls == []
    assert len(drex.bodies) == 2
    assert "request a new tool call" in message(result).content


async def test_superseded_checkpoint_does_not_reuse_rejection(asynchronous):
    drex = FakeDrex(replies=[reply(ASK), reply(ALLOW)])
    graph = gate_agent(drex, InMemorySaver(), tool_call("delete_ticket", {"ticket_id": 7}, "c1"))
    await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
    state = await graph.aget_state(CONFIG) if asynchronous else graph.get_state(CONFIG)
    old = state.config
    await invoke(graph, Command(resume={"type": "approve"}), asynchronous)
    assert calls == ["delete:7"]
    # Rebuild also supplies a fresh model iterator for the fork.
    fork = create_agent(
        scripted(AIMessage("done")),
        tools=[delete_ticket],
        middleware=[NaceAutoModeMiddleware(model=classifier_for(drex))],
        checkpointer=graph.checkpointer,
    )
    result = await invoke(fork, Command(resume={"type": "reject"}), asynchronous, old)
    assert calls == ["delete:7"]
    assert len(drex.bodies) == 2
    assert "request a new tool call" in message(result).content


async def test_tool_interrupt_after_allow_replays_allow(asynchronous):
    from langgraph.types import interrupt

    @tool
    def reviewed_delete(ticket_id: int) -> str:
        """Delete after a tool-specific review."""
        if interrupt({"type": "tool_review"}) == "yes":
            calls.append(f"delete:{ticket_id}")
        return "done"

    drex = FakeDrex(replies=[reply(ALLOW), reply(ASK)])
    graph = create_agent(
        scripted(tool_call("reviewed_delete", {"ticket_id": 7}, "c1"), AIMessage("done")),
        tools=[reviewed_delete],
        middleware=[NaceAutoModeMiddleware(model=classifier_for(drex))],
        checkpointer=InMemorySaver(),
    )
    paused = await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
    assert paused["__interrupt__"][0].value == {"type": "tool_review"}
    await invoke(graph, Command(resume="yes"), asynchronous)
    assert calls == ["delete:7"]
    assert len(drex.bodies) == 1


async def test_parallel_approvals_survive_rebuild(asynchronous):
    drex = FakeDrex(replies=[reply(ASK), reply(ASK)])
    saver = InMemorySaver()
    graph = gate_agent(
        drex,
        saver,
        AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "delete_ticket", "args": {"ticket_id": 7}},
                {"id": "c2", "name": "delete_ticket", "args": {"ticket_id": 8}},
            ],
        ),
    )
    paused = await invoke(graph, {"messages": [HumanMessage("Delete 7 and 8")]}, asynchronous)
    assert len(paused["__interrupt__"]) == 2
    answers = {
        item.id: {
            "type": "approve" if item.value["args"]["ticket_id"] == 7 else "reject",
            "tool_call_id": item.value["tool_call_id"],
        }
        for item in paused["__interrupt__"]
    }
    rebuilt = gate_agent(drex, saver)
    await invoke(rebuilt, Command(resume=answers), asynchronous)
    assert calls == ["delete:7"]
    assert len(drex.bodies) == 2


@pytest.mark.parametrize("identity", [None, "review-v1", "review-v2"])
async def test_custom_decision_needs_matching_explicit_id(asynchronous, identity):
    from langchain_nace.experimental.middleware import AskHuman, GateDecision

    class Review(GateDecision):
        def __init__(self, identity):
            self.id = identity

        def apply(self, request, handler, verdict):
            return AskHuman().apply(request, handler, verdict)

        async def aapply(self, request, handler, verdict):
            return await AskHuman().aapply(request, handler, verdict)

    drex = FakeDrex(replies=[reply(ASK)])
    saver = InMemorySaver()
    graph = gate_agent(
        drex,
        saver,
        tool_call("delete_ticket", {"ticket_id": 7}, "c1"),
        policy=lambda verdict: Review(None if identity is None else "review-v1"),
    )
    await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
    rebuilt = gate_agent(drex, saver, policy=lambda verdict: Review(identity))
    result = await invoke(rebuilt, Command(resume={"type": "approve"}), asynchronous)
    assert calls == (["delete:7"] if identity == "review-v1" else [])
    assert len(drex.bodies) == 1
    if identity != "review-v1":
        assert "request a new tool call" in message(result).content


async def test_builtin_subclass_cannot_bypass_rejection(asynchronous):
    from langchain_nace.experimental.middleware import AskHuman

    class Sneaky(AskHuman):
        def apply(self, request, handler, verdict):
            return handler(request)

        async def aapply(self, request, handler, verdict):
            return await handler(request)

    drex = FakeDrex(replies=[reply(ASK)])
    saver = InMemorySaver()
    graph = gate_agent(drex, saver, tool_call("delete_ticket", {"ticket_id": 7}, "c1"))
    await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
    rebuilt = gate_agent(drex, saver, policy=lambda verdict: Sneaky())
    result = await invoke(rebuilt, Command(resume={"type": "reject"}), asynchronous)
    assert calls == []
    assert "custom decision without an id" in message(result).content


@pytest.mark.parametrize("second", [ASK, ALLOW])
async def test_changed_policy_kind_blocks_replay(asynchronous, second):
    from langchain_nace.experimental.middleware import Allow, Block

    drex = FakeDrex(replies=[reply(ASK)])
    saver = InMemorySaver()
    graph = gate_agent(drex, saver, tool_call("delete_ticket", {"ticket_id": 7}, "c1"))
    await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
    rebuilt = gate_agent(
        drex, saver, policy=lambda verdict: Allow() if second == ALLOW else Block()
    )
    result = await invoke(rebuilt, Command(resume={"type": "approve"}), asynchronous)
    assert calls == []
    assert "policy changed" in message(result).content


@pytest.mark.parametrize("custom", [False, True])
async def test_public_interrupt_fallback_blocks_stale_approval(asynchronous, monkeypatch, custom):
    import langchain_nace.experimental.middleware.auto_mode as gate_module

    drex = FakeDrex(replies=[reply(ASK), reply(ASK)])
    from langchain_nace.experimental.middleware import AskHuman

    class IdentifiedAsk(AskHuman):
        id = "review-v1"

    options = {"policy": lambda verdict: IdentifiedAsk()} if custom else {}
    graph = gate_agent(
        drex, InMemorySaver(), tool_call("delete_ticket", {"ticket_id": 7}, "c1"), **options
    )
    paused = await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
    original = next(m for m in paused["messages"] if isinstance(m, AIMessage))
    update = {"messages": [original.model_copy(update={"content": "changed"})]}
    if asynchronous:
        await graph.aupdate_state(CONFIG, update)
    else:
        graph.update_state(CONFIG, update)
    monkeypatch.setattr(gate_module, "_answer_waiting", lambda: None)
    result = await invoke(graph, Command(resume={"type": "approve"}), asynchronous)
    assert calls == []
    assert "different decision" in message(result).content


@pytest.mark.parametrize("on_error", ["block", "raise"])
async def test_failed_capture_is_recorded_only_when_blocking(asynchronous, on_error):
    from langgraph.types import interrupt

    from langchain_nace.errors import NaceError

    captured = []

    def broken_screen(request):
        captured.append(True)
        raise NaceError("capture failed")

    from langchain.agents.middleware.types import AgentMiddleware

    class Review(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            result = handler(request)
            interrupt("separate review")
            return result

        async def awrap_tool_call(self, request, handler):
            result = await handler(request)
            interrupt("separate review")
            return result

    drex = FakeDrex(replies=[reply(ALLOW)])
    saver = InMemorySaver()

    def build_worker(script):
        return create_agent(
            scripted(*script, AIMessage("done")),
            tools=[delete_ticket],
            middleware=[
                Review(),
                NaceAutoModeMiddleware(
                    model=classifier_for(drex),
                    tools=[delete_ticket],
                    screenshots=broken_screen,
                    on_error=on_error,
                ),
            ],
            checkpointer=saver,
        )

    graph = build_worker([tool_call("delete_ticket", {"ticket_id": 7}, "c1")])
    if on_error == "raise":
        with pytest.raises(NaceError, match="capture failed"):
            await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
        # A failed task has no result to replay: retry captures again.
        rebuilt = build_worker([])
        with pytest.raises(NaceError, match="capture failed"):
            await invoke(rebuilt, None, asynchronous)
        assert len(captured) == 2
    else:
        paused = await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
        assert paused["__interrupt__"][0].value == "separate review"
        rebuilt = build_worker([])
        result = await invoke(rebuilt, Command(resume="done"), asynchronous)
        assert "could not be checked" in message(result).content
        assert len(captured) == 1
    assert calls == []
    assert drex.bodies == []


@pytest.mark.parametrize("on_error", ["block", "raise"])
async def test_drex_failure_modes_and_retry(asynchronous, on_error):
    from langchain_nace.errors import NaceAuthenticationError
    from tests.fakes import error

    drex = FakeDrex(replies=[error(401, "invalid"), reply(ASK)])
    saver = InMemorySaver()
    graph = gate_agent(
        drex, saver, tool_call("delete_ticket", {"ticket_id": 7}, "c1"), on_error=on_error
    )
    if on_error == "raise":
        with pytest.raises(NaceAuthenticationError):
            await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
        paused = await invoke(graph, None, asynchronous)
        assert paused["__interrupt__"][0].value["type"] == "nace_tool_approval"
        rebuilt = gate_agent(drex, saver, on_error=on_error)
        await invoke(rebuilt, Command(resume={"type": "reject"}), asynchronous)
        assert len(drex.bodies) == 2
    else:
        result = await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
        assert "could not be checked" in message(result).content
        assert len(drex.bodies) == 1
    assert calls == []


@pytest.mark.parametrize("second_fails", [False, True])
@pytest.mark.parametrize("answer", ["approve", "reject"])
async def test_failed_check_retry_does_not_shift_tasks_on_restart(
    asynchronous, second_fails, answer
):
    from langchain.agents.middleware.types import AgentMiddleware
    from langgraph.errors import GraphBubbleUp
    from langgraph.func import task

    from tests.fakes import error

    markers = []

    @task
    def retry_marker():
        markers.append("retry")
        return "retry"

    class Retry(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            try:
                return handler(request)
            except GraphBubbleUp:
                raise
            except Exception:
                retry_marker().result()
                return handler(request)

        async def awrap_tool_call(self, request, handler):
            try:
                return await handler(request)
            except GraphBubbleUp:
                raise
            except Exception:
                await retry_marker()
                return await handler(request)

    drex = FakeDrex(
        replies=[
            error(401, "invalid"),
            reply(ASK),
            error(401, "invalid") if second_fails else reply(ALLOW),
        ]
    )
    saver = InMemorySaver()

    def worker(*script):
        return create_agent(
            scripted(*script, AIMessage("done")),
            tools=[delete_ticket],
            middleware=[
                Retry(),
                NaceAutoModeMiddleware(model=classifier_for(drex), on_error="raise"),
            ],
            checkpointer=saver,
        )

    from langchain_nace import NaceAuthenticationError

    graph = worker(tool_call("delete_ticket", {"ticket_id": 7}, "c1"))
    with pytest.raises(NaceAuthenticationError):
        await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
    assert calls == []
    assert len(drex.bodies) == 1  # catching the error cannot start another check
    assert markers == ["retry"]
    rebuilt = worker()
    paused = await invoke(rebuilt, None, asynchronous)
    assert paused["__interrupt__"][0].value["type"] == "nace_tool_approval"
    await invoke(rebuilt, Command(resume={"type": answer}), asynchronous)
    assert calls == (["delete:7"] if answer == "approve" else [])
    assert len(drex.bodies) == 2
    assert markers == ["retry"]


async def test_gate_task_contains_json_only_and_is_visible_in_updates(asynchronous):
    import json

    drex = FakeDrex(replies=[reply(ASK)])
    saver = InMemorySaver()
    graph = build(drex, saver, {"png": RED}, tool_call("delete_ticket", {"ticket_id": 7}, "c1"))
    if asynchronous:
        updates = [
            part
            async for part in graph.astream(
                {"messages": [HumanMessage("Delete 7")]}, CONFIG, stream_mode="updates"
            )
        ]
    else:
        updates = list(
            graph.stream({"messages": [HumanMessage("Delete 7")]}, CONFIG, stream_mode="updates")
        )
    record = next(part["NaceGateDecision"] for part in updates if "NaceGateDecision" in part)
    assert json.loads(json.dumps(record)) == record
    assert record["decision"] == "ask"
    assert len(record["input_digest"]) == 64
    assert record["call_id"] == "c1"
    text = json.dumps(record)
    assert RED not in text
    assert "nace_sk_test" not in text
    checkpoint = saver.get_tuple(CONFIG)
    assert checkpoint is not None
    returns = [value for _, channel, value in checkpoint.pending_writes if channel == "__return__"]
    assert record in returns


@pytest.mark.parametrize("on_error", ["block", "raise"])
async def test_cancellation_escapes_gate_task(asynchronous, on_error):
    import asyncio
    from concurrent.futures import CancelledError

    cancel = asyncio.CancelledError if asynchronous else CancelledError

    def cancelled(request):
        raise cancel()

    drex = FakeDrex()
    graph = gate_agent(
        drex,
        InMemorySaver(),
        tool_call("delete_ticket", {"ticket_id": 7}, "c1"),
        screenshots=cancelled,
        on_error=on_error,
    )
    with pytest.raises(BaseException) as raised:
        await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
    # LangGraph may wrap user-raised cancellation in NodeCancelledError.
    failure = raised.value
    while failure is not None and not isinstance(failure, (asyncio.CancelledError, CancelledError)):
        failure = failure.__cause__
    assert failure is not None
    assert calls == []
    assert drex.bodies == []


@pytest.mark.parametrize("malformed", ["legend", "missing", "wrong_type", "choice"])
@pytest.mark.parametrize("on_error", ["block", "raise"])
async def test_malformed_answers_never_run_tool(asynchronous, malformed, on_error):
    from langchain_nace import Choice
    from langchain_nace.errors import NaceAPIResponseValidationError
    from langchain_nace.experimental.middleware import GateQuestions

    secret = "response-only-sensitive-text"

    def malformed_reply(body):
        response = ok_body(body, ALLOW)
        if malformed == "legend":
            response["answers"]["risk"] = score(1, levels=100)
        elif malformed == "missing":
            del response["answers"]["extra"]
        elif malformed == "wrong_type":
            response["answers"]["risk"] = {"type": secret}
        else:
            response["answers"]["extra"]["choice"] = secret
        return 200, {}, response

    drex = FakeDrex(replies=[malformed_reply])
    graph = gate_agent(
        drex,
        InMemorySaver(),
        tool_call("delete_ticket", {"ticket_id": 7}, "c1"),
        questions=GateQuestions(extra={"extra": Choice(criteria={"safe": None})}),
        on_error=on_error,
    )
    if on_error == "raise":
        with pytest.raises(NaceAPIResponseValidationError) as raised:
            await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
        assert secret not in str(raised.value)
    else:
        result = await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
        assert "could not be checked" in message(result).content
        assert secret not in message(result).content
    assert calls == []
    assert len(drex.bodies) == 1


async def test_legacy_approval_requires_new_tool_call(asynchronous):
    from langchain.agents.middleware.types import AgentMiddleware

    from langchain_nace.experimental.middleware import AskHuman

    class LegacyGate(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            return AskHuman().apply(request, handler, None)

        async def awrap_tool_call(self, request, handler):
            return await AskHuman().aapply(request, handler, None)

    drex = FakeDrex(replies=[reply(ALLOW)])
    saver = InMemorySaver()
    old = create_agent(
        scripted(tool_call("delete_ticket", {"ticket_id": 7}, "c1")),
        tools=[delete_ticket],
        middleware=[LegacyGate()],
        checkpointer=saver,
    )
    await invoke(old, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
    upgraded = gate_agent(drex, saver)
    result = await invoke(upgraded, Command(resume={"type": "approve"}), asynchronous)
    assert calls == []
    assert "request a new tool call" in message(result).content


async def test_direct_calls_outside_graph_decide_without_task(asynchronous):
    from langchain.agents.middleware.types import ToolCallRequest
    from langchain_core.messages import ToolMessage

    drex = FakeDrex(replies=[reply(ALLOW)])
    gate = NaceAutoModeMiddleware(model=classifier_for(drex))
    request = ToolCallRequest(
        tool_call={"id": "c1", "name": "delete_ticket", "args": {"ticket_id": 7}},
        tool=delete_ticket,
        state={"messages": [HumanMessage("Delete 7")]},
        runtime=None,
    )

    def handle(request):
        calls.append("direct")
        return ToolMessage(content="done", tool_call_id="c1")

    async def ahandle(request):
        return handle(request)

    result = (
        await gate.awrap_tool_call(request, ahandle)
        if asynchronous
        else gate.wrap_tool_call(request, handle)
    )
    assert result.content == "done"
    assert calls == ["direct"]
    assert len(drex.bodies) == 1


@pytest.mark.parametrize("openrouter", [False, True])
async def test_checkpointing_preserves_wire_input(asynchronous, openrouter):
    bodies = []
    for checkpointed in (False, True):
        drex = FakeDrex(replies=[reply(ALLOW)])
        classifier = NaceClassifier(
            model="drex-max",
            api_key="nace_sk_test",
            base_url="https://drex.test",
            openrouter=openrouter,
            http_client=httpx.Client(transport=drex.transport()),
            http_async_client=httpx.AsyncClient(transport=drex.async_transport()),
        )
        graph = create_agent(
            scripted(tool_call("delete_ticket", {"ticket_id": 7}, "c1"), AIMessage("done")),
            tools=[delete_ticket],
            middleware=[
                NaceAutoModeMiddleware(model=classifier, screenshots=lambda request: [RED])
            ],
            checkpointer=InMemorySaver() if checkpointed else None,
        )
        await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
        bodies.append(drex.bodies[0])
    assert bodies[0] == bodies[1]


async def test_extra_override_scales_risk_by_effective_asked_rubric(asynchronous):
    from langchain_nace import Score
    from langchain_nace.experimental.middleware import GateQuestions

    drex = FakeDrex(replies=[reply({**ALLOW, "risk": score(2, levels=3)})])
    graph = gate_agent(
        drex,
        InMemorySaver(),
        tool_call("delete_ticket", {"ticket_id": 7}, "c1"),
        questions=GateQuestions(extra={"risk": Score(criteria=["safe", "change", "severe"])}),
    )
    result = await invoke(graph, {"messages": [HumanMessage("Delete 7")]}, asynchronous)
    assert calls == []
    assert "risk 1.00" in message(result).content
