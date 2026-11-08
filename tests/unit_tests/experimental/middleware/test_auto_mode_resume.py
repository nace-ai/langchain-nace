"""Tool approvals across interrupts, shared tasks, and retries."""

import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from langchain_nace.experimental.middleware import NaceAutoModeMiddleware
from tests.agents import scripted, tool_call
from tests.fakes import FakeDrex, ok_body
from tests.unit_tests.experimental.middleware.test_auto_mode_lifecycle import (
    ALLOW,
    ASK,
    CONFIG,
    RED,
    classifier_for,
    invoke,
    reply,
)


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def asynchronous(request):
    return request.param


class Retry(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except ValueError:
            return handler(request)

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except ValueError:
            return await handler(request)


@pytest.mark.parametrize("unknown_counter", [False, True])
async def test_replayed_allow_cannot_skip_later_rejection(
    asynchronous, monkeypatch, unknown_counter
):
    import langchain_nace.experimental.middleware.auto_mode as module

    attempts = []

    @tool
    def act() -> str:
        """Perform an action that initially fails."""
        attempts.append(True)
        if len(attempts) == 1:
            raise ValueError("transient")
        return "ran"

    drex = FakeDrex(replies=[reply(ALLOW), reply(ASK)])
    gate = NaceAutoModeMiddleware(model=classifier_for(drex))
    # Construct the unsafe checkpoint produced before decision reuse was added.
    # Resume uses the real cache and must still honor the later rejection.
    native_entry_for = module._entry_for
    legacy = True

    def legacy_entry(gate, request):
        entry = native_entry_for(gate, request)
        return module._GateEntry(entry.scope) if legacy else entry

    monkeypatch.setattr(module, "_entry_for", legacy_entry)
    graph = create_agent(
        scripted(tool_call("act", {}, "a"), AIMessage("done")),
        tools=[act],
        middleware=[Retry(), gate],
        checkpointer=InMemorySaver(),
    )
    paused = await invoke(graph, {"messages": [HumanMessage("Act")]}, asynchronous)
    assert paused["__interrupt__"][0].value["tool_call_id"] == "a"
    legacy = False
    if unknown_counter:
        monkeypatch.setattr(module, "_answer_waiting", lambda: None)
    await invoke(graph, Command(resume={"type": "reject"}), asynchronous)
    assert attempts == [True]
    assert len(drex.bodies) == 2


@pytest.mark.parametrize("bound", [False, True])
async def test_shared_toolnode_cannot_consume_another_calls_approval(asynchronous, bound):
    ran = []

    @tool
    def act(target: str) -> str:
        """Act on a target."""
        ran.append(target)
        return "done"

    drex = FakeDrex(replies=[reply(ASK), reply(ASK)])
    gate = NaceAutoModeMiddleware(model=classifier_for(drex))
    builder = StateGraph(MessagesState)
    builder.add_node("choose", lambda state: {})
    builder.add_node(
        "tools",
        ToolNode([act], wrap_tool_call=gate.wrap_tool_call, awrap_tool_call=gate.awrap_tool_call),
    )
    builder.add_edge(START, "choose")
    builder.add_conditional_edges("choose", tools_condition)
    builder.add_edge("tools", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    paused = await invoke(
        graph,
        {
            "messages": [
                HumanMessage("Act on B only"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "a", "name": "act", "args": {"target": "A"}},
                        {"id": "b", "name": "act", "args": {"target": "B"}},
                    ],
                ),
            ]
        },
        asynchronous,
    )
    assert paused["__interrupt__"]
    answer = {"type": "approve", **({"tool_call_id": "b"} if bound else {})}
    # In a shared task the next interrupt slot belongs to A, even if a client
    # intended to approve B. Missing or mismatched binding must never run A.
    result = await invoke(graph, Command(resume=answer), asynchronous)
    assert "A" not in ran
    if not bound:
        assert ran == []
    assert result is not None


@pytest.mark.parametrize("scenario", ["outer_interrupt", "stacked_gates", "approved_retry"])
async def test_consumed_null_resume_does_not_block_normal_flow(asynchronous, scenario):
    ran = []

    @tool
    def act() -> str:
        """Act after approval."""
        ran.append(True)
        if scenario == "approved_retry" and len(ran) == 1:
            raise ValueError("transient")
        return "done"

    class Before(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            interrupt("outer")
            return handler(request)

        async def awrap_tool_call(self, request, handler):
            interrupt("outer")
            return await handler(request)

    first = FakeDrex(replies=[reply(ASK if scenario != "outer_interrupt" else ALLOW), reply(ALLOW)])
    second = FakeDrex(replies=[reply(ASK)])
    gate = NaceAutoModeMiddleware(model=classifier_for(first))

    class InnerGate(NaceAutoModeMiddleware):
        pass

    middleware = {
        "outer_interrupt": [Before(), gate],
        "stacked_gates": [gate, InnerGate(model=classifier_for(second))],
        "approved_retry": [ToolRetryMiddleware(max_retries=1, initial_delay=0), gate],
    }[scenario]
    graph = create_agent(
        scripted(tool_call("act", {}, "a"), AIMessage("done")),
        tools=[act],
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )
    await invoke(graph, {"messages": [HumanMessage("Act")]}, asynchronous)
    result = await invoke(graph, Command(resume={"type": "approve"}), asynchronous)
    if scenario == "stacked_gates":
        assert result["__interrupt__"][0].value["type"] == "nace_tool_approval"
        assert ran == []
        await invoke(graph, Command(resume={"type": "approve"}), asynchronous)
    assert len(ran) == (2 if scenario == "approved_retry" else 1)


@pytest.mark.parametrize("failure", [TypeError, DeprecationWarning])
def test_counter_peek_never_uses_reduce(monkeypatch, failure):
    import langchain_nace.experimental.middleware.auto_mode as module

    class Counter(itertools.count):
        def __reduce__(self):
            raise failure("not supported")

        def __repr__(self):
            return "count(0)"

    counter = Counter()
    pad = SimpleNamespace(
        interrupt_counter=counter.__next__,
        resume=[{"type": "reject"}],
        get_null_resume=lambda consume: None,
    )
    monkeypatch.setattr(
        module, "get_config", lambda: {"configurable": {"__pregel_scratchpad": pad}}
    )
    assert module._answer_waiting() is True
    assert next(counter) == 0


async def test_server_echoes_never_reach_decision_record(asynchronous):
    observed = []

    @tool
    def act() -> str:
        """Act safely."""
        return "done"

    def echo(body):
        response = ok_body(body, ASK)
        response["images"] = [RED]
        response["headers"] = {"authorization": "echoed-provider-key"}
        response["usage"]["echoed"] = RED
        response["answers"]["risk"]["echoed"] = RED
        response["answers"]["label"]["probabilities"][RED] = 0.0
        response["answers"]["future"] = {"type": RED, "echoed": RED}
        return 200, {}, response

    from langchain_nace import Choice
    from langchain_nace.experimental.middleware import GateQuestions, ThresholdPolicy

    def policy(verdict):
        observed.append(verdict.response)
        return ThresholdPolicy()(verdict)

    gate = NaceAutoModeMiddleware(
        model=classifier_for(FakeDrex(replies=[echo])),
        policy=policy,
        questions=GateQuestions(
            extra={"label": Choice(criteria={"safe": None}), "future": {"type": "future"}}
        ),
    )
    graph = create_agent(
        scripted(tool_call("act", {}, "a")),
        tools=[act],
        middleware=[gate],
        checkpointer=InMemorySaver(),
    )
    value = {"messages": [HumanMessage("Act")]}
    updates = (
        [part async for part in graph.astream(value, CONFIG, stream_mode="updates")]
        if asynchronous
        else list(graph.stream(value, CONFIG, stream_mode="updates"))
    )
    record = next(part["NaceGateDecision"] for part in updates if "NaceGateDecision" in part)
    assert RED not in record["response"]
    assert "echoed-provider-key" not in record["response"]
    assert observed[0].raw == {}
    assert observed[0].usage.model_extra == {}


def test_runtime_context_binding_is_documented():
    readme = (Path(__file__).parents[4] / "README.md").read_text()
    assert "entire `request.state`, `runtime.context`" in readme
    assert "runtime `configurable` values" in readme
