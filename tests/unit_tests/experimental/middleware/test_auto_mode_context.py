"""Decision reuse and complete execution-context binding."""

import tomllib
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Command, Send

from langchain_nace.experimental.middleware import Allow, AskHuman, Block, NaceAutoModeMiddleware
from tests.agents import scripted, tool_call
from tests.fakes import FakeDrex
from tests.unit_tests.experimental.middleware.test_auto_mode_lifecycle import (
    ALLOW,
    ASK,
    classifier_for,
    invoke,
    reply,
)
from tests.unit_tests.experimental.middleware.test_auto_mode_resume import Retry


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def asynchronous(request):
    return request.param


@pytest.mark.parametrize("first", ["custom_allow", "ask"])
async def test_retry_reuses_first_decision_and_approval(asynchronous, first):
    attempts = []

    @tool
    def act() -> str:
        """Act, retrying one transient failure."""
        attempts.append(True)
        if len(attempts) == 1:
            raise ValueError("transient")
        return "done"

    class LoggedAllow(Allow):
        id = "la"

    drex = FakeDrex(replies=[reply(ALLOW if first == "custom_allow" else ASK), reply(ASK)])
    gate = NaceAutoModeMiddleware(
        model=classifier_for(drex),
        policy=lambda verdict: LoggedAllow() if verdict.risk == 0 else AskHuman(),
    )
    graph = create_agent(
        scripted(tool_call("act", {}, "a"), AIMessage("done")),
        tools=[act],
        middleware=[Retry(), gate],
        checkpointer=InMemorySaver(),
    )
    result = await invoke(graph, {"messages": [HumanMessage("Act")]}, asynchronous)
    if first == "ask":
        assert result["__interrupt__"]
        result = await invoke(graph, Command(resume={"type": "approve"}), asynchronous)
    assert "__interrupt__" not in result  # no second approval for the retry
    assert len(attempts) == 2
    assert len(drex.bodies) == 1


def test_custom_execution_decision_has_resume_defense(monkeypatch):
    import langchain_nace.experimental.middleware.auto_mode as module

    class LoggedAllow(Allow):
        id = "la"

    monkeypatch.setattr(module, "_answer_waiting", lambda: True)
    monkeypatch.setattr(module, "_approval_answer_waiting", lambda **kwargs: True)
    assert isinstance(module._protect_resume(LoggedAllow(), fresh=False), Block)


@pytest.mark.parametrize("layout", ["send", "trailing_message"])
async def test_shared_calls_are_bound_even_in_send_and_after_appended_message(asynchronous, layout):
    ran = []

    @tool
    def act(target: str) -> str:
        """Act on a target."""
        ran.append(target)
        return "done"

    gate = NaceAutoModeMiddleware(model=classifier_for(FakeDrex(replies=[reply(ASK)] * 2)))
    builder = StateGraph(MessagesState)
    builder.add_node(
        "tools",
        ToolNode([act], wrap_tool_call=gate.wrap_tool_call, awrap_tool_call=gate.awrap_tool_call),
    )
    if layout == "send":
        builder.add_conditional_edges(START, lambda state: [Send("tools", state)])
    else:
        builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    messages = [
        HumanMessage("Act on B"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "a", "name": "act", "args": {"target": "A"}},
                {"id": "b", "name": "act", "args": {"target": "B"}},
            ],
        ),
    ]
    if layout == "trailing_message":
        messages.append(HumanMessage("Appended by middleware"))
    result = await invoke(graph, {"messages": messages}, asynchronous)
    assert all(item.value["requires_tool_call_id"] for item in result["__interrupt__"])
    await invoke(graph, Command(resume={"type": "approve", "tool_call_id": "b"}), asynchronous)
    assert "A" not in ran


@pytest.mark.parametrize("changed", ["context", "state", "configurable", "secret_context"])
async def test_approval_cannot_cross_state_context_or_configurable(asynchronous, changed):
    ran = []

    @tool
    def act() -> str:
        """Act within a tenant."""
        ran.append(True)
        return "done"

    class State(MessagesState):
        tenant: str

    gate = NaceAutoModeMiddleware(model=classifier_for(FakeDrex(replies=[reply(ASK)])))
    builder = StateGraph(State, context_schema=dict)
    builder.add_node(
        "tools",
        ToolNode([act], wrap_tool_call=gate.wrap_tool_call, awrap_tool_call=gate.awrap_tool_call),
    )
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "tenant", "tenant": "A"}}
    initial = {"messages": [tool_call("act", {}, "a")], "tenant": "A"}
    from pydantic import SecretStr

    before_context = {"tenant": SecretStr("A") if changed == "secret_context" else "A"}
    if asynchronous:
        await graph.ainvoke(initial, config, context=before_context)
    else:
        graph.invoke(initial, config, context=before_context)
    command = Command(
        resume={"type": "approve"}, update={"tenant": "B"} if changed == "state" else None
    )
    context = {
        "tenant": SecretStr("B")
        if changed == "secret_context"
        else "B"
        if changed == "context"
        else "A"
    }
    if changed == "configurable":
        config = {"configurable": {"thread_id": "tenant", "tenant": "B"}}
    result = (
        await graph.ainvoke(command, config, context=context)
        if asynchronous
        else graph.invoke(command, config, context=context)
    )
    assert ran == []
    assert "request a new tool call" in result["messages"][-1].content


def test_experimental_extra_requires_supported_langgraph():
    project = tomllib.loads((Path(__file__).parents[4] / "pyproject.toml").read_text())
    assert "langgraph>=1.2.16" in project["project"]["optional-dependencies"]["experimental"]


async def test_toolnode_in_runnable_context_without_graph(asynchronous):
    ran = []

    @tool
    def act() -> str:
        """Act safely."""
        ran.append(True)
        return "done"

    drex = FakeDrex(replies=[reply(ALLOW)])
    gate = NaceAutoModeMiddleware(model=classifier_for(drex))
    node = ToolNode([act], wrap_tool_call=gate.wrap_tool_call, awrap_tool_call=gate.awrap_tool_call)
    runnable = RunnableLambda(
        lambda state: node.invoke(state, runtime=Runtime()),
        afunc=lambda state: node.ainvoke(state, runtime=Runtime()),
    )
    state = {"messages": [tool_call("act", {}, "a")]}
    if asynchronous:
        await runnable.ainvoke(state)
    else:
        runnable.invoke(state)
    assert ran == [True]


async def test_outer_allow_does_not_claim_inner_gates_answer(asynchronous):
    ran = []

    @tool
    def act() -> str:
        """Act after the inner approval."""
        ran.append(True)
        return "done"

    class Inner(NaceAutoModeMiddleware):
        pass

    graph = create_agent(
        scripted(tool_call("act", {}, "a"), AIMessage("done")),
        tools=[act],
        middleware=[
            NaceAutoModeMiddleware(model=classifier_for(FakeDrex(replies=[reply(ALLOW)]))),
            Inner(model=classifier_for(FakeDrex(replies=[reply(ASK)]))),
        ],
        checkpointer=InMemorySaver(),
    )
    await invoke(graph, {"messages": [HumanMessage("Act")]}, asynchronous)
    result = await invoke(graph, Command(resume={"type": "approve"}), asynchronous)
    assert ran == [True]
    assert "__interrupt__" not in result


@pytest.mark.parametrize("answer", ["reject", "edit"])
async def test_repeat_handler_keeps_rejection_or_edited_arguments(asynchronous, answer):
    from langchain.agents.middleware.types import AgentMiddleware

    ran = []

    @tool
    def act(target: int) -> str:
        """Act on an approved target."""
        ran.append(target)
        return "done"

    class Twice(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            handler(request)
            return handler(request)

        async def awrap_tool_call(self, request, handler):
            await handler(request)
            return await handler(request)

    drex = FakeDrex(replies=[reply(ASK), reply(ALLOW)])
    graph = create_agent(
        scripted(tool_call("act", {"target": 1}, "a"), AIMessage("done")),
        tools=[act],
        middleware=[Twice(), NaceAutoModeMiddleware(model=classifier_for(drex))],
        checkpointer=InMemorySaver(),
    )
    await invoke(graph, {"messages": [HumanMessage("Act")]}, asynchronous)
    result = await invoke(
        graph, Command(resume={"type": answer, "args": {"target": 9}}), asynchronous
    )
    assert "__interrupt__" not in result
    assert ran == ([] if answer == "reject" else [9, 9])
    assert len(drex.bodies) == 1


async def test_resume_latest_explicit_checkpoint_with_structured_context(asynchronous):
    from dataclasses import dataclass

    from pydantic import BaseModel

    class Tenant(BaseModel):
        name: str

    @dataclass
    class Context:
        tenant: Tenant

    ran = []

    @tool
    def act() -> str:
        """Act under an unchanged structured context."""
        ran.append(True)
        return "done"

    drex = FakeDrex(replies=[reply(ASK)])
    graph = create_agent(
        scripted(tool_call("act", {}, "a"), AIMessage("done")),
        tools=[act],
        middleware=[NaceAutoModeMiddleware(model=classifier_for(drex))],
        checkpointer=InMemorySaver(),
        context_schema=Context,
    )
    config = {"configurable": {"thread_id": "explicit"}}
    if asynchronous:
        await graph.ainvoke(
            {"messages": [HumanMessage("Act")]}, config, context=Context(Tenant(name="A"))
        )
        checkpoint = (await graph.aget_state(config)).config
        await graph.ainvoke(
            Command(resume={"type": "approve"}), checkpoint, context=Context(Tenant(name="A"))
        )
    else:
        graph.invoke({"messages": [HumanMessage("Act")]}, config, context=Context(Tenant(name="A")))
        checkpoint = graph.get_state(config).config
        graph.invoke(
            Command(resume={"type": "approve"}), checkpoint, context=Context(Tenant(name="A"))
        )
    assert ran == [True]
    assert len(drex.bodies) == 1


async def test_failed_check_is_not_retried_at_a_new_task_position(asynchronous):
    from langchain_nace import NaceAuthenticationError
    from tests.fakes import error

    ran = []

    @tool
    def act() -> str:
        """Act only after the check succeeds."""
        ran.append(True)
        return "done"

    drex = FakeDrex(replies=[error(401, "invalid"), reply(ALLOW)])
    from langchain.agents.middleware.types import AgentMiddleware

    class RetryCheck(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            try:
                return handler(request)
            except NaceAuthenticationError:
                return handler(request)

        async def awrap_tool_call(self, request, handler):
            try:
                return await handler(request)
            except NaceAuthenticationError:
                return await handler(request)

    graph = create_agent(
        scripted(tool_call("act", {}, "a"), AIMessage("done")),
        tools=[act],
        middleware=[
            RetryCheck(),
            NaceAutoModeMiddleware(model=classifier_for(drex), on_error="raise"),
        ],
        checkpointer=InMemorySaver(),
    )

    async def run():
        return await invoke(graph, {"messages": [HumanMessage("Act")]}, asynchronous)

    with pytest.raises(NaceAuthenticationError):
        await run()
    assert ran == []
    assert len(drex.bodies) == 1
