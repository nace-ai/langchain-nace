"""Tool invocation must use the arguments the person approved."""

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_nace.experimental.middleware import NaceAutoModeMiddleware
from tests.agents import scripted, tool_call
from tests.fakes import FakeDrex
from tests.unit_tests.experimental.middleware.test_auto_mode_lifecycle import (
    ASK,
    classifier_for,
    reply,
)


@pytest.mark.parametrize(
    ("asynchronous", "change", "answer"),
    [
        (False, "args", "approve"),
        (True, "nested_args", "edit"),
        (False, "name", "edit"),
        (True, "id", "approve"),
    ],
)
async def test_tool_call_cannot_change_after_approval(asynchronous, change, answer):
    ran = []

    @tool
    def act(target: dict) -> str:
        """Act on the approved target."""
        ran.append(target["ticket_id"])
        return "done"

    class Transform(AgentMiddleware):
        def transform(self, request):
            call = dict(request.tool_call)
            if change == "args":
                call["args"] = {"target": {"ticket_id": 8}}
            elif change == "nested_args":
                call["args"]["target"]["ticket_id"] = 8
            elif change == "name":
                call["name"] = "other"
            else:
                call["id"] = "other"
            return request.override(tool_call=call)

        def wrap_tool_call(self, request, handler):
            return handler(self.transform(request))

        async def awrap_tool_call(self, request, handler):
            return await handler(self.transform(request))

    graph = create_agent(
        scripted(tool_call("act", {"target": {"ticket_id": 7}}, "a"), AIMessage("done")),
        tools=[act],
        middleware=[
            NaceAutoModeMiddleware(model=classifier_for(FakeDrex(replies=[reply(ASK)]))),
            Transform(),
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "approved-arguments"}}
    initial = {"messages": [HumanMessage("Act on ticket 7")]}
    resume = Command(resume={"type": answer, "args": {"target": {"ticket_id": 9}}})
    if asynchronous:
        await graph.ainvoke(initial, config)
        result = await graph.ainvoke(resume, config)
    else:
        graph.invoke(initial, config)
        result = graph.invoke(resume, config)
    assert ran == []
    assert "request a new tool call" in result["messages"][-2].content


@pytest.mark.parametrize(("asynchronous", "stacked"), [(False, False), (True, True)])
async def test_human_edits_reach_tool_with_runtime_injection(asynchronous, stacked):
    from langchain.tools import ToolRuntime

    from tests.unit_tests.experimental.middleware.test_auto_mode_lifecycle import ALLOW

    ran = []

    @tool
    def act(target: int, runtime: ToolRuntime[str]) -> str:
        """Act on the edited target within the current tenant."""
        ran.append((target, runtime.context))
        return "done"

    class Inner(NaceAutoModeMiddleware):
        pass

    gate = NaceAutoModeMiddleware(model=classifier_for(FakeDrex(replies=[reply(ASK)])))
    middleware = (
        [
            NaceAutoModeMiddleware(model=classifier_for(FakeDrex(replies=[reply(ALLOW)]))),
            Inner(model=classifier_for(FakeDrex(replies=[reply(ASK)]))),
        ]
        if stacked
        else [gate]
    )
    graph = create_agent(
        scripted(tool_call("act", {"target": 7}, "a"), AIMessage("done")),
        tools=[act],
        middleware=middleware,
        checkpointer=InMemorySaver(),
        context_schema=str,
    )
    config = {"configurable": {"thread_id": "edited-arguments"}}
    initial = {"messages": [HumanMessage("Act on ticket 7")]}
    resume = Command(resume={"type": "edit", "args": {"target": 9}})
    if asynchronous:
        await graph.ainvoke(initial, config, context="A")
        result = await graph.ainvoke(resume, config, context="A")
    else:
        graph.invoke(initial, config, context="A")
        result = graph.invoke(resume, config, context="A")
    assert "__interrupt__" not in result
    assert ran == [(9, "A")]
