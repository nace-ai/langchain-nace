"""Approval bindings when a gate chain changes between workers."""

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from langchain_nace.experimental.middleware import NaceAutoModeMiddleware
from tests.agents import scripted, tool_call
from tests.fakes import FakeDrex
from tests.unit_tests.experimental.middleware.test_auto_mode_lifecycle import (
    ALLOW,
    ASK,
    CONFIG,
    classifier_for,
    invoke,
    reply,
)


@pytest.mark.parametrize(
    ("asynchronous", "answer", "earlier_interrupt"),
    [(False, "approve", False), (True, "reject", True)],
)
async def test_removing_outer_gate_cannot_reuse_its_allow(asynchronous, answer, earlier_interrupt):
    ran = []

    @tool
    def act() -> str:
        """Act only under the gate chain the person saw."""
        ran.append(True)
        return "done"

    class Inner(NaceAutoModeMiddleware):
        pass

    saver = InMemorySaver()
    outer = NaceAutoModeMiddleware(model=classifier_for(FakeDrex(replies=[reply(ALLOW)])))
    inner = Inner(model=classifier_for(FakeDrex(replies=[reply(ASK)])))
    first = create_agent(
        scripted(tool_call("act", {}, "a"), AIMessage("done")),
        tools=[act],
        middleware=[outer, inner],
        checkpointer=saver,
    )
    paused = await invoke(first, {"messages": [HumanMessage("Act")]}, asynchronous)
    assert paused["__interrupt__"][0].value["nace_gate_id"] == "gate:1"
    assert paused["__interrupt__"][0].value["nace_gate_count"] == 2
    checkpoint = saver.get_tuple(CONFIG)
    assert checkpoint is not None
    records = [value for _, channel, value in checkpoint.pending_writes if channel == "__return__"]
    assert {(r["gate_scope"], r["gate_count"]) for r in records} == {
        ("gate:0", 1),
        ("gate:1", 2),
    }

    class Before(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            interrupt("another question")
            return handler(request)

        async def awrap_tool_call(self, request, handler):
            interrupt("another question")
            return await handler(request)

    rebuilt = create_agent(
        scripted(AIMessage("done")),
        tools=[act],
        middleware=[Before(), inner] if earlier_interrupt else [inner],
        checkpointer=saver,
    )
    result = await invoke(rebuilt, Command(resume={"type": answer}), asynchronous)
    assert ran == []
    assert "request a new tool call" in result["messages"][-2].content
