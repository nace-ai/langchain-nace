"""Tool approvals compare identity fields rather than export representations."""

from dataclasses import dataclass, field

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field, PrivateAttr, computed_field, model_serializer

from langchain_nace.experimental.middleware import NaceAutoModeMiddleware
from tests.agents import scripted, tool_call
from tests.fakes import FakeDrex
from tests.unit_tests.experimental.middleware.test_auto_mode_lifecycle import (
    ASK,
    classifier_for,
    reply,
)


@pytest.mark.parametrize(
    ("asynchronous", "kind", "changed"),
    [
        (False, "excluded", True),
        (True, "private", True),
        (False, "dynamic_private", True),
        (False, "computed", True),
        (False, "dataclass", True),
        (False, "computed", False),
    ],
)
async def test_context_identity_fields_cannot_be_hidden_from_digest(asynchronous, kind, changed):
    active = {"tenant": "A"}
    if kind == "excluded":

        class Context(BaseModel):
            tenant: str = Field(exclude=True)

        before, after = Context(tenant="A"), Context(tenant="B")
    elif kind == "private":

        class Context(BaseModel):
            _tenant: str = PrivateAttr()

        before, after = Context(), Context()
        before._tenant, after._tenant = "A", "B"
    elif kind == "dynamic_private":

        class Context(BaseModel):
            pass

        before, after = Context(), Context()
        before._tenant, after._tenant = "A", "B"
    elif kind == "computed":

        class Context(BaseModel):
            @computed_field
            @property
            def tenant(self) -> str:
                return active["tenant"]

            @model_serializer
            def export(self) -> dict:
                return {}

        before = after = Context()
    else:

        @dataclass
        class Context:
            tenant: str = field(repr=False, compare=False)

        before, after = Context("A"), Context("B")
    ran = []

    @tool
    def act() -> str:
        """Act within the current tenant."""
        ran.append(True)
        return "done"

    graph = create_agent(
        scripted(tool_call("act", {}, "a"), AIMessage("done")),
        tools=[act],
        middleware=[NaceAutoModeMiddleware(model=classifier_for(FakeDrex(replies=[reply(ASK)])))],
        checkpointer=InMemorySaver(),
        context_schema=type(before),
    )
    config = {"configurable": {"thread_id": "identity"}}
    if asynchronous:
        await graph.ainvoke({"messages": [HumanMessage("Act")]}, config, context=before)
    else:
        graph.invoke({"messages": [HumanMessage("Act")]}, config, context=before)
    if not changed:
        after = before
    active["tenant"] = "B" if changed else "A"
    result = (
        await graph.ainvoke(Command(resume={"type": "approve"}), config, context=after)
        if asynchronous
        else graph.invoke(Command(resume={"type": "approve"}), config, context=after)
    )
    assert ran == ([] if changed else [True])
    if changed:
        assert "request a new tool call" in result["messages"][-2].content


@pytest.mark.parametrize("container", ["mapping", "model", "dataclass", "set", "frozenset"])
def test_context_container_identity_is_preserved(container):
    before, after = ({1: "A", "1": "B"}, {1: "C", "1": "B"})
    if container == "model":

        class Context(BaseModel):
            tenants: dict

        before, after = (Context(tenants=before), Context(tenants=after))
    elif container == "dataclass":

        @dataclass
        class Context:
            tenants: dict

        before, after = (Context(before), Context(after))
    elif container == "set":
        before, after = ({1, "1"}, ["1", 1])
    elif container == "frozenset":
        before, after = (frozenset({1, "1"}), {1, "1"})
    ran = []

    @tool
    def act() -> str:
        """Act within the current tenant collection."""
        ran.append(True)
        return "done"

    graph = create_agent(
        scripted(tool_call("act", {}, "a"), AIMessage("done")),
        tools=[act],
        middleware=[NaceAutoModeMiddleware(model=classifier_for(FakeDrex(replies=[reply(ASK)])))],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "container-identity"}}
    graph.invoke({"messages": [HumanMessage("Act")]}, config, context=before)
    result = graph.invoke(Command(resume={"type": "approve"}), config, context=after)
    assert ran == []
    assert "request a new tool call" in result["messages"][-2].content


def test_context_mapping_and_set_order_does_not_change_approval():
    members = [(1, "A"), ("1", 2), (False, "B")]
    before = {"tenants": {1: "A", "1": "B"}, "members": set(members), "frozen": frozenset(members)}
    after = {
        "frozen": frozenset(reversed(members)),
        "members": set(reversed(members)),
        "tenants": {"1": "B", 1: "A"},
    }
    ran = []

    @tool
    def act() -> str:
        """Act within an unchanged collection."""
        ran.append(True)
        return "done"

    graph = create_agent(
        scripted(tool_call("act", {}, "a"), AIMessage("done")),
        tools=[act],
        middleware=[NaceAutoModeMiddleware(model=classifier_for(FakeDrex(replies=[reply(ASK)])))],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "container-order"}}
    graph.invoke({"messages": [HumanMessage("Act")]}, config, context=before)
    graph.invoke(Command(resume={"type": "approve"}), config, context=after)
    assert ran == [True]
