"""LangGraph cookbook: route tickets with Drex, ask a person when Drex isn't sure.

`nace_conditional_edge` asks Drex which team owns a ticket and follows the label;
below `min_confidence` it goes to `human_review`, which pauses the graph with
`interrupt()` until a person answers.

    export NACE_API_KEY=nace_sk_...
    uv run python examples/langgraph_router.py
"""

from __future__ import annotations

import warnings
from typing import Any, TypedDict

from langchain_core._api import LangChainBetaWarning
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langchain_nace import Choice, NaceClassifier, nace_conditional_edge

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

TEAMS = Choice(
    instructions="Which team should own this support ticket?",
    criteria={
        "billing": "Charges, invoices, refunds, plans.",
        "technical": "Bugs, outages, errors, integrations.",
        "sales": "Pricing questions, new purchases, upgrades.",
    },
)


class Ticket(TypedDict, total=False):
    text: str
    team: str


def build(min_confidence: float = 0.7) -> Any:
    graph = StateGraph(Ticket)
    graph.add_node("triage", lambda t: {})
    for team in TEAMS.criteria:
        graph.add_node(team, lambda t, team=team: {"team": team})
        graph.add_edge(team, END)
    graph.add_node("human_review", lambda t: {"team": interrupt({"ticket": t["text"]})})
    graph.add_edge("human_review", END)
    graph.add_edge(START, "triage")
    edge = nace_conditional_edge(
        NaceClassifier(), TEAMS, state=lambda t: t["text"], min_confidence=min_confidence
    )
    graph.add_conditional_edges("triage", edge, [*TEAMS.criteria, "human_review"])
    return graph.compile(checkpointer=InMemorySaver())


def main() -> None:
    app = build()
    tickets = [
        "I was charged twice for March, please refund one charge.",
        "The API returns 500 errors since this morning.",
        "Hi, quick one about my account.",
    ]
    for i, text in enumerate(tickets):
        config = {"configurable": {"thread_id": str(i)}}
        result = app.invoke({"text": text}, config)
        if "__interrupt__" in result:
            print(f"{text!r} -> Drex not sure, asking a person")
            result = app.invoke(Command(resume="billing"), config)
            print(f"   person chose: {result['team']}")
        else:
            print(f"{text!r} -> {result['team']}")


if __name__ == "__main__":
    main()
