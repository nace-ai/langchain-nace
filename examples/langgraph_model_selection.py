"""Route a support request, then let Drex Router pick the model.

`NaceClassifier` answers a Choice question: billing or support. A conditional
edge sends the request to that agent. `NaceModelSelectorMiddleware` picks the
generative model inside the agent. The chat models are scripted. The Drex calls
are real.

    export NACE_API_KEY=nace_sk_...
    uv run python examples/langgraph_model_selection.py
"""

from __future__ import annotations

import warnings
from typing import Any, TypedDict

from langchain.agents import create_agent
from langchain_core._api import LangChainBetaWarning
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langchain_nace import Choice, NaceClassifier, NaceModelSelector, nace_conditional_edge
from langchain_nace.experimental.middleware import ModelCandidate, NaceModelSelectorMiddleware

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

ROUTE_TIMEOUT = 60.0
TEAMS = Choice(
    instructions="Which team should own this support request?",
    criteria={
        "billing": "Charges, invoices, refunds, plans.",
        "support": "Bugs, outages, how-to questions.",
    },
)


class Request(TypedDict, total=False):
    text: str
    team: str
    model: str | None
    reason: str
    probabilities: dict[str, float]


class _Scripted(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> _Scripted:  # type: ignore[override]
        return self


def _agent(team: str, selector: NaceModelSelector) -> Any:
    fast = _Scripted(messages=iter([AIMessage(f"{team} reply from the fast candidate.")]))
    strong = _Scripted(messages=iter([AIMessage(f"{team} reply from the strong candidate.")]))
    middleware = NaceModelSelectorMiddleware(
        candidates={
            "openai/gpt-4o-mini": ModelCandidate(fast),
            "anthropic/claude-haiku-5.5": ModelCandidate(strong),
        },
        priority="balanced",
        fallback="openai/gpt-4o-mini",
        timeout=ROUTE_TIMEOUT,
        selector=selector,
    )
    return create_agent(fast, tools=[], middleware=[middleware])


def build() -> Any:
    selector = NaceModelSelector(timeout=ROUTE_TIMEOUT)
    agents = {team: _agent(team, selector) for team in TEAMS.criteria}

    def answer(team: str, state: Request) -> dict[str, Any]:
        result = agents[team].invoke({"messages": [HumanMessage(state["text"])]})
        selection = result["nace_model_selection"]
        return {
            "team": team,
            "model": selection["model"],
            "reason": selection["reason"],
            "probabilities": selection["probabilities"],
        }

    def team_node(team: str) -> Any:
        def node(state: Request) -> dict[str, Any]:
            return answer(team, state)

        return node

    def human(state: Request) -> dict[str, Any]:
        team = interrupt({"ticket": state["text"]})
        if team not in TEAMS.criteria:
            team = "support"
        return answer(str(team), state)

    graph = StateGraph(Request)
    graph.add_node("triage", lambda _state: {})
    graph.add_node("billing", team_node("billing"))
    graph.add_node("support", team_node("support"))
    graph.add_node("human_review", human)
    graph.add_edge(START, "triage")
    for team in (*TEAMS.criteria, "human_review"):
        graph.add_edge(team, END)
    edge = nace_conditional_edge(
        NaceClassifier(), TEAMS, state=lambda state: state["text"], min_confidence=0.7
    )
    graph.add_conditional_edges("triage", edge, [*TEAMS.criteria, "human_review"])
    return graph.compile(checkpointer=InMemorySaver())


def main() -> None:
    app = build()
    tickets = [
        "I was charged twice for March. Please refund one charge.",
        "The API returns 500 errors since this morning.",
    ]
    for index, text in enumerate(tickets):
        config = {"configurable": {"thread_id": str(index)}}
        result = app.invoke({"text": text}, config)
        if "__interrupt__" in result:
            print(f"{text!r} -> Drex not sure, asking a person")
            result = app.invoke(Command(resume="support"), config)
        print(f"{text!r} -> {result['team']}")
        print(f"model={result['model']}")
        print(f"reason={result['reason']}")
        print(f"probabilities={result['probabilities']}")


if __name__ == "__main__":
    main()
