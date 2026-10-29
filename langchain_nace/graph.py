"""LangGraph helper: route a graph with Drex choice probabilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from langchain_nace.classifier import NaceClassifier
from langchain_nace.types import Choice, ChoiceAnswer, ClassifierRequest, State

_QUESTION_ID = "route"


def nace_conditional_edge(
    classifier: NaceClassifier,
    question: Choice,
    *,
    state: Callable[[Any], State],
    min_confidence: float,
    low_confidence: str = "human_review",
) -> Runnable[Any, str]:
    """Return a path function for `StateGraph.add_conditional_edges`.

    The edge asks Drex `question` about `state(graph_state)` and returns the chosen
    label. When Drex's confidence is below `min_confidence` it returns
    `low_confidence` instead, so the graph can go to a node that calls `interrupt()`
    for a human. Works in sync and async graphs.

    Example:
        ```python
        edge = nace_conditional_edge(
            NaceClassifier(),
            Choice(instructions="Who handles this?", criteria={"billing": None, "tech": None}),
            state=lambda s: s["messages"],
            min_confidence=0.7,
        )
        graph.add_conditional_edges("triage", edge, ["billing", "tech", "human_review"])
        ```

    """

    def request(graph_state: Any) -> ClassifierRequest:
        return {"state": state(graph_state), "questions": {_QUESTION_ID: question}}

    def pick(answer: ChoiceAnswer) -> str:
        return answer.choice if answer.confidence >= min_confidence else low_confidence

    def config_for(config: RunnableConfig) -> RunnableConfig:
        return {
            **config,
            "metadata": {**(config.get("metadata") or {}), "nace_min_confidence": min_confidence},
        }

    def route(graph_state: Any, config: RunnableConfig) -> str:
        response = classifier.invoke(request(graph_state), config_for(config))
        return pick(response.choices[_QUESTION_ID])

    async def aroute(graph_state: Any, config: RunnableConfig) -> str:
        response = await classifier.ainvoke(request(graph_state), config_for(config))
        return pick(response.choices[_QUESTION_ID])

    return RunnableLambda(route, afunc=aroute, name="nace_conditional_edge")


__all__ = ["nace_conditional_edge"]
