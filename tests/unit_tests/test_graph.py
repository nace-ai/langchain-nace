from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langchain_nace import Choice, NaceClassifier, nace_conditional_edge
from tests.fakes import FakeDrex

QUESTION = Choice(instructions="Who handles this?", criteria={"billing": None, "technical": None})


class S(TypedDict, total=False):
    text: str
    handled_by: str


def build(classifier: NaceClassifier):
    graph = StateGraph(S)
    graph.add_node("triage", lambda s: {})
    graph.add_node("billing", lambda s: {"handled_by": "billing"})
    graph.add_node("technical", lambda s: {"handled_by": "technical"})
    graph.add_node("human_review", lambda s: {"handled_by": interrupt({"text": s["text"]})})
    graph.add_edge(START, "triage")
    edge = nace_conditional_edge(
        classifier, QUESTION, state=lambda s: s["text"], min_confidence=0.7
    )
    graph.add_conditional_edges("triage", edge, ["billing", "technical", "human_review"])
    for node in ("billing", "technical", "human_review"):
        graph.add_edge(node, END)
    return graph.compile(checkpointer=InMemorySaver())


def test_edge_routes_on_confident_answer(classifier: NaceClassifier) -> None:
    result = build(classifier).invoke(
        {"text": "charged twice"}, {"configurable": {"thread_id": "1"}}
    )
    assert result["handled_by"] == "billing"


async def test_low_confidence_interrupts_then_resumes(
    classifier: NaceClassifier, drex: FakeDrex
) -> None:
    drex.overrides["route"] = {
        "type": "choice",
        "choice": "technical",
        "confidence": 0.4,
        "probabilities": {"billing": 0.48, "technical": 0.52},
    }
    graph = build(classifier)
    config = {"configurable": {"thread_id": "2"}}
    paused = await graph.ainvoke({"text": "it's weird"}, config)
    assert paused["__interrupt__"][0].value == {"text": "it's weird"}
    done = await graph.ainvoke(Command(resume="technical"), config)
    assert done["handled_by"] == "technical"
