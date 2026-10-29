import pytest
from langchain_core.runnables import RunnableLambda

from langchain_nace import Choice, NaceClassifier, NaceLowConfidenceError, NaceRouter
from tests.fakes import FakeDrex

QUESTION = Choice(instructions="Which team?", criteria={"billing": "Payments", "technical": "Bugs"})


def low_confidence(drex: FakeDrex) -> None:
    drex.overrides["route"] = {
        "type": "choice",
        "choice": "technical",
        "confidence": 0.3,
        "probabilities": {"billing": 0.45, "technical": 0.55},
    }


def test_router_picks_label(classifier: NaceClassifier, drex: FakeDrex) -> None:
    decision = NaceRouter(
        classifier=classifier, question=QUESTION, state=lambda x: x["text"]
    ).invoke({"text": "charged twice"})
    assert decision.route == "billing"
    assert decision.answer.probabilities == {"billing": 0.8, "technical": 0.2}
    assert drex.bodies[0]["state"] == "charged twice"


def test_router_low_confidence_falls_back(classifier: NaceClassifier, drex: FakeDrex) -> None:
    low_confidence(drex)
    decision = NaceRouter(
        classifier=classifier, question=QUESTION, min_confidence=0.6, fallback="human"
    ).invoke("hmm")
    assert decision.route == "human"
    assert decision.fell_back


def test_router_low_confidence_without_fallback_raises(
    classifier: NaceClassifier, drex: FakeDrex
) -> None:
    low_confidence(drex)
    with pytest.raises(NaceLowConfidenceError):
        NaceRouter(classifier=classifier, question=QUESTION, min_confidence=0.6).invoke("hmm")


def test_branch_runs_the_selected_runnable(classifier: NaceClassifier) -> None:
    chain = NaceRouter(classifier=classifier, question=QUESTION).branch(
        {
            "billing": RunnableLambda(lambda x: f"billing:{x}"),
            "technical": RunnableLambda(lambda x: "tech"),
        },
        default=RunnableLambda(lambda x: "default"),
    )
    assert chain.invoke("charged twice") == "billing:charged twice"


async def test_branch_async_and_unknown_label_runs_default(
    classifier: NaceClassifier, drex: FakeDrex
) -> None:
    drex.overrides["route"] = {
        "type": "choice",
        "choice": "sales",
        "confidence": 0.9,
        "probabilities": {"sales": 0.9},
    }
    chain = NaceRouter(classifier=classifier, question=QUESTION).branch(
        {"billing": RunnableLambda(lambda x: "billing")},
        default=RunnableLambda(lambda x: "default"),
    )
    assert await chain.ainvoke("x") == "default"


def test_router_trace_metadata(classifier: NaceClassifier) -> None:
    from langchain_core.tracers.context import collect_runs

    with collect_runs() as cb:
        NaceRouter(
            classifier=classifier, question=QUESTION, min_confidence=0.5, fallback="h"
        ).invoke("x")
    metadata = cb.traced_runs[0].extra["metadata"]
    assert metadata["nace_min_confidence"] == 0.5
    assert metadata["nace_fallback"] == "h"
