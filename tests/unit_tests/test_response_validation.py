"""Responses must answer the questions sent, without exposing bad data in errors."""

import pytest

from langchain_nace import Choice, ClassifierResponse, Noul, Score, nace_conditional_edge
from langchain_nace.errors import NaceAPIResponseValidationError
from tests.agents import noul, score

CHOICE = {"type": "choice", "choice": "safe", "confidence": 0.9, "probabilities": {"safe": 1.0}}
SECRET = "response-only-sensitive-text"


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        (Noul(), None),
        (Noul(), {"type": SECRET}),
        (Choice(criteria={"safe": None}), {**CHOICE, "choice": SECRET}),
        (Score(criteria=["low", "high"]), score(1, levels=9)),
        (Score(criteria=["low", "high"]), noul(0.9)),
        ({"type": "future"}, None),
    ],
)
def test_answers_must_fit_asked_questions(question, answer):
    response = ClassifierResponse.from_body(
        {"answers": {} if answer is None else {"q": answer}}, None
    )
    with pytest.raises(NaceAPIResponseValidationError) as raised:
        response.validate_answers({"q": question})
    assert SECRET not in str(raised.value)
    assert SECRET not in repr(raised.value)


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "noul", "noul": SECRET},
        {"type": "noul", "noul": "0.5"},
        {"type": "noul", "noul": True},
        {**score(0), "score": "0"},
        {**CHOICE, "confidence": "0.9"},
        {"type": "noul", "noul": float("nan")},
        {"type": "noul", "noul": -0.1},
        {**CHOICE, "confidence": 1.1},
        {**CHOICE, "probabilities": {SECRET: float("inf")}},
        {**score(0), "legend": {"0": "low", "2": SECRET}},
        {**score(0), "legend": []},
        {**score(0), "score": 5},
        {**score(0), "probabilities": {SECRET: 1}},
        {"type": "score", "score": SECRET},
        {"type": "choice", "choice": SECRET},
    ],
)
def test_malformed_known_answers_raise_sanitized_errors(answer):
    with pytest.raises(NaceAPIResponseValidationError) as raised:
        ClassifierResponse.from_body({"answers": {SECRET: answer}}, None)
    assert SECRET not in str(raised.value)
    assert SECRET not in repr(raised.value)
    assert raised.value.__suppress_context__ or raised.value.__context__ is None


@pytest.mark.parametrize("body", [{}, {"answers": []}, {"answers": {"q": None}}, {"usage": SECRET}])
def test_malformed_response_structure_is_sanitized(body):
    with pytest.raises(NaceAPIResponseValidationError) as raised:
        ClassifierResponse.from_body(body, None)
    assert SECRET not in str(raised.value)


def test_future_types_and_extra_answers_remain_supported():
    response = ClassifierResponse.from_body(
        {"answers": {"future": {"type": "future", "value": SECRET}, "extra": noul(0.8)}},
        None,
        {"future": {"type": "future"}},
    )
    assert response.answers["future"].type == "future"
    assert response.nouls["extra"].noul == 0.8


@pytest.mark.parametrize("confidence", [0.1, 0.9])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_conditional_edge_refuses_unoffered_label(classifier, drex, confidence, asynchronous):
    drex.overrides["route"] = {**CHOICE, "choice": SECRET, "confidence": confidence}
    edge = nace_conditional_edge(
        classifier,
        Choice(criteria={"safe": None}),
        state=lambda state: state,
        min_confidence=0.5,
    )

    async def invoke():
        return await edge.ainvoke("route this") if asynchronous else edge.invoke("route this")

    with pytest.raises(NaceAPIResponseValidationError) as raised:
        await invoke()
    assert SECRET not in str(raised.value)
