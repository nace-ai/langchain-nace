import pytest
from pydantic import ValidationError

from langchain_nace import Choice, ClassifierResponse, Noul, NoulCriteria, RawAnswer, Score
from langchain_nace.types import to_wire


def test_noul_without_instructions_sends_empty_string() -> None:
    wire = to_wire(Noul(criteria=NoulCriteria(true="The customer asks for money back.")))
    assert wire == {
        "type": "noul",
        "instructions": "",
        "criteria": {"true": "The customer asks for money back."},
    }


def test_choice_null_description_kept() -> None:
    wire = to_wire(Choice(instructions="Tone?", criteria={"calm": None, "angry": "Shouting"}))
    assert wire["criteria"] == {"calm": None, "angry": "Shouting"}


def test_structured_instructions_become_json_text() -> None:
    assert to_wire(Noul(instructions={"q": "x"}))["instructions"] == '{"q":"x"}'
    assert to_wire(Score(instructions="s", criteria=[{"a": 1}, "b"]))["criteria"] == [
        '{"a":1}',
        "b",
    ]


def test_raw_question_passes_through_with_instructions_string() -> None:
    wire = to_wire({"type": "rank", "criteria": ["a", "b"], "instructions": None})
    assert wire == {"type": "rank", "criteria": ["a", "b"], "instructions": ""}


def test_choice_needs_a_label() -> None:
    with pytest.raises(ValidationError):
        Choice(criteria={})


def test_typesafe_readme_example_constructs() -> None:
    Choice(instructions="Which team?", criteria={"billing": "Payments", "technical": "Bugs"})
    Noul(instructions="Does this message express urgency?")
    Score(instructions="How frustrated?", criteria=["calm", "frustrated", "angry"])


def test_response_keeps_unknown_answer_types_and_converts_legend() -> None:
    body = {
        "model": "drex-v2.0",
        "answers": {
            "s": {
                "type": "score",
                "score": 1.4,
                "legend": ["calm", "angry"],
                "probabilities": {"0": 0.3, "1": 0.7},
                "confidence": 0.6,
            },
            "future": {"type": "rank", "order": ["a", "b"]},
        },
        "usage": {"input_tokens": 10, "image_tokens": 0, "future_field": 1},
        "evaluation_time_ms": 50,
        "new_top_level": True,
    }
    response = ClassifierResponse.from_body(body, "req_1")
    assert response.scores["s"].legend == {0: "calm", 1: "angry"}
    assert response.scores["s"].probabilities == {0: 0.3, 1: 0.7}
    assert isinstance(response.answers["future"], RawAnswer)
    assert response.answers["future"].model_extra == {"order": ["a", "b"]}
    assert response.raw["new_top_level"] is True
    assert response.request_id == "req_1"
