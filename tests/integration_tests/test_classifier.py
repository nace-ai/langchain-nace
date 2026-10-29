"""Live contract tests against Drex (prod by default). Synthetic data only."""

from __future__ import annotations

import base64
from pathlib import Path

import nace_sdk
import pytest

from langchain_nace import (
    Choice,
    ClassifierResponse,
    NaceImageModelError,
    NaceUnprocessableEntityError,
    Noul,
    NoulCriteria,
    Score,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
RED = base64.b64encode((FIXTURES / "red.png").read_bytes()).decode()
BLUE = base64.b64encode((FIXTURES / "blue.png").read_bytes()).decode()
MODELS = ["drex-v2.0", "drex-max"]
TICKET = (
    "I was charged twice for my March invoice and I'm really annoyed. Please refund one of them."
)


def check(response: ClassifierResponse, request_ids: list, test: str, model: str) -> None:
    assert response.request_id
    assert (response.usage.input_tokens or 0) > 0
    for answer in response.choices.values():
        assert abs(sum(answer.probabilities.values()) - 1) < 1e-2
    for answer in response.scores.values():
        assert abs(sum(answer.probabilities.values()) - 1) < 1e-2
    request_ids.append((test, model, response.request_id))


@pytest.mark.parametrize("model", MODELS)
def test_single_noul(live, request_ids, model: str) -> None:
    r = live(model).invoke(
        {
            "state": TICKET,
            "questions": {"refund": Noul(instructions="Is the customer asking for a refund?")},
        }
    )
    assert r.nouls["refund"].noul > 0.7
    check(r, request_ids, "single noul", model)


@pytest.mark.parametrize("model", MODELS)
def test_single_choice(live, request_ids, model: str) -> None:
    r = live(model).invoke(
        {
            "state": TICKET,
            "questions": {
                "team": Choice(
                    instructions="Which team should handle this?",
                    criteria={
                        "billing": "Payments, invoices, refunds",
                        "technical": "Bugs and outages",
                    },
                )
            },
        }
    )
    assert r.choices["team"].choice == "billing"
    assert set(r.choices["team"].probabilities) == {"billing", "technical"}
    check(r, request_ids, "single choice", model)


@pytest.mark.parametrize("model", MODELS)
def test_single_score(live, request_ids, model: str) -> None:
    r = live(model).invoke(
        {
            "state": TICKET,
            "questions": {
                "mood": Score(
                    instructions="How upset is the customer?",
                    criteria=["calm", "annoyed", "furious"],
                )
            },
        }
    )
    assert set(r.scores["mood"].probabilities) == {0, 1, 2}
    assert r.scores["mood"].legend == {0: "calm", 1: "annoyed", 2: "furious"}
    check(r, request_ids, "single score", model)


@pytest.mark.parametrize("model", MODELS)
def test_mixed_batch_in_one_request(live, request_ids, model: str) -> None:
    r = live(model).invoke(
        {
            "state": TICKET,
            "questions": {
                "refund": Noul(instructions="Is the customer asking for a refund?"),
                "team": Choice(
                    instructions="Which team?", criteria={"billing": None, "technical": None}
                ),
                "mood": Score(instructions="How upset?", criteria=["calm", "annoyed", "furious"]),
            },
        }
    )
    assert set(r.answers) == {"refund", "team", "mood"}
    check(r, request_ids, "mixed batch", model)


@pytest.mark.parametrize("model", MODELS)
def test_noul_criteria_without_instructions_and_null_choice_description(
    live, request_ids, model: str
) -> None:
    r = live(model).invoke(
        {
            "state": TICKET,
            "questions": {
                "refund": Noul(criteria=NoulCriteria(true="The customer asks for money back.")),
                "tone": Choice(
                    instructions="What is the tone?",
                    criteria={"polite": None, "rude": "Insults or shouting"},
                ),
            },
        }
    )
    assert r.nouls["refund"].noul > 0.5
    check(r, request_ids, "noul no instructions + null description", model)


def test_max_one_image(live, request_ids) -> None:
    r = live("drex-max").invoke(
        {
            "state": "One image is attached.",
            "images": [RED],
            "questions": {"red": Noul(instructions="Is Image 1 mostly red?")},
        }
    )
    assert r.nouls["red"].noul > 0.7
    assert (r.usage.image_tokens or 0) > 0
    check(r, request_ids, "max 1 image", "drex-max")


@pytest.mark.parametrize(
    ("images", "expected"), [((RED, BLUE), "Image 1"), ((BLUE, RED), "Image 2")]
)
def test_max_two_images_in_order(live, request_ids, images: tuple[str, str], expected: str) -> None:
    r = live("drex-max").invoke(
        {
            "state": "Two images are attached.",
            "images": list(images),
            "questions": {
                "red": Choice(
                    instructions="Which image is red?", criteria={"Image 1": None, "Image 2": None}
                )
            },
        }
    )
    assert r.choices["red"].choice == expected
    check(r, request_ids, f"max 2 images, red is {expected}", "drex-max")


def test_max_accepts_more_than_16_questions(live) -> None:
    """The former 16-question cap must not reject valid larger Max requests."""
    questions = {
        f"q{i}": Noul(instructions=f"Does the text mention the number {i}?") for i in range(17)
    }
    r = live("drex-max").invoke({"state": TICKET, "questions": questions})
    assert set(r.nouls) == set(questions)


def test_images_with_v2_refused_client_side(live) -> None:
    with pytest.raises(NaceImageModelError, match="drex-max"):
        live("drex-v2.0").invoke(
            {"state": "x", "images": [RED], "questions": {"q": Noul(instructions="Red?")}}
        )


def test_422_surfaces_path_and_message(live) -> None:
    with pytest.raises(NaceUnprocessableEntityError) as caught:
        live("drex-v2.0").invoke(
            {"state": TICKET, "questions": {"q": {"type": "rank", "criteria": ["a"]}}}
        )
    paths = [issue.get("path") for issue in caught.value.issues]
    assert any(p and p.startswith("questions") for p in paths)
    assert "questions" in str(caught.value)
    print(f"\nunknown question type -> {caught.value}")


def test_every_listed_model_answers(live, request_ids) -> None:
    import os

    with nace_sdk.NaceClient(
        api_key=os.environ["NACE_API_KEY_LIVE"],
        base_url=os.environ.get("NACE_BASE_URL_LIVE") or None,
    ) as client:
        listed = [
            m.name for m in client.models.list().models if not m.description.startswith("Retired")
        ]
    assert "drex-v2.0" in listed
    for model in listed:
        r = live(model).invoke(
            {"state": TICKET, "questions": {"refund": Noul(instructions="Refund request?")}}
        )
        assert 0 <= r.nouls["refund"].noul <= 1
        check(r, request_ids, f"listed model {model}", model)


async def test_async_and_batch(live, request_ids) -> None:
    classifier = live("drex-v2.0")
    r = await classifier.ainvoke(
        {"state": TICKET, "questions": {"refund": Noul(instructions="Refund?")}}
    )
    check(r, request_ids, "ainvoke", "drex-v2.0")
    results = classifier.batch(
        [
            {"state": t, "questions": {"refund": Noul(instructions="Refund?")}}
            for t in (TICKET, "The app crashes on login.")
        ]
    )
    assert results[0].nouls["refund"].noul > results[1].nouls["refund"].noul


def test_default_model_answers_as_drex_max(request_ids) -> None:
    import os

    from langchain_nace import NaceClassifier

    classifier = NaceClassifier(
        api_key=os.environ["NACE_API_KEY_LIVE"],
        base_url=os.environ.get("NACE_BASE_URL_LIVE") or None,
    )
    r = classifier.invoke(
        {
            "state": "One image.",
            "images": [RED],
            "questions": {"red": Noul(instructions="Is Image 1 red?")},
        }
    )
    assert classifier.model == "drex-max"
    assert r.model == "drex-max"
    assert r.nouls["red"].noul > 0.7
    request_ids.append(("default model + image, no model= given", r.model, r.request_id))
