"""Live OpenRouter backend check (needs OPENROUTER_API_KEY_LIVE)."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from langchain_nace import Choice, NaceClassifier, Noul

KEY = os.environ.get("OPENROUTER_API_KEY_LIVE")
FIXTURES = Path(__file__).parents[1] / "fixtures"
RED = base64.b64encode((FIXTURES / "red.png").read_bytes()).decode()
BLUE = base64.b64encode((FIXTURES / "blue.png").read_bytes()).decode()

pytestmark = pytest.mark.skipif(not KEY, reason="set OPENROUTER_API_KEY_LIVE")


def openrouter(model: str) -> NaceClassifier:
    return NaceClassifier(model=model, api_key=KEY, base_url="https://openrouter.ai/api/v1")


@pytest.mark.parametrize("model", ["nace-ai/drex-v2.0", "nace-ai/drex-max"])
def test_openrouter_text(model: str) -> None:
    r = openrouter(model).invoke(
        {
            "state": "I was charged twice, refund please",
            "questions": {"refund": Noul(instructions="Refund request?")},
        }
    )
    assert r.nouls["refund"].noul > 0.7


def test_openrouter_images_in_state_content_array() -> None:
    r = openrouter("nace-ai/drex-max").invoke(
        {
            "state": "Two images.",
            "images": [BLUE, RED],
            "questions": {
                "red": Choice(
                    instructions="Which image is red?", criteria={"Image 1": None, "Image 2": None}
                )
            },
        }
    )
    assert r.choices["red"].choice == "Image 2"


def test_openrouter_errors_are_readable() -> None:
    from langchain_nace import (
        NaceAuthenticationError,
        NaceBadRequestError,
        NaceUnprocessableEntityError,
    )

    def run(model: str, questions: dict, key: str | None = None) -> None:
        NaceClassifier(
            model=model, api_key=key or KEY, base_url="https://openrouter.ai/api/v1", max_retries=0
        ).invoke({"state": "refund please", "questions": questions})

    refund = {"q": Noul(instructions="Refund?")}
    with pytest.raises(NaceAuthenticationError) as bad_key:
        run("nace-ai/drex-v2.0", refund, key="sk-or-v1-" + "0" * 64)
    with pytest.raises(NaceBadRequestError) as unknown:
        run("nace-ai/drex-v99.9", refund)
    with pytest.raises(NaceUnprocessableEntityError) as too_many:
        run("nace-ai/drex-max", {f"q{i}": Noul(instructions=f"mentions {i}?") for i in range(17)})
    with pytest.raises(NaceBadRequestError) as bad_type:
        run("nace-ai/drex-v2.0", {"q": {"type": "rank", "criteria": ["a"]}})
    for caught in (bad_key, unknown, too_many, bad_type):
        print(f"\n{caught.value}")
        assert str(caught.value).startswith("OpenRouter returned")
        assert "HTTP 4" not in str(caught.value)
    assert "does not exist" in str(unknown.value)
    assert "16" in str(too_many.value)
    assert "questions.q.type" in str(bad_type.value)
