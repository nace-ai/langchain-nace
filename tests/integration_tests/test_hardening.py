"""Live checks beyond the contract cases: failures, limits, volume, every image shape."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from langchain_nace import (
    Choice,
    NaceAPIError,
    NaceAuthenticationError,
    NaceClassifier,
    NaceUnprocessableEntityError,
    Noul,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
RED = base64.b64encode((FIXTURES / "red.png").read_bytes()).decode()
BLUE = base64.b64encode((FIXTURES / "blue.png").read_bytes()).decode()
WHICH_RED = {
    "red": Choice(instructions="Which image is red?", criteria={"Image 1": None, "Image 2": None})
}


def test_bad_key_is_a_clear_401() -> None:
    bad = NaceClassifier(
        api_key="nace_sk_definitely_not_a_real_key",
        base_url=os.environ.get("NACE_BASE_URL_LIVE") or None,
    )
    with pytest.raises(NaceAuthenticationError) as caught:
        bad.invoke({"state": "x", "questions": {"q": Noul(instructions="?")}})
    assert caught.value.status == 401
    print(f"\nbad key -> {caught.value}")


def test_unknown_drex_name_is_refused_or_served_by_drex_latest(live) -> None:
    """Newer Drex refuses an unknown `drex-*` id with a 422 on `model`; older Drex
    served it like `drex-latest`. The package passes the id through either way."""
    refused: NaceUnprocessableEntityError | None = None
    try:
        r = live("drex-v99.9").invoke({"state": "x", "questions": {"q": Noul(instructions="?")}})
    except NaceUnprocessableEntityError as e:
        refused = e
    if refused is not None:
        print(f"\ndrex-v99.9 -> {refused}")
        assert any(i.get("path") == "model" for i in refused.issues)
    else:
        print(f"\ndrex-v99.9 answered by {r.model}")
        assert r.model.startswith("drex-")


def test_non_drex_model_is_refused_with_path_model(live) -> None:
    with pytest.raises(NaceUnprocessableEntityError) as caught:
        live("jev-latest").invoke({"state": "x", "questions": {"q": Noul(instructions="?")}})
    assert any(i.get("path") == "model" for i in caught.value.issues)
    print(f"\njev-latest -> {caught.value}")


def test_nine_images_refused_by_drex(live) -> None:
    with pytest.raises(NaceUnprocessableEntityError) as caught:
        live("drex-max").invoke(
            {"state": "x", "images": [RED] * 9, "questions": {"q": Noul(instructions="Red?")}}
        )
    assert any(str(i.get("path", "")).startswith("images") for i in caught.value.issues)
    print(f"\n9 images -> {caught.value}")


def test_body_over_4_mib_refused_by_drex(live) -> None:
    with pytest.raises(NaceAPIError) as caught:
        live("drex-v2.0").invoke(
            {"state": "a " * (2_200_000), "questions": {"q": Noul(instructions="?")}}
        )
    assert caught.value.status in {400, 413, 422}
    print(f"\n4.4 MB body -> {caught.value}")


@pytest.mark.parametrize(
    "block",
    [
        lambda d: {"type": "image", "base64": d, "mime_type": "image/png"},
        lambda d: {"type": "image", "source_type": "base64", "data": d, "mime_type": "image/png"},
        lambda d: {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{d}"}},
    ],
    ids=["v1-base64", "v0-source_type", "image_url"],
)
def test_langchain_image_blocks_inside_messages(live, request_ids, block) -> None:
    message = HumanMessage(
        content=[{"type": "text", "text": "Two pictures:"}, block(BLUE), block(RED)]
    )
    r = live("drex-max").invoke({"state": [message], "questions": WHICH_RED})
    assert r.choices["red"].choice == "Image 2"
    request_ids.append(("message image blocks", "drex-max", r.request_id))


def test_image_from_file_path_and_bytes(live) -> None:
    r = live("drex-max").invoke(
        {
            "state": "Two pictures.",
            "questions": WHICH_RED,
            "images": [FIXTURES / "red.png", (FIXTURES / "blue.png").read_bytes()],
        }
    )
    assert r.choices["red"].choice == "Image 1"


def test_batch_of_20_without_rate_limit_errors(live) -> None:
    classifier = live("drex-v2.0")
    texts = [
        f"Order #{i}: {'refund please' if i % 2 else 'where is my package'}" for i in range(20)
    ]
    start = time.monotonic()
    results = classifier.batch(
        [{"state": t, "questions": {"refund": Noul(instructions="Refund request?")}} for t in texts]
    )
    elapsed = time.monotonic() - start
    refunds = [r.nouls["refund"].noul > 0.5 for r in results]
    assert refunds == [bool(i % 2) for i in range(20)]
    print(f"\n20 requests in {elapsed:.1f}s (max 8 in flight)")


async def test_abatch(live) -> None:
    results = await live("drex-v2.0").abatch(
        [
            {"state": t, "questions": {"q": Noul(instructions="Is this about money?")}}
            for t in ("I was overcharged", "The app crashes")
        ]
    )
    assert results[0].nouls["q"].noul > results[1].nouls["q"].noul


def test_json_state_and_long_questions(live) -> None:
    state = {
        "customer": {"plan": "pro", "tenure_months": 30},
        "messages": [HumanMessage("Cancel my plan now.")],
    }
    r = live("drex-v2.0").invoke(
        {
            "state": state,
            "questions": {
                "churn": Noul(
                    instructions="Is the customer trying to cancel? "
                    + "Consider plan and tenure. " * 50
                )
            },
        }
    )
    assert r.nouls["churn"].noul > 0.7
