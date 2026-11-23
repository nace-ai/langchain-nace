"""Live Drex Router checks. Synthetic prompts only."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from langchain_nace import NaceModelSelector
from tests.integration_tests.conftest import BASE_URL, KEY

FAST = "openai/gpt-4o-mini"
STRONG = "anthropic/claude-haiku-5.5"
CUSTOM_ID = "my-org/fine-tuned-support-model"
CUSTOM_PROFILE = {
    "provider": "my-org",
    "price_in": 0.4,
    "price_out": 1.6,
    "context": 128000,
    "tags": ["support", "chat"],
}
PROMPT = [HumanMessage("I was charged twice for March. Please refund one charge.")]


def selector(candidates: dict) -> NaceModelSelector:
    # The server stops the router after 40 seconds. Wait past that.
    return NaceModelSelector(api_key=KEY, base_url=BASE_URL, candidates=candidates, timeout=60)


def check(selection: dict, ids: set[str], request_ids: list, test: str) -> None:
    assert selection["reason"] == "selected"
    assert selection["selected"] in ids
    assert set(selection["probabilities"]) == ids
    assert abs(sum(selection["probabilities"].values()) - 1) < 1e-2
    request_ids.append((test, "route", selection["request_id"]))


def test_two_known_ids(request_ids) -> None:
    # Case 21. Two bare known ids. The probabilities sum to 1.
    ids = {FAST, STRONG}
    selection = selector({FAST: None, STRONG: None}).select(PROMPT)
    check(selection, ids, request_ids, "route two known ids")


async def test_two_known_ids_async(request_ids) -> None:
    ids = {FAST, STRONG}
    selection = await selector({FAST: None, STRONG: None}).aselect(PROMPT)
    check(selection, ids, request_ids, "route two known ids async")


def test_custom_profile_returns_its_id(request_ids) -> None:
    # Case 21. A custom profile keeps the id we sent.
    ids = {CUSTOM_ID, FAST}
    selection = selector({CUSTOM_ID: CUSTOM_PROFILE, FAST: None}).select(PROMPT)
    check(selection, ids, request_ids, "route custom profile")
    assert CUSTOM_ID in selection["probabilities"]


async def test_custom_profile_returns_its_id_async(request_ids) -> None:
    ids = {CUSTOM_ID, FAST}
    selection = await selector({CUSTOM_ID: CUSTOM_PROFILE, FAST: None}).aselect(PROMPT)
    check(selection, ids, request_ids, "route custom profile async")
    assert CUSTOM_ID in selection["probabilities"]
