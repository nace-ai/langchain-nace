from __future__ import annotations

import os
import warnings

import pytest
from langchain_core._api import LangChainBetaWarning

from langchain_nace import NaceClassifier

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

KEY = os.environ.get("NACE_API_KEY_LIVE")
BASE_URL = os.environ.get("NACE_BASE_URL_LIVE") or None  # unset or empty = prod


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        item.add_marker(pytest.mark.live)
        if not KEY:
            item.add_marker(pytest.mark.skip(reason="set NACE_API_KEY_LIVE to run live tests"))


@pytest.fixture(scope="session")
def live():
    def make(model: str, **kwargs: object) -> NaceClassifier:
        return NaceClassifier(model=model, api_key=KEY, base_url=BASE_URL, **kwargs)  # type: ignore[arg-type]

    return make


@pytest.fixture(scope="session")
def request_ids() -> list[tuple[str, str, str | None]]:
    """(test, model, request_id) for every live call; printed at the end of the run."""
    return []


@pytest.fixture(scope="session", autouse=True)
def _print_request_ids(request_ids: list[tuple[str, str, str | None]]):
    yield
    if request_ids:
        print("\nLive request ids:")
        for test, model, rid in request_ids:
            print(f"  {test:<45} {model:<12} {rid}")
