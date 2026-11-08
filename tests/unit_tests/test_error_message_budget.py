"""Display budgets apply after request ids and credentials are redacted."""

import httpx
import nace_sdk
import pytest

from langchain_nace.errors import convert_error
from tests.fakes import error


@pytest.mark.parametrize(("provider", "repeat"), [("Drex", 32), ("OpenRouter", 4096)])
def test_redacted_request_id_cannot_expand_final_error_past_budget(provider, repeat):
    secret = "abcd"
    failure = nace_sdk.NaceAPIError(
        402,
        error(402, "detail " * 2000)[2],
        httpx.Headers({"x-request-id": secret * repeat}),
    )
    text = str(convert_error(failure, provider, secrets=[secret]))
    assert len(text) <= 600
    assert secret not in text
    link = (
        "https://console.nace.ai/dashboard/billing"
        if provider == "Drex"
        else "https://openrouter.ai/settings/credits"
    )
    assert link in text
    assert text.splitlines()[-1].startswith("(request_id=")
    assert text.endswith(")")
