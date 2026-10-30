"""Credential isolation at the actual sync and async request boundary."""

import httpx
import pytest
from langchain_core.load import dumpd

from langchain_nace import NaceClassifier, Noul
from tests.fakes import FakeDrex

CALL = {"state": "hello", "questions": {"q": Noul(instructions="Greeting?")}}


@pytest.mark.parametrize(
    "url",
    [
        "https://openrouter.ai/api/v1",
        "https://evilopenrouter.ai/api/v1",
        "https://openrouter.ai./api/v1",
        "https://console.nace.ai.evil.test",
        "http://console.nace.ai",
        "https://console.nace.ai:444",
        "https://user@console.nace.ai",
    ],
)
def test_custom_host_requires_explicit_key(monkeypatch, drex: FakeDrex, url: str) -> None:
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    monkeypatch.setenv("NACE_BASE_URL", url)
    with pytest.raises(ValueError, match="explicit api_key"):
        NaceClassifier(http_client=httpx.Client(transport=drex.transport()))
    assert drex.requests == []


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("headers", [None, {"X-Vercel-Protection-Bypass": "FAKE_EXPLICIT_BYPASS"}])
@pytest.mark.parametrize(
    ("url", "path", "openrouter"),
    [
        ("https://openrouter.ai/api/v1", "/api/v1/systemone", True),
        ("https://openrouter.ai./api/v1", "/api/v1/systemone", True),
        ("https://evilopenrouter.ai/api", "/api/v1/systemone", False),
    ],
)
async def test_custom_key_without_bypass(
    monkeypatch, drex, url, path, openrouter, async_call, headers
):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "FAKE_BYPASS_TOKEN")
    classifier = NaceClassifier(
        api_key="sk-or-explicit-test",
        base_url=url,
        default_headers=headers,
        http_client=httpx.Client(transport=drex.transport()),
        http_async_client=httpx.AsyncClient(transport=drex.async_transport()),
    )
    if async_call:
        await classifier.ainvoke(CALL)
    else:
        classifier.invoke(CALL)
    request = drex.requests[0]
    assert request.headers["authorization"] == "Bearer sk-or-explicit-test"
    assert "x-vercel-protection-bypass" not in request.headers
    assert ("x-openrouter-title" in request.headers) is openrouter
    assert request.url.path == path
    serialized = str(dumpd(classifier))
    assert "NACE_API_KEY" not in serialized
    assert "sk-or-explicit-test" not in serialized


@pytest.mark.parametrize("async_call", [False, True])
async def test_trusted_host_uses_environment_credentials(monkeypatch, drex, async_call):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "FAKE_BYPASS_TOKEN")
    classifier = NaceClassifier(
        http_client=httpx.Client(transport=drex.transport()),
        http_async_client=httpx.AsyncClient(transport=drex.async_transport()),
    )
    if async_call:
        await classifier.ainvoke(CALL)
    else:
        classifier.invoke(CALL)
    assert drex.requests[0].headers["authorization"] == "Bearer nace_sk_FAKE_FOR_TEST"
    assert drex.requests[0].headers["x-vercel-protection-bypass"] == "FAKE_BYPASS_TOKEN"
