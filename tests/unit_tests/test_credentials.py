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


@pytest.mark.parametrize("key", ["", "   "])
def test_blank_explicit_key_on_custom_host_is_rejected(monkeypatch, drex, key):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    with pytest.raises(ValueError, match="explicit api_key"):
        NaceClassifier(
            api_key=key,
            base_url="https://evil.test",
            http_client=httpx.Client(transport=drex.transport()),
        )


def test_empty_explicit_key_on_nace_host_does_not_fall_back_to_env(monkeypatch, drex):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    monkeypatch.setenv("DREX_API_KEY", "nace_sk_FAKE_LEGACY_FOR_TEST")
    with pytest.raises(ValueError, match=r"(?i)api key"):
        NaceClassifier(api_key="", http_client=httpx.Client(transport=drex.transport()))
    assert drex.requests == []


def test_unset_base_url_is_sent_explicitly(monkeypatch, drex):
    monkeypatch.setenv("DREX_BASE_URL", "https://evil.test")
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    classifier = NaceClassifier(
        base_url=None,
        http_client=httpx.Client(transport=drex.transport()),
    )
    classifier.invoke(CALL)
    assert drex.requests[0].url.host == "console.nace.ai"


def test_default_serialization_pins_destination(monkeypatch, drex):
    monkeypatch.delenv("NACE_BASE_URL", raising=False)
    classifier = NaceClassifier(
        api_key="nace_sk_FAKE_FOR_TEST",
        http_client=httpx.Client(transport=drex.transport()),
    )
    assert dumpd(classifier)["kwargs"]["base_url"] == "https://console.nace.ai"


@pytest.mark.parametrize(
    ("url", "secret"),
    [
        ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        ("https://gateway.example", "NACE_CUSTOM_API_KEY"),
    ],
)
def test_custom_host_serializes_provider_secret(drex, url, secret):
    classifier = NaceClassifier(
        api_key="sk-explicit-test",
        base_url=url,
        http_client=httpx.Client(transport=drex.transport()),
    )
    assert dumpd(classifier)["kwargs"]["api_key"]["id"] == [secret]


def test_openrouter_format_override_on_custom_host_keeps_custom_secret(monkeypatch, drex):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-REAL_OPENROUTER_FAKE")
    classifier = NaceClassifier(
        api_key="sk-gateway-test",
        base_url="https://gateway.example",
        openrouter=True,
        http_client=httpx.Client(transport=drex.transport()),
    )
    serialized = dumpd(classifier)
    assert serialized["kwargs"]["api_key"]["id"] == ["NACE_CUSTOM_API_KEY"]
    # `load` refuses this namespace, so resolve the secret the way a reload would.
    secrets = {"NACE_CUSTOM_API_KEY": "sk-gateway-reloaded"}
    reloaded = NaceClassifier(
        api_key=secrets[serialized["kwargs"]["api_key"]["id"][0]],
        base_url=serialized["kwargs"]["base_url"],
        openrouter=serialized["kwargs"].get("openrouter"),
        http_client=httpx.Client(transport=drex.transport()),
    )
    reloaded.invoke(CALL)
    assert drex.requests[0].headers["authorization"] == "Bearer sk-gateway-reloaded"


@pytest.mark.parametrize("host", ["console.nace.ai.", "CONSOLE.NACE.AI", "drex.nace.ai:443"])
def test_trusted_host_variants_keep_ambient_key(monkeypatch, drex, host):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    classifier = NaceClassifier(
        base_url=f"https://{host}",
        http_client=httpx.Client(transport=drex.transport()),
    )
    classifier.invoke(CALL)
    assert drex.requests[0].headers["authorization"] == "Bearer nace_sk_FAKE_FOR_TEST"


def test_password_only_userinfo_is_not_trusted(monkeypatch, drex):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    with pytest.raises(ValueError, match="explicit api_key"):
        NaceClassifier(
            base_url="https://:pw@console.nace.ai",
            http_client=httpx.Client(transport=drex.transport()),
        )


def test_bypass_cookie_header_is_removed_on_custom_host(drex):
    classifier = NaceClassifier(
        api_key="sk-explicit-test",
        base_url="https://gateway.example",
        default_headers={"X-Vercel-Set-Bypass-Cookie": "true"},
        http_client=httpx.Client(transport=drex.transport()),
    )
    classifier.invoke(CALL)
    assert "x-vercel-set-bypass-cookie" not in drex.requests[0].headers


@pytest.mark.parametrize("async_call", [False, True])
async def test_redirect_to_custom_host_does_not_forward_credentials(monkeypatch, drex, async_call):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "FAKE_BYPASS_TOKEN")
    drex.replies.append((307, {"location": "https://external.test/v1/systemone"}, {}))
    with httpx.Client(transport=drex.transport(), follow_redirects=True) as sync_client:
        async with httpx.AsyncClient(
            transport=drex.async_transport(), follow_redirects=True
        ) as async_client:
            classifier = NaceClassifier(
                default_headers={"X-Vercel-Set-Bypass-Cookie": "true"},
                http_client=sync_client,
                http_async_client=async_client,
            )
            if async_call:
                await classifier.ainvoke(CALL)
            else:
                classifier.invoke(CALL)
    assert len(drex.requests) == 2
    first, redirected = drex.requests
    assert first.url.host == "console.nace.ai"
    assert first.headers["authorization"] == "Bearer nace_sk_FAKE_FOR_TEST"
    assert first.headers["x-vercel-protection-bypass"] == "FAKE_BYPASS_TOKEN"
    assert first.headers["x-vercel-set-bypass-cookie"] == "true"
    assert redirected.url.host == "external.test"
    assert "authorization" not in redirected.headers
    assert "x-vercel-protection-bypass" not in redirected.headers
    assert "x-vercel-set-bypass-cookie" not in redirected.headers


def test_configurable_base_url_does_not_carry_environment_key(monkeypatch, drex):
    from langchain_core.runnables import ConfigurableField

    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    classifier = NaceClassifier(
        http_client=httpx.Client(transport=drex.transport()),
        http_async_client=httpx.AsyncClient(transport=drex.async_transport()),
    )
    assert classifier.api_key is None
    configurable = classifier.configurable_fields(base_url=ConfigurableField(id="base_url"))
    with pytest.raises(ValueError, match="explicit api_key"):
        configurable.invoke(CALL, {"configurable": {"base_url": "https://openrouter.ai/api/v1"}})
    assert not drex.requests
