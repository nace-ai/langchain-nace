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
        "https://openrouter.ai./api/v1",
    ],
)
def test_openrouter_never_inherits_a_nace_key(monkeypatch, drex: FakeDrex, url: str) -> None:
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    monkeypatch.setenv("NACE_BASE_URL", url)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        NaceClassifier(http_client=httpx.Client(transport=drex.transport()))
    assert drex.requests == []


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    ("url", "legacy"),
    [
        ("https://onprem.customer.test:8443/service", False),
        ("http://127.0.0.1:8000/service", False),
        ("https://evilopenrouter.ai/api", False),
        ("https://onprem.customer.test/service", True),
    ],
)
async def test_custom_endpoint_environment_pair_works_without_explicit_key(
    monkeypatch, drex, url, legacy, async_call
):
    prefix = "DREX" if legacy else "NACE"
    if legacy:
        monkeypatch.setenv("NACE_API_KEY", " ")
        monkeypatch.setenv("NACE_BASE_URL", " ")
    else:
        monkeypatch.setenv("DREX_API_KEY", "nace_sk_FAKE_LEGACY")
        monkeypatch.setenv("DREX_BASE_URL", "https://other.customer.test")
    monkeypatch.setenv(f"{prefix}_API_KEY", "nace_sk_FAKE_ONPREM")
    monkeypatch.setenv(f"{prefix}_BASE_URL", url)
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "FAKE_BYPASS_TOKEN")
    classifier = NaceClassifier(
        default_headers={"X-Vercel-Set-Bypass-Cookie": "true"},
        http_client=httpx.Client(transport=drex.transport()),
        http_async_client=httpx.AsyncClient(transport=drex.async_transport()),
    )
    if async_call:
        await classifier.ainvoke(CALL)
    else:
        classifier.invoke(CALL)
    [request] = drex.requests
    assert str(request.url) == f"{url}/v1/systemone"
    assert request.headers["authorization"] == "Bearer nace_sk_FAKE_ONPREM"
    assert "x-vercel-protection-bypass" not in request.headers
    assert "x-vercel-set-bypass-cookie" not in request.headers
    assert "x-openrouter-title" not in request.headers
    assert classifier.api_key is None


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize("url", ["https://openrouter.ai/api/v1", "https://OPENROUTER.AI./api/v1"])
async def test_openrouter_uses_its_environment_key(monkeypatch, drex, url, async_call):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_NACE")
    monkeypatch.setenv("NACE_BASE_URL", url)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE_PROVIDER")
    classifier = NaceClassifier(
        http_client=httpx.Client(transport=drex.transport()),
        http_async_client=httpx.AsyncClient(transport=drex.async_transport()),
    )
    if async_call:
        await classifier.ainvoke(CALL)
    else:
        classifier.invoke(CALL)
    [request] = drex.requests
    assert request.headers["authorization"] == "Bearer sk-or-FAKE_PROVIDER"
    assert request.url.path == "/api/v1/systemone"


@pytest.mark.parametrize("openrouter", [False, True])
def test_key_provider_follows_explicit_endpoint_not_environment_or_format(
    monkeypatch, drex, openrouter
):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_ONPREM")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE_PROVIDER")
    monkeypatch.setenv("NACE_BASE_URL", "https://openrouter.ai/api/v1")
    classifier = NaceClassifier(
        base_url="https://onprem.customer.test/service",
        openrouter=openrouter,
        http_client=httpx.Client(transport=drex.transport()),
    )
    classifier.invoke(CALL)
    assert drex.requests[0].headers["authorization"] == "Bearer nace_sk_FAKE_ONPREM"


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
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE_NOT_EXPLICIT")
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
    assert ("OPENROUTER_API_KEY" in serialized) is openrouter
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
    with pytest.raises(ValueError, match=r"(?i)api key"):
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


def test_none_base_url_resolves_the_legacy_environment_endpoint(monkeypatch, drex):
    monkeypatch.setenv("DREX_BASE_URL", "https://evil.test")
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    classifier = NaceClassifier(
        base_url=None,
        http_client=httpx.Client(transport=drex.transport()),
    )
    classifier.invoke(CALL)
    assert drex.requests[0].url.host == "evil.test"


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
        ("https://gateway.example", "NACE_API_KEY"),
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
    assert serialized["kwargs"]["api_key"]["id"] == ["NACE_API_KEY"]
    # `load` refuses this namespace, so resolve the secret the way a reload would.
    secrets = {"NACE_API_KEY": "sk-gateway-reloaded"}
    reloaded = NaceClassifier(
        api_key=secrets[serialized["kwargs"]["api_key"]["id"][0]],
        base_url=serialized["kwargs"]["base_url"],
        openrouter=serialized["kwargs"].get("openrouter"),
        http_client=httpx.Client(transport=drex.transport()),
    )
    reloaded.invoke(CALL)
    assert drex.requests[0].headers["authorization"] == "Bearer sk-gateway-reloaded"


@pytest.mark.parametrize("host", ["console.nace.ai.", "CONSOLE.NACE.AI", "drex.stage.nace.ai:443"])
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
    with pytest.raises(ValueError, match="userinfo"):
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
@pytest.mark.parametrize("host", ["external.test", "console.test.nace.ai"])
async def test_redirect_to_another_origin_does_not_forward_credentials(
    monkeypatch, drex, async_call, host
):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "FAKE_BYPASS_TOKEN")
    drex.replies.append((307, {"location": f"https://{host}/v1/systemone"}, {}))
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
    assert redirected.url.host == host
    assert "authorization" not in redirected.headers
    assert "x-vercel-protection-bypass" not in redirected.headers
    assert "x-vercel-set-bypass-cookie" not in redirected.headers


@pytest.mark.parametrize("async_call", [False, True])
async def test_same_origin_redirect_keeps_credentials_and_existing_callbacks(
    monkeypatch, drex, async_call
):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "FAKE_BYPASS_TOKEN")
    drex.replies.append((307, {"location": "/moved/v1/systemone"}, {}))
    callbacks = []

    def observed(request):
        callbacks.append(str(request.url))

    async def aobserved(request):
        observed(request)

    with httpx.Client(
        transport=drex.transport(), follow_redirects=True, event_hooks={"request": [observed]}
    ) as sync_client:
        async with httpx.AsyncClient(
            transport=drex.async_transport(),
            follow_redirects=True,
            event_hooks={"request": [aobserved]},
        ) as async_client:
            classifier = NaceClassifier(
                http_client=sync_client,
                http_async_client=async_client,
                default_headers={"X-Vercel-Set-Bypass-Cookie": "true"},
            )
            if async_call:
                await classifier.ainvoke(CALL)
            else:
                classifier.invoke(CALL)
    assert callbacks == [str(request.url) for request in drex.requests]
    assert len(drex.requests) == 2
    for request in drex.requests:
        assert request.headers["authorization"] == "Bearer nace_sk_FAKE_FOR_TEST"
        assert request.headers["x-vercel-protection-bypass"] == "FAKE_BYPASS_TOKEN"
        assert request.headers["x-vercel-set-bypass-cookie"] == "true"


@pytest.mark.parametrize("async_call", [False, True])
async def test_http_to_https_redirect_does_not_forward_auth_or_cookies(
    monkeypatch, drex, async_call
):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_ONPREM")
    monkeypatch.setenv("NACE_BASE_URL", "http://onprem.customer.test")
    drex.replies.append((307, {"location": "https://onprem.customer.test/v1/systemone"}, {}))
    # HTTPX rebuilds Cookie on redirects from the client jar, so the jar is what
    # the origin guard must strip.
    with httpx.Client(transport=drex.transport(), follow_redirects=True) as sync_client:
        sync_client.cookies.set("session", "FAKE", domain="onprem.customer.test")
        async with httpx.AsyncClient(
            transport=drex.async_transport(), follow_redirects=True
        ) as async_client:
            async_client.cookies.set("session", "FAKE", domain="onprem.customer.test")
            classifier = NaceClassifier(http_client=sync_client, http_async_client=async_client)
            if async_call:
                await classifier.ainvoke(CALL)
            else:
                classifier.invoke(CALL)
    first, redirected = drex.requests
    assert first.headers["authorization"] == "Bearer nace_sk_FAKE_ONPREM"
    assert first.headers["cookie"] == "session=FAKE"
    assert "authorization" not in redirected.headers
    assert "cookie" not in redirected.headers


@pytest.mark.parametrize("async_call", [False, True])
@pytest.mark.parametrize(
    "url",
    [
        "http://console.nace.ai",
        "https://console.nace.ai:8443",
        "https://evil.nace.ai",
        "https://console.nace.ai.evil.test",
    ],
)
async def test_bypass_headers_only_go_to_exact_nace_https_origins(
    monkeypatch, drex, url, async_call
):
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "FAKE_BYPASS_TOKEN")
    classifier = NaceClassifier(
        api_key="sk-explicit-test",
        base_url=url,
        default_headers={"X-Vercel-Set-Bypass-Cookie": "true"},
        http_client=httpx.Client(transport=drex.transport()),
        http_async_client=httpx.AsyncClient(transport=drex.async_transport()),
    )
    if async_call:
        await classifier.ainvoke(CALL)
    else:
        classifier.invoke(CALL)
    headers = drex.requests[0].headers
    assert "x-vercel-protection-bypass" not in headers
    assert "x-vercel-set-bypass-cookie" not in headers


@pytest.mark.parametrize("host", ["console.nace.ai.", "CONSOLE.NACE.AI", "drex.stage.nace.ai:443"])
def test_trusted_host_variants_keep_bypass_header(monkeypatch, drex, host):
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "FAKE_BYPASS_TOKEN")
    classifier = NaceClassifier(
        api_key="sk-explicit-test",
        base_url=f"https://{host}",
        http_client=httpx.Client(transport=drex.transport()),
    )
    classifier.invoke(CALL)
    assert drex.requests[0].headers["x-vercel-protection-bypass"] == "FAKE_BYPASS_TOKEN"


def test_blank_openrouter_key_is_rejected(monkeypatch, drex):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    monkeypatch.setenv("OPENROUTER_API_KEY", " ")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        NaceClassifier(
            base_url="https://openrouter.ai/api/v1",
            http_client=httpx.Client(transport=drex.transport()),
        )
    assert drex.requests == []


def test_openrouter_host_with_format_disabled_still_needs_openrouter_key(monkeypatch, drex):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        NaceClassifier(
            base_url="https://openrouter.ai/api/v1",
            openrouter=False,
            http_client=httpx.Client(transport=drex.transport()),
        )
    classifier = NaceClassifier(
        api_key="sk-or-explicit-test",
        base_url="https://openrouter.ai/api/v1",
        openrouter=False,
        http_client=httpx.Client(transport=drex.transport()),
    )
    assert dumpd(classifier)["kwargs"]["api_key"]["id"] == ["OPENROUTER_API_KEY"]
    assert drex.requests == []


def test_configurable_base_url_does_not_carry_environment_key(monkeypatch, drex):
    from langchain_core.runnables import ConfigurableField

    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    classifier = NaceClassifier(
        http_client=httpx.Client(transport=drex.transport()),
        http_async_client=httpx.AsyncClient(transport=drex.async_transport()),
    )
    assert classifier.api_key is None
    configurable = classifier.configurable_fields(base_url=ConfigurableField(id="base_url"))
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        configurable.invoke(CALL, {"configurable": {"base_url": "https://openrouter.ai/api/v1"}})
    assert not drex.requests
