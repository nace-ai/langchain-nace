"""OpenRouter app attribution and key isolation at the request boundary."""

import httpx
import pytest

from langchain_nace import NaceClassifier, Noul
from tests.fakes import FakeDrex

CALL = {"state": "hello", "questions": {"q": Noul(instructions="Greeting?")}}
OPENROUTER_URLS = [
    "https://openrouter.ai/api/v1",
    "https://eu.openrouter.ai/api/v1",
    "https://EU.OpenRouter.ai./api/v1",
]


def build(drex: FakeDrex, **options: object) -> NaceClassifier:
    return NaceClassifier(http_client=httpx.Client(transport=drex.transport()), **options)  # type: ignore[arg-type]


@pytest.mark.parametrize("url", OPENROUTER_URLS)
def test_openrouter_requests_carry_the_default_app_headers(drex: FakeDrex, url: str) -> None:
    build(drex, base_url=url, api_key="sk-or-test").invoke(CALL)
    [request] = drex.requests
    assert request.headers["http-referer"] == "https://nace.ai"
    assert request.headers["x-openrouter-title"] == "Drex for LangChain"


@pytest.mark.parametrize("url", ["https://console.nace.ai", "https://gateway.example/service"])
def test_other_hosts_never_get_the_app_headers(drex: FakeDrex, url: str) -> None:
    build(drex, base_url=url, api_key="nace_sk_test").invoke(CALL)
    [request] = drex.requests
    assert "http-referer" not in request.headers
    assert "x-openrouter-title" not in request.headers


@pytest.mark.parametrize("forced", [True, False])
def test_the_app_headers_follow_the_host_not_the_format_override(
    drex: FakeDrex, forced: bool
) -> None:
    build(drex, base_url="https://gateway.example", api_key="k", openrouter=forced).invoke(CALL)
    assert "http-referer" not in drex.requests[0].headers


@pytest.mark.parametrize(
    "overrides",
    [
        {"http-referer": "https://app.example", "x-openrouter-title": "My App"},
        {"HTTP-REFERER": "https://app.example", "X-OpenRouter-Title": "My App"},
        {"Http-Referer": "https://app.example", "X-Title": "My App"},
    ],
)
def test_a_caller_override_replaces_the_default_in_any_case(
    drex: FakeDrex, overrides: dict[str, str]
) -> None:
    build(
        drex,
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        default_headers=overrides,
    ).invoke(CALL)
    [request] = drex.requests
    assert request.headers.get_list("http-referer") == ["https://app.example"]
    titles = request.headers.get_list("x-openrouter-title") + request.headers.get_list("x-title")
    assert titles == ["My App"]


def test_a_partial_override_keeps_the_other_default(drex: FakeDrex) -> None:
    build(
        drex,
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        default_headers={"http-referer": "https://app.example"},
    ).invoke(CALL)
    [request] = drex.requests
    assert request.headers.get_list("http-referer") == ["https://app.example"]
    assert request.headers["x-openrouter-title"] == "Drex for LangChain"


@pytest.mark.parametrize("url", OPENROUTER_URLS)
def test_a_nace_key_from_the_environment_never_reaches_openrouter(
    monkeypatch, drex: FakeDrex, url: str
) -> None:
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_FOR_TEST")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build(drex, base_url=url)
    assert drex.requests == []


@pytest.mark.parametrize("url", OPENROUTER_URLS)
def test_an_explicit_nace_key_never_reaches_openrouter(drex: FakeDrex, url: str) -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build(drex, base_url=url, api_key="nace_sk_FAKE_FOR_TEST")
    assert drex.requests == []


def test_openrouter_key_is_preferred_over_nace_key(monkeypatch, drex: FakeDrex) -> None:
    monkeypatch.setenv("NACE_API_KEY", "sk-or-FROM_NACE_VAR")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FROM_OPENROUTER_VAR")
    build(drex, base_url="https://eu.openrouter.ai/api/v1").invoke(CALL)
    assert drex.requests[0].headers["authorization"] == "Bearer sk-or-FROM_OPENROUTER_VAR"


def test_an_openrouter_key_kept_in_the_nace_variable_still_works(
    monkeypatch, drex: FakeDrex
) -> None:
    monkeypatch.setenv("NACE_API_KEY", "sk-or-LEGACY_PLACEMENT")
    build(drex, base_url="https://openrouter.ai/api/v1").invoke(CALL)
    assert drex.requests[0].headers["authorization"] == "Bearer sk-or-LEGACY_PLACEMENT"


def test_no_key_at_all_on_openrouter_names_the_variable(drex: FakeDrex) -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build(drex, base_url="https://openrouter.ai/api/v1")
