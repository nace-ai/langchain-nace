"""Error summaries must not expose upstream keys or credentials."""

import traceback

import httpx
import nace_sdk
import pytest

from langchain_nace import NaceAPIError, NaceError
from langchain_nace.errors import convert_error
from tests.fakes import error
from tests.unit_tests.test_batch_stop import DISPATCH_REQUEST
from tests.unit_tests.test_batch_stop import DISPATCH_REQUEST as CALL


@pytest.mark.parametrize("validation", [False, True])
def test_sdk_error_text_is_bounded_and_drops_response_keys(validation):
    secret = "response-supplied-answer-key-" * 1000
    original = (
        nace_sdk.NaceAPIResponseValidationError(200, {}, httpx.Headers(), f"answers.{secret}.noul")
        if validation
        else nace_sdk.NaceError(f"answers.{secret}: invalid response")
    )
    converted = convert_error(original)
    assert isinstance(converted, NaceError)
    assert len(str(converted)) <= 500
    assert "response-supplied-answer-key" not in str(converted)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("provider", ["Drex", "OpenRouter"])
async def test_keys_and_bearer_tokens_are_redacted(make_classifier, drex, asynchronous, provider):
    key = "opaque-configured-provider-secret"
    bearer = "opaque-token-without-a-known-prefix"
    drex.replies.append(
        error(
            401,
            f"Invalid API key: {key}; Authorization: Bearer {bearer}",
            issues=[{"path": key, "message": f"Bearer {bearer}"}],
        )
    )
    classifier = make_classifier(
        api_key=key,
        max_retries=0,
        base_url="https://openrouter.ai/api/v1"
        if provider == "OpenRouter"
        else "https://drex.test",
    )

    async def invoke():
        return await classifier.ainvoke(CALL) if asynchronous else classifier.invoke(CALL)

    with pytest.raises(NaceAPIError) as caught:
        await invoke()
    rendered = (
        str(caught.value) + repr(caught.value) + "".join(traceback.format_exception(caught.value))
    )
    assert key not in rendered
    assert bearer not in rendered


@pytest.mark.parametrize("prefix", ["<?xml version='1.0'?>", "\ufeff<html>", "<head>", "<body>"])
def test_html_variants_have_one_line_summary(prefix):
    original = nace_sdk.NaceAPIError(
        502,
        prefix + "\n<script>response-body-secret</script>\n" * 100,
        httpx.Headers(),
    )
    converted = convert_error(original)
    assert str(converted) == "Drex returned an HTML error page (HTTP 502)"
    assert "response-body-secret" not in repr(converted)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("source", ["default_headers", "client_headers", "rotating_client_headers"])
async def test_header_values_and_issues_are_redacted(make_classifier, drex, asynchronous, source):
    secret = "opaque-header-credential"
    drex.replies.append(
        error(401, f"Invalid key {secret}", issues=[{"path": secret, "message": secret}])
    )
    classifier = make_classifier(
        default_headers={"X-Custom-Credential": secret} if source == "default_headers" else {},
        max_retries=0,
    )
    if source != "default_headers":
        # Configured transport credentials may change after construction.
        classifier.http_client.headers["X-Custom-Credential"] = secret
        classifier.http_async_client.headers["X-Custom-Credential"] = secret
    if source == "rotating_client_headers":

        def rotate(body):
            classifier.http_client.headers["X-Custom-Credential"] = "rotated"
            classifier.http_async_client.headers["X-Custom-Credential"] = "rotated"
            return error(401, f"Invalid key {secret}", issues=[{"path": secret, "message": secret}])

        drex.replies[0] = rotate

    async def invoke():
        return (
            await classifier.ainvoke(DISPATCH_REQUEST)
            if asynchronous
            else classifier.invoke(DISPATCH_REQUEST)
        )

    with pytest.raises(NaceAPIError) as caught:
        await invoke()
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value.issues)


@pytest.mark.parametrize("content_type", ["text/html", "application/xhtml+xml", ""])
def test_markup_uses_content_type_or_initial_angle(content_type):
    body = "unusual gateway response" if content_type else " \ufeff \u2003<svg>private body</svg>"
    failure = nace_sdk.NaceAPIError(502, body, httpx.Headers({"content-type": content_type}))
    assert str(convert_error(failure)) == "Drex returned an HTML error page (HTTP 502)"


def test_final_message_budget_preserves_guidance():
    request_id = "r" * 128
    failure = nace_sdk.NaceAPIError(
        402, error(402, "detail " * 2000)[2], httpx.Headers({"x-request-id": request_id})
    )
    message = str(convert_error(failure))
    assert len(message) <= 600
    assert "https://console.nace.ai/dashboard/billing" in message
    assert request_id in message
