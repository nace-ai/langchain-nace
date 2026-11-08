import pytest
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelRateLimitError,
)

from langchain_nace import (
    NaceAuthenticationError,
    NaceClassifier,
    NaceInsufficientCreditError,
    NaceOverloadedError,
    NaceRateLimitError,
    NaceUnprocessableEntityError,
    Noul,
)
from tests.fakes import FakeDrex, error

REQUEST = {"state": "x", "questions": {"q": Noul(instructions="?")}}


def test_422_lists_each_issue(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.replies.append(
        error(
            422,
            "invalid request",
            issues=[
                {"path": "questions", "message": "at most 16 questions allowed"},
                {"path": "questions.q.criteria", "message": "must be an object"},
            ],
        )
    )
    with pytest.raises(NaceUnprocessableEntityError) as caught:
        classifier.invoke(REQUEST)
    text = str(caught.value)
    assert "questions: at most 16 questions allowed" in text
    assert "questions.q.criteria: must be an object" in text
    assert caught.value.issues[0]["path"] == "questions"
    assert caught.value.request_id == "req_test"
    assert isinstance(caught.value, ModelInvalidRequestError)
    assert len(drex.requests) == 1  # never retried


def test_402_shows_billing_link(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.replies.append(
        error(402, "This account has insufficient available funds.", kind="insufficient_credit")
    )
    with pytest.raises(
        NaceInsufficientCreditError, match=r"https://console\.nace\.ai/dashboard/billing"
    ):
        classifier.invoke(REQUEST)


def test_401_is_a_langchain_auth_error(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.replies.append(error(401, "invalid api key", kind="authentication_error"))
    with pytest.raises(NaceAuthenticationError) as caught:
        classifier.invoke(REQUEST)
    assert isinstance(caught.value, ModelAuthenticationError)


def test_429_retried_after_retry_after_ms(
    classifier: NaceClassifier, drex: FakeDrex, sleeps: list[float]
) -> None:
    drex.replies.append(
        error(429, "slow down", kind="rate_limit_error", headers={"retry-after-ms": "1500"})
    )
    response = classifier.invoke(REQUEST)
    assert response.nouls["q"].noul == 0.9
    assert len(drex.requests) == 2
    assert sleeps == [1.5]
    assert drex.requests[1].headers["x-nace-retry-count"] == "1"


async def test_529_retried_async(
    classifier: NaceClassifier, drex: FakeDrex, sleeps: list[float]
) -> None:
    drex.replies.append(
        error(529, "overloaded", kind="overloaded_error", headers={"retry-after-ms": "250"})
    )
    response = await classifier.ainvoke(REQUEST)
    assert response.nouls["q"].noul == 0.9
    assert sleeps == [0.25]


def test_retries_run_out(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.replies.extend(
        [error(529, "overloaded", kind="overloaded_error", headers={"retry-after-ms": "10"})] * 3
    )
    with pytest.raises(NaceOverloadedError):
        classifier.invoke(REQUEST)
    assert len(drex.requests) == 3


def test_rate_limit_error_class(make_classifier, drex: FakeDrex) -> None:
    drex.replies.append(error(429, "slow down", kind="rate_limit_error"))
    with pytest.raises(NaceRateLimitError) as caught:
        make_classifier(max_retries=0).invoke(REQUEST)
    assert isinstance(caught.value, ModelRateLimitError)


def test_error_text_has_no_request_body(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.replies.append(error(422, "bad", issues=[{"path": "state", "message": "too long"}]))
    with pytest.raises(NaceUnprocessableEntityError) as caught:
        classifier.invoke(
            {"state": "SECRET-STATE-VALUE", "questions": {"q": Noul(instructions="?")}}
        )
    assert "SECRET-STATE-VALUE" not in str(caught.value)
    assert "SECRET-STATE-VALUE" not in repr(caught.value)


@pytest.mark.parametrize(
    ("status", "cls_name"),
    [
        (400, "NaceBadRequestError"),
        (403, "NacePermissionDeniedError"),
        (404, "NaceNotFoundError"),
        (418, "NaceAPIError"),
    ],
)
def test_other_statuses(make_classifier, drex: FakeDrex, status: int, cls_name: str) -> None:
    import langchain_nace.errors as errors

    drex.replies.append(error(status, "nope"))
    with pytest.raises(getattr(errors, cls_name)) as caught:
        make_classifier(max_retries=0).invoke(REQUEST)
    assert caught.value.status_code == status
    assert cls_name in repr(caught.value)


def test_5xx_after_retries(make_classifier, drex: FakeDrex) -> None:
    from langchain_nace.errors import NaceInternalServerError

    drex.replies.append(error(503, "down", kind="api_error"))
    with pytest.raises(NaceInternalServerError):
        make_classifier(max_retries=0).invoke(REQUEST)


def test_connection_and_timeout_errors(sleeps: list[float]) -> None:
    import httpx
    from langchain_core.exceptions import ModelConnectionError, ModelTimeoutError

    from langchain_nace import NaceAPIConnectionError, NaceAPITimeoutError

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    for handler, cls, base in (
        (refuse, NaceAPIConnectionError, ModelConnectionError),
        (slow, NaceAPITimeoutError, ModelTimeoutError),
    ):
        client = NaceClassifier(
            api_key="nace_sk_test",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(cls) as caught:
            client.invoke(REQUEST)
        assert isinstance(caught.value, base)


def test_non_json_success_body(make_classifier, drex: FakeDrex) -> None:
    from langchain_nace import NaceError

    drex.replies.append((200, {}, ["not", "an", "object"]))
    with pytest.raises(NaceError):
        make_classifier().invoke(REQUEST)


def test_sdk_validation_error_is_converted(classifier: NaceClassifier) -> None:
    from langchain_nace import NaceError

    with pytest.raises(NaceError, match="instructions or a true/false"):
        classifier.invoke({"state": "x", "questions": {"q": Noul()}})


def test_single_issue_repeating_the_message_is_printed_once(
    classifier: NaceClassifier, drex: FakeDrex
) -> None:
    drex.replies.append(
        error(
            422,
            "request body must be at most 4194304 bytes",
            issues=[{"path": "", "message": "request body must be at most 4194304 bytes"}],
        )
    )
    with pytest.raises(NaceUnprocessableEntityError) as caught:
        classifier.invoke(REQUEST)
    assert str(caught.value).count("4194304") == 1
    assert str(caught.value).startswith("Drex returned 422")


def test_openrouter_errors_name_openrouter_and_read_cleanly(
    make_classifier, drex: FakeDrex
) -> None:
    router = make_classifier(
        model="nace-ai/drex-max", base_url="https://openrouter.ai/api/v1", max_retries=0
    )
    drex.replies.append(
        (
            422,
            {},
            {
                "error": {
                    "code": 422,
                    "message": "HTTP 422: "
                    '{"detail":"questions must be an object of 1..16 questions"}',
                }
            },
        )
    )
    with pytest.raises(NaceUnprocessableEntityError) as caught:
        router.invoke(REQUEST)
    assert str(caught.value).startswith(
        "OpenRouter returned 422: questions must be an object of 1..16 questions"
    )
    assert "HTTP 422" not in str(caught.value)


def test_openrouter_schema_issue_list_becomes_path_message(make_classifier, drex: FakeDrex) -> None:
    from langchain_nace.errors import NaceBadRequestError

    router = make_classifier(
        model="nace-ai/drex-v2.0", base_url="https://openrouter.ai/api/v1", max_retries=0
    )
    zod = '[{"code":"invalid_union","path":["questions","q","type"],"message":"Invalid input"}]'
    drex.replies.append((400, {}, {"error": {"code": 400, "message": zod}}))
    with pytest.raises(NaceBadRequestError) as caught:
        router.invoke(REQUEST)
    assert "questions.q.type: Invalid input" in str(caught.value)
    assert caught.value.issues == [{"path": "questions.q.type", "message": "Invalid input"}]
    assert "invalid_union" not in str(caught.value)


def test_retry_budget_is_disabled(classifier: NaceClassifier) -> None:
    retry = classifier._client._retry
    assert retry.timeout is None
    assert classifier._aclient._retry.timeout is None


def test_slow_failed_attempt_is_still_retried(
    classifier: NaceClassifier, drex: FakeDrex, sleeps: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 529 that arrives after 50 s must still be retried (the SDK's 30 s budget skipped it)."""
    from nace_sdk._core import retry as retry_module

    clock = {"now": 1000.0}
    monkeypatch.setattr(retry_module.time, "monotonic", lambda: clock["now"], raising=False)

    def slow_overloaded(body: dict) -> tuple:
        clock["now"] += 50.0
        return error(529, "overloaded", kind="overloaded_error", headers={"retry-after-ms": "100"})

    drex.replies.append(slow_overloaded)
    response = classifier.invoke(REQUEST)
    assert response.nouls["q"].noul == 0.9
    assert len(drex.requests) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_large_html_error_has_small_display_and_traceback(asynchronous):
    import traceback

    import httpx

    from langchain_nace import NaceInternalServerError

    html = "<!DOCTYPE html>\n<html><body>" + "gateway failure\n" * 9000 + "</body></html>"
    assert len(html) > 135_000

    def handler(request):
        return httpx.Response(
            502, text=html, headers={"content-type": "text/html", "x-request-id": "req_html"}
        )

    classifier = NaceClassifier(
        api_key="nace_sk_test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    if asynchronous:
        with pytest.raises(NaceInternalServerError) as caught:
            await classifier.ainvoke(REQUEST)
    else:
        with pytest.raises(NaceInternalServerError) as caught:
            classifier.invoke(REQUEST)
    failure = caught.value
    assert failure.body == html
    assert "HTML error page (HTTP 502)" in str(failure)
    assert str(failure).splitlines() == [
        "Drex returned an HTML error page (HTTP 502)",
        "(request_id=req_html)",
    ]
    assert len(str(failure)) < 200
    assert len(repr(failure)) < 250
    trace = "".join(traceback.format_exception(failure))
    assert len(trace) < 8000
    assert "gateway failure" not in trace
    assert failure.__cause__ is None
    assert failure.__suppress_context__


@pytest.mark.parametrize("provider", ["Drex", "OpenRouter"])
def test_long_error_is_clipped_before_guidance(provider):
    import httpx
    import nace_sdk

    from langchain_nace.errors import convert_error

    body = {"error": {"message": "detail " * 2000}}
    sdk_error = nace_sdk.NaceAPIError(402, body, httpx.Headers({"x-request-id": "req_long"}))
    converted = convert_error(sdk_error, provider)
    summary, guidance, request_id = str(converted).splitlines()
    assert len(summary) == 500
    assert summary.endswith("…")
    assert converted.body is body
    link = (
        "https://openrouter.ai/settings/credits"
        if provider == "OpenRouter"
        else "https://console.nace.ai/dashboard/billing"
    )
    assert link in guidance
    assert request_id == "(request_id=req_long)"


def test_openrouter_402_links_to_provider_credits(make_classifier, drex):
    router = make_classifier(base_url="https://openrouter.ai/api/v1", max_retries=0)
    drex.replies.append(error(402, "not enough credits"))
    with pytest.raises(NaceInsufficientCreditError) as caught:
        router.invoke(REQUEST)
    assert "https://openrouter.ai/settings/credits" in str(caught.value)
    assert "console.nace.ai" not in str(caught.value)


def test_long_issues_are_clipped_but_preserved():
    import httpx
    import nace_sdk

    from langchain_nace.errors import convert_error

    issues = [{"path": "state", "message": "invalid " * 2000}]
    original = nace_sdk.NaceAPIError(
        422, {"error": {"message": "bad", "issues": issues}}, httpx.Headers()
    )
    converted = convert_error(original)
    assert len(str(converted)) == 500
    assert converted.issues == issues
