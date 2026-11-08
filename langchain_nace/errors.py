"""Errors raised by langchain-nace.

API errors also inherit LangChain's `Model*Error` classes, so code that catches, for
example, `ModelRateLimitError` handles Drex the same way as other providers. Messages
are bounded summaries; the complete upstream response remains available on `.body`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

import nace_sdk
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)

BILLING_URL = "https://console.nace.ai/dashboard/billing"
OPENROUTER_CREDITS_URL = "https://openrouter.ai/settings/credits"
MAX_MESSAGE_CHARS = 500
MAX_FINAL_MESSAGE_CHARS = 600
MAX_REQUEST_ID_CHARS = 384


class NaceError(Exception):
    """Base class for every error this package raises."""


class NaceBatchStoppedError(NaceError):
    """An input was not sent because an earlier request stopped its batch.

    `__cause__` holds the failure that stopped dispatch.
    """


class NaceImageError(NaceError, ValueError):
    """An image can't be sent as given (a remote URL, an unknown block, ...)."""


class NaceImageModelError(NaceImageError):
    """Images were sent to a model that reads text only."""


class NaceMediaError(NaceError, ValueError):
    """State holds media Drex can't read (audio, video, files)."""


class NaceAPIError(NaceError):
    """Drex answered with an unsuccessful HTTP status."""

    def __init__(self, error: nace_sdk.NaceAPIError, message: str | None = None) -> None:
        """Wrap an SDK error, keeping its status, request id and issues."""
        super().__init__(message or str(error))
        self.status: int = error.status
        self.request_id: str | None = error.request_id
        self.type: str | None = error.type
        self.issues: list[dict[str, str]] = error.issues
        self.headers = error.headers
        self.body: Any = error.body

    @property
    def status_code(self) -> int:
        """Alias for `status`."""
        return self.status

    def __repr__(self) -> str:
        """Describe the error without the response body."""
        return f"{type(self).__name__}({str(self)!r})"


class NaceBadRequestError(NaceAPIError, ModelInvalidRequestError):
    """HTTP 400."""


class NaceAuthenticationError(NaceAPIError, ModelAuthenticationError):
    """HTTP 401: the API key is missing, wrong or revoked."""


class NaceInsufficientCreditError(NaceAPIError, ModelAPIError):
    """HTTP 402: the account has no available funds."""

    is_retryable = False


class NacePermissionDeniedError(NaceAPIError, ModelPermissionDeniedError):
    """HTTP 403."""


class NaceNotFoundError(NaceAPIError, ModelNotFoundError):
    """HTTP 404."""


class NaceUnprocessableEntityError(NaceAPIError, ModelInvalidRequestError):
    """HTTP 422: Drex rejected the request; `issues` says where."""


class NaceRateLimitError(NaceAPIError, ModelRateLimitError):
    """HTTP 429, after the retries ran out."""


class NaceInternalServerError(NaceAPIError, ModelAPIError):
    """HTTP 5xx, after the retries ran out."""


class NaceOverloadedError(NaceInternalServerError):
    """HTTP 529: Drex is overloaded; retried with `retry-after-ms` before raising."""


class NaceAPIConnectionError(NaceError, ModelConnectionError):
    """Drex could not be reached."""


class NaceAPITimeoutError(NaceAPIConnectionError, ModelTimeoutError):
    """The request timed out."""


class NaceAPIResponseValidationError(NaceError):
    """Drex answered 200 with a body this package can't read."""


_BY_STATUS: dict[int, type[NaceAPIError]] = {
    400: NaceBadRequestError,
    401: NaceAuthenticationError,
    402: NaceInsufficientCreditError,
    403: NacePermissionDeniedError,
    404: NaceNotFoundError,
    422: NaceUnprocessableEntityError,
    429: NaceRateLimitError,
    529: NaceOverloadedError,
}


def _unwrap(message: str) -> tuple[str, list[dict[str, str]]]:
    """Make a gateway's raw error text readable.

    OpenRouter passes upstream errors through as text such as
    `HTTP 422: {"detail": "..."}` or a JSON list of schema issues. Return a short
    message and any `{"path", "message"}` issues found in it.
    """
    text = message.strip()
    wrapped = re.match(r"^HTTP \d{3}: (.*)$", text, re.DOTALL)
    if wrapped:
        text = wrapped.group(1).strip()
    try:
        data: Any = json.loads(text)
    except ValueError:
        return text, []
    if isinstance(data, dict):
        inner = data.get("detail") or data.get("message") or data.get("error")
        if isinstance(inner, str):
            return inner, []
        if isinstance(inner, dict) and isinstance(inner.get("message"), str):
            return inner["message"], []
        data = inner
    if isinstance(data, list):
        issues = []
        for item in data:
            if isinstance(item, dict) and item.get("message"):
                path = item.get("path", "")
                if isinstance(path, list):
                    path = ".".join(str(part) for part in path)
                issues.append({"path": str(path), "message": str(item["message"])})
        if issues:
            return "the request is invalid", issues
    return text, []


def _message(
    error: nace_sdk.NaceAPIError,
    provider: str,
    message: str | None,
    issues: list[dict[str, str]],
    *,
    secrets: Sequence[str] = (),
) -> str:
    content_type = error.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    html = content_type in {"text/html", "application/xhtml+xml"} or bool(
        message and re.match(r"[\s\ufeff]*<", message)
    )
    error_provider = provider
    provider = _redact(provider, secrets)
    message = _redact(message, secrets) if message else None
    issues = [{key: _redact(value, secrets) for key, value in issue.items()} for issue in issues]
    text = (
        f"{provider} returned an HTML error page (HTTP {error.status})"
        if html
        else f"{provider} returned {error.status}"
    )
    if message and not html:
        text += f": {message}"
    lines = [f"{i.get('path') or '(request)'}: {i['message']}" for i in issues if i.get("message")]
    # Drex's top-level message often repeats a single issue; don't print it twice.
    repeated = len(issues) == 1 and issues[0].get("message") == message
    if lines and not html and not repeated and lines != [message]:
        text += "\n" + "\n".join(f"  {line}" for line in lines)
    suffix = ""
    if error.status == 402:
        link = OPENROUTER_CREDITS_URL if error_provider == "OpenRouter" else BILLING_URL
        suffix += f"\nAdd funds at {link}"
    if error.request_id and re.fullmatch(r"[A-Za-z0-9._:-]+", error.request_id):
        request_id = _redact(error.request_id, secrets)
        if len(request_id) > MAX_REQUEST_ID_CHARS:
            request_id = request_id[: MAX_REQUEST_ID_CHARS - 1] + "…"
        suffix += f"\n(request_id={request_id})"
    budget = min(MAX_MESSAGE_CHARS, MAX_FINAL_MESSAGE_CHARS - len(suffix))
    if len(text) > budget:
        text = text[: budget - 1] + "…"
    return text + suffix


def _redact(text: str, secrets: Sequence[str]) -> str:
    patterns = [
        r"(?i:\bBearer\s+[A-Za-z0-9._~+/-]+=*)",
        r"\b(?:nace_sk_|sk-or-|sk-)[A-Za-z0-9_-]+",
        *(re.escape(secret) for secret in sorted(set(secrets), key=len, reverse=True) if secret),
    ]

    def replace(match: re.Match[str]) -> str:
        return "Bearer [REDACTED]" if re.match(r"(?i)Bearer\s", match[0]) else "[REDACTED]"

    # One pass: a later secret must not expand a replacement inserted earlier.
    return re.sub("|".join(patterns), replace, text)


def _sdk_summary(error: Exception, provider: str) -> str:
    # Keep known local guidance, without interpolated question ids or SDK paths.
    if re.fullmatch(
        r'Noul question ".*" needs instructions or a true/false criterion\.', str(error), re.DOTALL
    ):
        return "A Noul question needs instructions or a true/false criterion."
    return f"{provider} could not complete the request."


def convert_error(
    error: Exception, provider: str = "Drex", *, secrets: Sequence[str] = ()
) -> Exception:
    """Map an `nace_sdk` error to this package's error classes.

    `provider` names who answered (`"Drex"`, or `"OpenRouter"` on that backend), so
    the message points the user to the right place.
    """
    if isinstance(error, nace_sdk.NaceAPITimeoutError):
        return NaceAPITimeoutError(f"The request to {provider} timed out.")
    if isinstance(error, nace_sdk.NaceAPIConnectionError):
        return NaceAPIConnectionError(f"Could not connect to {provider}.")
    if isinstance(error, nace_sdk.NaceAPIResponseValidationError):
        return NaceAPIResponseValidationError(f"{provider} returned an invalid response.")
    if isinstance(error, nace_sdk.NaceAPIError):
        cls = _BY_STATUS.get(error.status)
        if cls is None:
            cls = NaceInternalServerError if error.status >= 500 else NaceAPIError
        message, found = _unwrap(error.message) if error.message else (None, [])
        issues = error.issues or found
        safe_issues = [
            {key: _redact(value, secrets) for key, value in issue.items()} for issue in issues
        ]
        text = _message(error, provider, message, issues, secrets=secrets)
        converted = cls(error, text)
        converted.issues = safe_issues
        return converted
    if isinstance(error, nace_sdk.NaceError):
        return NaceError(_redact(_sdk_summary(error, provider), secrets))
    return error


__all__ = [
    "BILLING_URL",
    "OPENROUTER_CREDITS_URL",
    "NaceAPIConnectionError",
    "NaceAPIError",
    "NaceAPIResponseValidationError",
    "NaceAPITimeoutError",
    "NaceAuthenticationError",
    "NaceBadRequestError",
    "NaceBatchStoppedError",
    "NaceError",
    "NaceImageError",
    "NaceImageModelError",
    "NaceInsufficientCreditError",
    "NaceInternalServerError",
    "NaceMediaError",
    "NaceNotFoundError",
    "NaceOverloadedError",
    "NacePermissionDeniedError",
    "NaceRateLimitError",
    "NaceUnprocessableEntityError",
    "convert_error",
]
