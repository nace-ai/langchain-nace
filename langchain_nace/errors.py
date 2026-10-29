"""Errors raised by langchain-nace.

API errors also inherit LangChain's `Model*Error` classes, so code that catches, for
example, `ModelRateLimitError` handles Drex the same way as other providers. Messages
never include the request or response body, so state cannot leak into logs.
"""

from __future__ import annotations

import json
import re
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


class NaceError(Exception):
    """Base class for every error this package raises."""


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
    error: nace_sdk.NaceAPIError, provider: str, message: str | None, issues: list[dict[str, str]]
) -> str:
    text = f"{provider} returned {error.status}"
    if message:
        text += f": {message}"
    lines = [f"{i.get('path') or '(request)'}: {i['message']}" for i in issues if i.get("message")]
    # Drex's top-level message often repeats a single issue; don't print it twice.
    repeated = len(issues) == 1 and issues[0].get("message") == message
    if lines and not repeated and lines != [message]:
        text += "\n" + "\n".join(f"  {line}" for line in lines)
    if error.status == 402:
        text += f"\nAdd funds at {BILLING_URL}"
    if error.request_id:
        text += f"\n(request_id={error.request_id})"
    return text


def convert_error(error: Exception, provider: str = "Drex") -> Exception:
    """Map an `nace_sdk` error to this package's error classes.

    `provider` names who answered (`"Drex"`, or `"OpenRouter"` on that backend), so
    the message points the user to the right place.
    """
    if isinstance(error, nace_sdk.NaceAPITimeoutError):
        return NaceAPITimeoutError(str(error))
    if isinstance(error, nace_sdk.NaceAPIConnectionError):
        return NaceAPIConnectionError(str(error))
    if isinstance(error, nace_sdk.NaceAPIResponseValidationError):
        return NaceAPIResponseValidationError(str(error))
    if isinstance(error, nace_sdk.NaceAPIError):
        cls = _BY_STATUS.get(error.status)
        if cls is None:
            cls = NaceInternalServerError if error.status >= 500 else NaceAPIError
        message, found = _unwrap(error.message) if error.message else (None, [])
        issues = error.issues or found
        converted = cls(error, _message(error, provider, message, issues))
        converted.issues = issues
        return converted
    if isinstance(error, nace_sdk.NaceError):
        return NaceError(str(error))
    return error


__all__ = [
    "BILLING_URL",
    "NaceAPIConnectionError",
    "NaceAPIError",
    "NaceAPIResponseValidationError",
    "NaceAPITimeoutError",
    "NaceAuthenticationError",
    "NaceBadRequestError",
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
