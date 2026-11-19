"""Internal helper for the NDI components: credentials, sources and job lifecycles.

Every `nace_sdk` call the NDI components make goes through `NdiClient`, so the credential
rules of `NaceClassifier` (endpoint-bound key, OpenRouter refusal, redaction) apply to
them unchanged.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar
from urllib.parse import unquote, urlsplit

import httpx
import nace_sdk
from nace_sdk.constants import API_KEY_ENV, BASE_URL_ENV, read_env
from pydantic import SecretStr

from langchain_nace.classifier import (
    CLIENT_NAME,
    DEFAULT_BASE_URL,
    MIN_TIMEOUT,
    NaceClassifier,
    _is_openrouter,
    _set_user_agent,
)
from langchain_nace.errors import (
    MAX_MESSAGE_CHARS,
    NaceAPITimeoutError,
    NaceError,
    _redact,
    convert_error,
)

logger = logging.getLogger(__name__)

PROVIDER = "NDI"
DEFAULT_JOB_TIMEOUT = 300.0
Source = str | os.PathLike[str]

T = TypeVar("T")


class NdiClient:
    """One `nace_sdk.NaceClient` plus the NDI lifecycle rules (uploads, waits, clean-up)."""

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Resolve credentials and build the SDK client; configuration errors surface here.

        Raises:
            ValueError: The endpoint is OpenRouter, is not an absolute http(s) URL,
                carries userinfo, or the key is unusable.

        """
        explicit = NaceClassifier._blank_base_url_is_default(base_url)
        endpoint = (
            explicit
            or NaceClassifier._blank_base_url_is_default(read_env(BASE_URL_ENV))
            or DEFAULT_BASE_URL
        )
        try:
            # HTTPX applies IDNA, so a host written with a Unicode dot also lands on
            # openrouter.ai; refuse that before the key lookup, as for the exact host.
            url = httpx.URL(endpoint)
        except httpx.InvalidURL as error:
            raise ValueError("base_url is not a valid URL.") from error
        if url.scheme not in ("http", "https") or not url.host:
            raise ValueError("base_url must be an absolute http(s) URL.")
        if _is_openrouter(endpoint) or url.host.removesuffix(".") == "openrouter.ai":
            message = "NDI is not available through OpenRouter; use a Nace endpoint."
            raise ValueError(message)
        # An explicit empty key stays "" (the SDK refuses it) instead of falling back to
        # the environment, exactly as in `NaceClassifier`.
        if api_key is None:
            key: str | None = read_env(API_KEY_ENV)
        else:
            key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        http = http_client or httpx.Client(timeout=MIN_TIMEOUT)
        if _set_user_agent not in http.event_hooks["request"]:
            http.event_hooks["request"].append(_set_user_agent)
        # The SDK sends the key stripped, so that is the form an error could echo.
        self._secrets = tuple(
            secret.strip() for secret in (key, read_env(API_KEY_ENV)) if secret and secret.strip()
        )
        try:
            self._client = nace_sdk.NaceClient(
                api_key=key,
                base_url=endpoint,
                retry=nace_sdk.RetryPolicy(max_retries=max_retries, timeout=None),
                timeout=MIN_TIMEOUT,
                client_info=CLIENT_NAME,
                http_client=http,
            )
        except nace_sdk.NaceError as error:
            raise ValueError(str(error)) from error

    def call(self, fn: Callable[[nace_sdk.NaceClient], T]) -> T:
        """Run one SDK call, converting its errors to this package's (redacted) errors."""
        try:
            return fn(self._client)
        except nace_sdk.NaceError as error:
            raise convert_error(error, PROVIDER, secrets=self._secrets) from None

    @contextmanager
    def source(
        self,
        source: Source,
        *,
        file_name: str | None = None,
        workspace_id: str | None = None,
    ) -> Iterator[Any]:
        """Yield a document source: a URL as given, a local file uploaded and cleaned up.

        Raises:
            ValueError: The URL is not `https://`, has no file type, or the file is missing.

        """
        text = source if isinstance(source, str) else os.fspath(source)
        if "://" in text:
            if not text.startswith("https://"):
                raise ValueError("Only https:// URLs are supported.")
            name = file_name or unquote(urlsplit(text).path.rsplit("/", 1)[-1])
            if "." not in name[1:-1]:
                message = (
                    "The URL's last path segment has no file extension, so the file type "
                    'is unknown; pass file_name (for example file_name="invoice.pdf").'
                )
                raise ValueError(message)
            yield {"type": "url", "url": text, "file_name": name}
            return
        if not os.path.isfile(text):
            raise ValueError(f"No such file: {text}")
        # Upload under a unique path: the same bytes at a fixed path resolve to the file
        # that is already there, so two loads of one file would share (and delete) it.
        upload_path = f"langchain-nace/{uuid.uuid4().hex}/{file_name or os.path.basename(text)}"
        try:
            uploaded = self.call(
                lambda client: client.documents.upload(
                    text, path=upload_path, file_name=file_name, workspace_id=workspace_id
                )
            )
        except OSError:
            raise ValueError(f"Cannot read the file: {text}") from None
        try:
            yield uploaded.as_source()
        finally:
            try:
                self.call(
                    lambda client: client.documents.delete_file(
                        uploaded.workspace_id, uploaded.file_id
                    )
                )
            except Exception:
                logger.warning("Could not delete the uploaded NDI file %s.", uploaded.file_id)

    @contextmanager
    def job(
        self,
        create: Callable[[nace_sdk.NaceClient, int], nace_sdk.DocumentJob],
        *,
        timeout: float,
    ) -> Iterator[nace_sdk.DocumentJob]:
        """Run one document job to a terminal state and delete it, also on failure.

        Raises:
            ValueError: `timeout` is not positive.
            NaceAPITimeoutError: The job did not finish within `timeout` seconds.
            NaceError: The job finished without succeeding.

        """
        if timeout <= 0:
            message = "timeout must be a positive number of seconds."
            raise ValueError(message)
        wait_seconds = max(1, min(60, int(timeout)))
        started = time.monotonic()
        job = self.call(lambda client: create(client, wait_seconds))
        try:
            if not job.is_terminal:
                remaining = max(timeout - (time.monotonic() - started), 0.0)

                def finish(client: nace_sdk.NaceClient) -> nace_sdk.DocumentJob:
                    try:
                        return client.jobs.wait(
                            job.job_id, timeout=remaining, raise_on_failure=False
                        )
                    except nace_sdk.NaceJobTimeoutError:
                        raise NaceAPITimeoutError(
                            f"The NDI job {job.job_id} did not finish in {timeout:g} s."
                        ) from None

                job = self.call(finish)
            if job.status != "succeeded":
                detail = ": ".join(
                    str(part) for part in (job.error_code, job.error_message) if part
                )
                message = f"NDI {job.kind} job {job.job_id} {job.status}: {detail or 'no detail'}"
                raise NaceError(_redact(message, self._secrets)[:MAX_MESSAGE_CHARS])
            yield job
        finally:
            try:
                self.call(lambda client: client.jobs.delete(job.job_id))
            except Exception:
                logger.warning("Could not delete the NDI job %s.", job.job_id)
