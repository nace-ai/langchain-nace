"""Internal helper for the NDI components: credentials, sources and job lifecycles.

Every `nace_sdk` call the NDI components make goes through `NdiClient`, so the credential
rules of `NaceClassifier` (endpoint-bound key, OpenRouter refusal, redaction) apply to
them unchanged. A component's `timeout` bounds its whole job: the create request, the
wait and a truncated-content download.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Queue
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
    NaceAPIError,
    NaceAPITimeoutError,
    NaceError,
    _redact,
    convert_error,
)

logger = logging.getLogger(__name__)

PROVIDER = "NDI"
DEFAULT_JOB_TIMEOUT = 300.0
Source = str | os.PathLike[str]

#: The largest hold a create asks the server for (the SDK's cap).
CREATE_HOLD_CAP = 60
#: The largest hold a job read asks the server for (the SDK's cap).
MAX_HOLD_SECONDS = 25
#: The SDK raises a held read's HTTP timeout to at least this many seconds, so below
#: this much budget left the poll loop reads without a hold instead.
HELD_READ_FLOOR = 95.0
#: An unheld poll waits this long before asking again, doubling up to the maximum.
POLL_SECONDS = 1.0
MAX_POLL_SECONDS = 10.0

T = TypeVar("T")

#: The SDK emits its wire log, with request and response bodies, on this logger.
SDK_LOGGER_NAME = "nace_sdk"


class _SdkLogRedactor(logging.Filter):
    """Redact the keys NDI clients were configured with from `nace_sdk` log records.

    The SDK's own filter masks credential headers and known secret body fields, but a
    server can echo a key inside an error message; this filter removes the configured
    keys from the rendered text of every record. It is attached after the SDK's filter,
    so bodies are already rendered as strings when it runs.
    """

    def __init__(self) -> None:
        super().__init__()
        self._secrets: tuple[str, ...] = ()

    def add_secrets(self, secrets: Sequence[str]) -> None:
        """Remember more keys to redact; called as clients are configured."""
        if secrets:
            self._secrets = self._secrets + tuple(secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact the configured keys from a record's message and arguments."""
        secrets = self._secrets
        if not secrets:
            return True
        if isinstance(record.msg, str):
            record.msg = _redact(record.msg, secrets)
        args = record.args
        if isinstance(args, tuple):
            record.args = tuple(
                _redact(arg, secrets) if isinstance(arg, str) else arg for arg in args
            )
        elif isinstance(args, dict):
            record.args = {
                key: _redact(value, secrets) if isinstance(value, str) else value
                for key, value in args.items()
            }
        return True


_SDK_LOG_REDACTOR = _SdkLogRedactor()


def _redact_body(value: Any, secrets: Sequence[str]) -> Any:
    """Redact every string inside a JSON-like error body."""
    if isinstance(value, str):
        return _redact(value, secrets)
    if isinstance(value, dict):
        return {key: _redact_body(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_body(item, secrets) for item in value]
    return value


def _is_job_pending(error: NaceAPIError) -> bool:
    """Whether a failed document create may still run server-side.

    The SDK documents the case: a `503` with code `job_pending` (always carrying an
    `Idempotency-Key`, as every create does) may still start the job, and resending
    the same body with that key resumes it instead of starting another.
    """
    return error.status == 503 and error.code == "job_pending" and error.idempotency_key is not None


class NdiClient:
    """One `nace_sdk.NaceClient` plus the NDI lifecycle rules (uploads, waits, clean-up)."""

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
        retry_timeout: float | None = None,
    ) -> None:
        """Resolve credentials and build the SDK client; configuration errors surface here.

        `retry_timeout` bounds each SDK call's retries (attempts plus backoff), so a
        busy server cannot retry a request past the component's job budget.

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
        self._max_retries = max_retries
        # A server can echo a key inside an error body, and the SDK's DEBUG wire log
        # prints that body verbatim (its own filter masks only credential headers and
        # secret body fields); keep the configured keys out of that log too.
        _SDK_LOG_REDACTOR.add_secrets(self._secrets)
        sdk_logger = logging.getLogger(SDK_LOGGER_NAME)
        if _SDK_LOG_REDACTOR not in sdk_logger.filters:
            sdk_logger.addFilter(_SDK_LOG_REDACTOR)
        try:
            self._client = nace_sdk.NaceClient(
                api_key=key,
                base_url=endpoint,
                retry=nace_sdk.RetryPolicy(max_retries=max_retries, timeout=retry_timeout),
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
            converted = convert_error(error, PROVIDER, secrets=self._secrets)
            if isinstance(converted, NaceAPIError):
                # The wrapper keeps the upstream body for inspection; redact it like
                # the message, so a key the server echoed does not survive there.
                converted.body = _redact_body(converted.body, self._secrets)
            try:
                raise converted from None
            finally:
                # Raising inside this handler re-attaches the raw SDK error, whose
                # body and message can echo a key; detach it again.
                converted.__context__ = None

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
        keep_file = False
        try:
            yield uploaded.as_source()
        except NaceAPIError as error:
            if _is_job_pending(error):
                # The pending job and its documented replay need the uploaded file:
                # keep it, and tell the user where it is and how to delete it.
                kept = (
                    f"The uploaded NDI file was kept (workspace {uploaded.workspace_id}, "
                    f"file {uploaded.file_id}): a 503 job_pending may still run, and "
                    f"replaying it with idempotency key {error.idempotency_key} needs "
                    "the file. Delete it with nace_sdk once you are done retrying: "
                    f"client.documents.delete_file({uploaded.workspace_id!r}, "
                    f"{uploaded.file_id!r})."
                )
                error.args = (f"{error}\n{kept}",)
                keep_file = True
            raise
        finally:
            if not keep_file:
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
        create: Callable[[nace_sdk.NaceClient, int | None, float], nace_sdk.DocumentJob],
        *,
        timeout: float,
    ) -> Iterator[nace_sdk.DocumentJob]:
        """Run one document job to a terminal state and delete it, also on failure.

        `timeout` bounds the whole job, the create request included. The create's hold
        and its HTTP timeout share the budget (the SDK extends the request's timeout
        by the hold), and every poll carries the time left as its HTTP timeout, with
        its retries stopped at the same point, so no request can outlast the deadline
        and none is sent after it. A job that is not done by then raises
        `NaceAPITimeoutError`, and the clean-up cancels it.

        `create` receives the SDK client, the `wait_seconds` its request asks the
        server to hold (`None` when the budget is shorter than the one-second minimum),
        and the per-call HTTP timeout.

        Raises:
            ValueError: `timeout` is not positive.
            NaceAPITimeoutError: The job did not finish within `timeout` seconds.
            NaceError: The job finished without succeeding.

        """
        if timeout <= 0:
            message = "timeout must be a positive number of seconds."
            raise ValueError(message)
        deadline = time.monotonic() + timeout
        # The hold and the HTTP timeout share the budget: the SDK extends the
        # request's timeout by the hold, so `timeout - hold` caps the request at the
        # budget. A budget under two seconds asks for no hold at all.
        wait_seconds = min(CREATE_HOLD_CAP, int(timeout / 2)) or None
        job = self.call(lambda client: create(client, wait_seconds, timeout - (wait_seconds or 0)))
        try:
            if not job.is_terminal:
                job = self._poll(job, deadline=deadline, timeout=timeout)
            if time.monotonic() > deadline:
                raise NaceAPITimeoutError(
                    f"The NDI job {job.job_id} did not finish in {timeout:g} s."
                )
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

    def _bounded_retry(self, budget: float) -> nace_sdk.RetryPolicy:
        """The configured retries, stopped before they reach `budget` seconds."""
        return nace_sdk.RetryPolicy(max_retries=self._max_retries, timeout=budget)

    def _read(self, job_id: str, *, hold: int, timeout: float) -> nace_sdk.DocumentJob:
        """One job read: held `hold` seconds at most, with HTTP timeout and retries in `timeout`."""
        return self.call(
            lambda client: client.jobs.get(
                job_id, wait_seconds=hold, timeout=timeout, retry=self._bounded_retry(timeout)
            )
        )

    def _poll(
        self, job: nace_sdk.DocumentJob, *, deadline: float, timeout: float
    ) -> nace_sdk.DocumentJob:
        """Read the job until it is terminal, never reading past `deadline`.

        While at least the SDK's floor for held reads remains, each read asks the
        server to hold the job and carries the time left as its HTTP timeout. Below
        that floor the SDK would raise a held read's timeout back to the floor, so the
        loop polls without a hold, backing off from one second to ten between reads.
        """
        interval = POLL_SECONDS
        while not job.is_terminal:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NaceAPITimeoutError(
                    f"The NDI job {job.job_id} did not finish in {timeout:g} s."
                )
            hold = min(MAX_HOLD_SECONDS, int(remaining)) if remaining >= HELD_READ_FLOOR else 0
            job = self._read(job.job_id, hold=hold, timeout=remaining)
            if job.is_terminal:
                break
            time.sleep(min(interval, max(deadline - time.monotonic(), 0.0)))
            interval = min(interval * 2, MAX_POLL_SECONDS)
        return job

    def download(self, job_id: str, path: str, destination: Path, *, timeout: float) -> Path:
        """Save one file of a job, abandoning a transfer stalled past `timeout` seconds.

        The SDK streams a job file with no read timeout, so a stalled server would hold
        the stream — and the job's clean-up — open forever. Past the budget the transfer
        is left to a daemon thread and `NaceAPITimeoutError` is raised; the caller's job
        clean-up then cancels the job, which ends the stream.

        Raises:
            NaceAPITimeoutError: The download did not finish within `timeout` seconds.

        """
        budget = max(timeout, 0.0)
        box: Queue[Path | BaseException] = Queue()

        def fetch() -> None:
            try:
                box.put(
                    self.call(
                        lambda client: client.jobs.download(
                            job_id, path, destination, wait_timeout=budget
                        )
                    )
                )
            except Exception as error:
                box.put(error)

        thread = threading.Thread(target=fetch, name="langchain-nace-ndi-download", daemon=True)
        thread.start()
        try:
            item = box.get(timeout=budget)
        except Empty:
            raise NaceAPITimeoutError(
                f"The download of the NDI job {job_id} file did not finish in {timeout:g} s."
            ) from None
        if isinstance(item, BaseException):
            raise item
        return item
