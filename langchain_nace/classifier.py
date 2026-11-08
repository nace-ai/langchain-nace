"""`NaceClassifier`: Drex decisions as a LangChain Runnable."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Sequence
from concurrent.futures import CancelledError
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, NoReturn, cast
from urllib.parse import urlsplit

import httpx
import nace_sdk
from langchain_core._api import beta
from langchain_core.callbacks import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import (
    ensure_config,
    get_config_list,
    get_executor_for_config,
)
from langchain_core.utils import from_env, secret_from_env
from langsmith.run_helpers import get_current_run_tree
from nace_sdk.constants import API_KEY_ENV, BASE_URL_ENV, read_env
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr, field_validator
from typing_extensions import override

from langchain_nace._images import as_data_url, to_drex_image
from langchain_nace._state import encode_state
from langchain_nace._version import __version__
from langchain_nace.errors import (
    NaceAPIError,
    NaceBatchStoppedError,
    NaceError,
    NaceImageError,
    NaceImageModelError,
    convert_error,
)
from langchain_nace.types import ClassifierRequest, ClassifierResponse, to_wire

logger = logging.getLogger(__name__)
_request_error_headers: ContextVar[list[str] | None] = ContextVar(
    "nace_request_error_headers", default=None
)

# The default model is Drex Max, moved to the newest flagship with each Drex release.
# Users pin a model with `model=` or NACE_DEFAULT_DECISION_MODEL; nothing else in the
# package depends on this value.
DEFAULT_MODEL = "drex-max"
CLIENT_NAME = f"langchain-nace/{__version__}"
MIN_TIMEOUT = 60.0
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_BASE_URL = "https://console.nace.ai"
NACE_HOSTS = frozenset(
    {
        "console.nace.ai",
        "drex.nace.ai",
    }
)


def _is_nace_destination(base_url: str | None) -> bool:
    url = urlsplit(base_url or DEFAULT_BASE_URL)
    return (
        url.scheme == "https"
        and (url.hostname or "").removesuffix(".") in NACE_HOSTS
        and url.port in (None, 443)
        and url.username is None
        and url.password is None
    )


# Facts about released, pinned versions never change, so this list can't go stale.
# Every other model id (new releases, aliases such as drex-latest) is sent as given
# and Drex decides.
TEXT_ONLY_MODELS = frozenset({"drex-v1.5", "drex-v2.0"})


def is_text_only(model: str) -> bool:
    """Whether `model` is a released version known to read text only."""
    return model.rsplit("/", 1)[-1].lstrip("~") in TEXT_ONLY_MODELS


def _is_openrouter(base_url: str | None) -> bool:
    if not base_url:
        return False
    return (urlsplit(base_url).hostname or "").removesuffix(".") == "openrouter.ai"


def _set_user_agent(request: httpx.Request) -> None:
    request.headers["User-Agent"] = CLIENT_NAME
    origin = (
        request.url.scheme,
        request.url.host.removesuffix("."),
        request.url.port,
    )
    # HTTPX carries extensions to redirect requests. Bind credentials to the
    # first request's origin, including HTTPS upgrades and other Nace hosts.
    original_origin = request.extensions.setdefault("langchain_nace.original_origin", origin)
    if origin != original_origin:
        request.headers.pop("Authorization", None)
        request.headers.pop("Cookie", None)
        request.headers.pop("x-vercel-protection-bypass", None)
        request.headers.pop("x-vercel-set-bypass-cookie", None)
    # Check the final request URL, including redirected requests and custom clients.
    # Older nace-sdk versions add this environment token for every destination.
    if not _is_nace_destination(str(request.url)):
        request.headers.pop("x-vercel-protection-bypass", None)
        request.headers.pop("x-vercel-set-bypass-cookie", None)
    captured = _request_error_headers.get()
    if captured is not None:
        # Capture what was sent, including auth hooks and credentials rotated
        # while a request was in flight. Generated routing/length values are not keys.
        captured.extend(
            value
            for name, value in request.headers.items()
            if name.lower() not in {"host", "content-length"}
        )
    # The SDK has handed this registered input to HTTPX. Release the stop lock
    # before I/O, while the call and its SDK retries remain in flight.
    dispatch = _dispatch_context.get()
    if dispatch is not None:
        dispatch.release()


async def _aset_user_agent(request: httpx.Request) -> None:
    _set_user_agent(request)


def _with_max_concurrency(
    config: RunnableConfig | Sequence[RunnableConfig] | None,
) -> RunnableConfig | list[RunnableConfig]:
    if config is None:
        return {"max_concurrency": DEFAULT_MAX_CONCURRENCY}
    if isinstance(config, dict):
        if config.get("max_concurrency"):
            return config
        return {**config, "max_concurrency": DEFAULT_MAX_CONCURRENCY}
    return [
        c if c.get("max_concurrency") else {**c, "max_concurrency": DEFAULT_MAX_CONCURRENCY}
        for c in config
    ]


@dataclass
class _BatchState:
    owner: NaceClassifier
    return_exceptions: bool
    failure: Exception | None = None
    lock: Lock = field(default_factory=Lock)

    def record(self, error: Exception) -> None:
        if not self.return_exceptions or (
            isinstance(error, NaceAPIError) and error.status in (401, 402, 403)
        ):
            with self.lock:
                if self.failure is None:
                    self.failure = error

    def check(self) -> None:
        if self.failure is not None:
            self.reject()

    def reject(self, _: ClassifierRequest | None = None) -> NoReturn:
        raise NaceBatchStoppedError(
            "Not sent: an earlier request in this batch failed."
        ) from self.failure


_batch_context: ContextVar[_BatchState | None] = ContextVar("nace_batch", default=None)
_batch_invocation: ContextVar[_BatchState | None] = ContextVar(
    "nace_batch_invocation", default=None
)


@dataclass
class _Dispatch:
    batch: _BatchState
    locked: bool = True

    def release(self) -> None:
        if self.locked:
            self.locked = False
            self.batch.lock.release()


_dispatch_context: ContextVar[_Dispatch | None] = ContextVar("nace_dispatch", default=None)


@beta()
class NaceClassifier(RunnableSerializable[ClassifierRequest, ClassifierResponse]):
    """Ask Drex typed questions about text, JSON, messages or images.

    Questions about one state are answered in one request. Answers keep their full
    probabilities, so thresholds stay in your code.

    Any model id is accepted and sent as given; Drex enforces each model's limits and
    the error explains what to change. Images are refused before sending only for
    models known to read text only (`drex-v1.5`, `drex-v2.0`).

    Example:
        ```python
        from langchain_nace import Choice, NaceClassifier, Noul

        classifier = NaceClassifier()  # drex-max; reads NACE_API_KEY
        response = classifier.invoke(
            {
                "state": "I was charged twice for March. Please refund one.",
                "questions": {
                    "refund": Noul(instructions="Is the customer asking for a refund?"),
                    "team": Choice(
                        instructions="Which team should handle this?",
                        criteria={"billing": "Payments", "technical": "Bugs"},
                    ),
                },
            }
        )
        response.nouls["refund"].noul          # probability of yes
        response.choices["team"].probabilities  # every label's probability
        ```

    Args:
        model: Drex model id. Defaults to `NACE_DEFAULT_DECISION_MODEL`, else
            `drex-max` (reads images; up to 16 questions per request). Use
            `drex-v2.0` for text-only batches of up to 512 questions.
        api_key: Provider key. Defaults to `NACE_API_KEY` (legacy `DREX_API_KEY`)
            for the configured endpoint. OpenRouter uses `OPENROUTER_API_KEY`.
        base_url: API root. Defaults to `NACE_BASE_URL`, else `https://console.nace.ai`.
            Also works with OpenRouter (`https://openrouter.ai/api`), the Vercel AI
            Gateway (`.../typesafe`) and other TypeSafe-compatible hosts.
        timeout: Seconds per request; at least 60, because Drex may take 55 s.
        max_retries: Retries for 408, 429, 5xx and 529, honouring `retry-after-ms`.
        openrouter: Send OpenRouter's request format. Inferred from `base_url`.
        model_kwargs: Extra top-level body fields, for options Drex adds later.

    """

    model: str = Field(
        default_factory=from_env("NACE_DEFAULT_DECISION_MODEL", default=DEFAULT_MODEL),
        min_length=1,
    )
    api_key: SecretStr | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    base_url: str | None = Field(
        default_factory=lambda: read_env(BASE_URL_ENV), validate_default=True
    )
    timeout: float = Field(default=MIN_TIMEOUT, ge=MIN_TIMEOUT)
    max_retries: int = Field(default=2, ge=0)
    openrouter: bool | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    default_headers: dict[str, str] | None = Field(default=None, exclude=True, repr=False)
    http_client: httpx.Client | None = Field(default=None, exclude=True, repr=False)
    """Optional `httpx.Client` (custom transport, proxy, tests)."""
    http_async_client: httpx.AsyncClient | None = Field(default=None, exclude=True, repr=False)
    """Optional `httpx.AsyncClient`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _client: nace_sdk.NaceClient = PrivateAttr()
    _aclient: nace_sdk.AsyncNaceClient = PrivateAttr()
    _effective_base_url: str = PrivateAttr()
    _redaction_secrets: tuple[str, ...] = PrivateAttr(default=())

    @field_validator("base_url")
    @classmethod
    def _blank_base_url_is_default(cls, base_url: str | None) -> str | None:
        # An empty NACE_BASE_URL (common in CI and .env files) means "use the default".
        value = base_url.strip() or None if base_url is not None else None
        if value is not None:
            parsed = urlsplit(value)
            if parsed.username is not None or parsed.password is not None:
                message = "base_url must not contain userinfo; pass credentials using api_key."
                raise ValueError(message)
        return value

    @field_validator("model")
    @classmethod
    def _strip_model(cls, model: str) -> str:
        model = model.strip()
        if not model:
            message = "model must not be empty."
            raise ValueError(message)
        return model

    @override
    def model_post_init(self, context: Any, /) -> None:
        self._effective_base_url = (
            self._blank_base_url_is_default(
                self.base_url or read_env(BASE_URL_ENV) or DEFAULT_BASE_URL
            )
            or DEFAULT_BASE_URL
        )
        env_key: SecretStr | None = None
        if self.api_key is None:
            # Keep the ambient key out of the public `api_key` field: LangChain rebuilds
            # configurable models from the fields, which would make it look explicit.
            if _is_openrouter(self._effective_base_url):
                env_key = secret_from_env("OPENROUTER_API_KEY", default=None)()
                if env_key is None or not env_key.get_secret_value().strip():
                    message = "OpenRouter requires api_key or OPENROUTER_API_KEY."
                    raise ValueError(message)
            elif value := read_env(API_KEY_ENV):
                env_key = SecretStr(value)
        headers = dict(self.default_headers or {})
        if self._openrouter:
            headers.setdefault("HTTP-Referer", "https://nace.ai")
            headers.setdefault("X-OpenRouter-Title", "Drex for LangChain")
        # The SDK's total retry budget defaults to 30 s, which skips the retry after a
        # slow failed attempt (Drex may take 55 s). `timeout` already bounds each
        # attempt and `max_retries` their number, so the budget is disabled, as in
        # deepset's typesafe-haystack and nace-haystack.
        retry = nace_sdk.RetryPolicy(max_retries=self.max_retries, timeout=None)
        # `SecretStr("")` is falsy; an explicit empty key must reach the SDK as "" (which
        # it rejects) rather than become None and fall back to the SDK's own env lookup.
        secret = self.api_key if self.api_key is not None else env_key
        key = secret.get_secret_value() if secret is not None else None
        self._redaction_secrets = tuple(
            value
            for value in (
                key,
                read_env(API_KEY_ENV),
                os.environ.get("OPENROUTER_API_KEY"),
                *headers.values(),
                *(
                    value.split(" ", 1)[-1]
                    for name, value in headers.items()
                    if name.lower() in {"authorization", "proxy-authorization"}
                ),
            )
            if value
        )
        sync_http = self.http_client or httpx.Client(timeout=self.timeout)
        if _set_user_agent not in sync_http.event_hooks["request"]:
            sync_http.event_hooks["request"].append(_set_user_agent)
        async_http = self.http_async_client or httpx.AsyncClient(timeout=self.timeout)
        if _aset_user_agent not in async_http.event_hooks["request"]:
            async_http.event_hooks["request"].append(_aset_user_agent)
        options: dict[str, Any] = {
            "api_key": key,
            "base_url": self._base_url,
            "retry": retry,
            "timeout": self.timeout,
            "headers": headers,
            "client_info": CLIENT_NAME,
        }
        try:
            self._client = nace_sdk.NaceClient(**options, http_client=sync_http)
            self._aclient = nace_sdk.AsyncNaceClient(**options, http_client=async_http)
        except nace_sdk.NaceError as error:
            raise ValueError(str(error)) from error

    @property
    def _openrouter(self) -> bool:
        return (
            self.openrouter
            if self.openrouter is not None
            else _is_openrouter(self._effective_base_url)
        )

    @property
    def _provider(self) -> str:
        return "OpenRouter" if self._openrouter else "Drex"

    @property
    def _base_url(self) -> str | None:
        if self._openrouter:
            # OpenRouter documents `.../api/v1`; the SDK appends `/v1/systemone` itself.
            return self._effective_base_url.rstrip("/").removesuffix("/v1")
        return self._effective_base_url

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain", "classifiers", "nace"]

    @property
    @override
    def lc_attributes(self) -> dict[str, Any]:
        # Always record the destination, so a reload cannot pair the saved secret
        # name with a different base_url taken from the reload environment.
        return {"base_url": self._effective_base_url}

    @property
    @override
    def lc_secrets(self) -> dict[str, str]:
        # The provider follows the destination, not the request-format override.
        return {
            "api_key": "OPENROUTER_API_KEY"
            if _is_openrouter(self._effective_base_url)
            else "NACE_API_KEY"
        }

    @override
    def invoke(
        self,
        input: ClassifierRequest,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Answer one request.

        Raises:
            NaceImageModelError: Images were sent to a text-only model.
            NaceAPIError: Drex refused the request (subclasses per status).
            NaceAPIConnectionError: Drex could not be reached.

        """
        invocation = _batch_invocation.get()
        if invocation is not None and invocation.owner is self:
            return self._classify(input)
        return self._call_with_config(self._classify, input, self._traced(config), run_type="llm")

    @override
    async def ainvoke(
        self,
        input: ClassifierRequest,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Answer one request asynchronously."""
        invocation = _batch_invocation.get()
        if invocation is not None and invocation.owner is self:
            return await self._aclassify(input)
        return await self._acall_with_config(
            self._aclassify, input, self._traced(config), run_type="llm"
        )

    @override
    def batch(
        self,
        inputs: list[ClassifierRequest],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[ClassifierResponse]:
        """Answer requests with bounded concurrency, stopping dispatch after a failure.

        With `return_exceptions=True`, only HTTP 401/402/403 stop dispatch.
        Unsent inputs fail in their own callback run with `NaceBatchStoppedError`.
        Requests already sent finish, including SDK retries. A cancelled input
        fails alone; other inputs continue.
        """
        if not inputs:
            return []
        configs = get_config_list(_with_max_concurrency(config), len(inputs))
        max_concurrency = configs[0].get("max_concurrency") or DEFAULT_MAX_CONCURRENCY
        if max_concurrency < 1:
            message = "max_concurrency must be positive."
            raise ValueError(message)
        state = _BatchState(self, return_exceptions)
        results: list[Any] = [None] * len(inputs)
        next_index = 0

        def worker() -> None:
            nonlocal next_index
            token = _batch_context.set(state)
            try:
                while True:
                    with state.lock:
                        if next_index == len(inputs):
                            return
                        index = next_index
                        next_index += 1

                    def outcome(
                        request: ClassifierRequest,
                        run_manager: CallbackManagerForChainRun,
                        config: RunnableConfig,
                    ) -> ClassifierResponse:
                        try:
                            state.check()
                            # Trace the complete override outcome once, so an
                            # internally recovered error cannot stop the batch.
                            invocation_token = _batch_invocation.set(state)
                            try:
                                return self.invoke(request, config, **kwargs)
                            finally:
                                _batch_invocation.reset(invocation_token)
                        except CancelledError:
                            raise
                        except Exception as error:
                            state.record(error)
                            raise

                    try:
                        results[index] = self._call_with_config(
                            outcome, inputs[index], self._traced(configs[index]), run_type="llm"
                        )
                    except Exception as error:
                        if not isinstance(error, CancelledError):
                            state.record(error)
                        results[index] = error
            finally:
                _batch_context.reset(token)

        with get_executor_for_config(configs[0]) as executor:
            workers = [executor.submit(worker) for _ in range(min(max_concurrency, len(inputs)))]
            for future in workers:
                future.result()
        if not return_exceptions:
            if state.failure is not None:
                raise state.failure
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        return cast("list[ClassifierResponse]", results)

    @override
    async def abatch(
        self,
        inputs: list[ClassifierRequest],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[ClassifierResponse]:
        """Answer asynchronously with the same stop policy as `batch`.

        Every dispatched input goes through `ainvoke`, including subclass overrides.
        Only its final, unrecovered outcome stops dispatch, before error callbacks.
        """
        if not inputs:
            return []
        configs = get_config_list(_with_max_concurrency(config), len(inputs))
        max_concurrency = configs[0].get("max_concurrency") or DEFAULT_MAX_CONCURRENCY
        if max_concurrency < 1:
            message = "max_concurrency must be positive."
            raise ValueError(message)
        state = _BatchState(self, return_exceptions)
        results: list[Any] = [None] * len(inputs)
        next_index = 0

        async def worker() -> None:
            nonlocal next_index
            token = _batch_context.set(state)
            try:
                while next_index < len(inputs):
                    index = next_index
                    next_index += 1

                    async def outcome(
                        request: ClassifierRequest,
                        run_manager: AsyncCallbackManagerForChainRun,
                        config: RunnableConfig,
                    ) -> ClassifierResponse:
                        try:
                            state.check()
                            invocation_token = _batch_invocation.set(state)
                            try:
                                return await self.ainvoke(request, config, **kwargs)
                            finally:
                                _batch_invocation.reset(invocation_token)
                        except CancelledError:
                            raise
                        except Exception as error:
                            state.record(error)
                            raise

                    try:
                        results[index] = await asyncio.create_task(
                            self._acall_with_config(
                                outcome, inputs[index], self._traced(configs[index]), run_type="llm"
                            )
                        )
                    except asyncio.CancelledError as error:
                        task = asyncio.current_task()
                        if task is not None and task.cancelling():
                            raise
                        results[index] = error
                    except Exception as error:
                        if not isinstance(error, CancelledError):
                            state.record(error)
                        results[index] = error
            finally:
                _batch_context.reset(token)

        await asyncio.gather(*(worker() for _ in range(min(max_concurrency, len(inputs)))))
        if not return_exceptions:
            if state.failure is not None:
                raise state.failure
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        return cast("list[ClassifierResponse]", results)

    def _prepare(self, request: ClassifierRequest) -> dict[str, Any]:
        model = self.model
        text_only = is_text_only(model)
        explicit = [to_drex_image(image) for image in request.get("images") or ()]
        if explicit and text_only:
            message = (
                f"{model} reads text only, so it can't take images. "
                "Use an image model such as model='drex-max'."
            )
            raise NaceImageModelError(message)
        state, lifted = encode_state(
            request["state"],
            images="error" if text_only else "lift",
            first_index=len(explicit) + 1,
        )
        questions = {name: to_wire(q) for name, q in request["questions"].items()}
        if not questions:
            message = "At least one question is required."
            raise ValueError(message)
        images = explicit + lifted
        if images and self._openrouter:
            state, images = _openrouter_state(state, images), []
        return {
            "state": state,
            "questions": questions,
            "model": model,
            "images": images or None,
            "extra_body": self.model_kwargs or None,
        }

    def _error_secrets(self) -> tuple[str, ...]:
        values = [
            *self._redaction_secrets,
            *(self.default_headers or {}).values(),
            *(_request_error_headers.get() or []),
        ]
        for client in (self.http_client, self.http_async_client):
            if client is not None:
                values.extend(client.headers.values())
        return tuple(values + [value.split(" ", 1)[-1] for value in values if " " in value])

    def _classify(self, request: ClassifierRequest) -> ClassifierResponse:
        batch = _batch_context.get()
        current = batch if batch is not None and batch.owner is self else None
        if current is not None:
            current.check()
        call = self._prepare(request)
        dispatch = _Dispatch(current) if current is not None else None
        if dispatch is not None:
            dispatch.batch.lock.acquire()
        token = _dispatch_context.set(dispatch)
        header_token = _request_error_headers.set([])
        try:
            if current is not None:
                current.check()
            result = self._client.system_one(**call)
        except nace_sdk.NaceError as error:
            raise convert_error(error, self._provider, secrets=self._error_secrets()) from None
        finally:
            if dispatch is not None:
                dispatch.release()
            _dispatch_context.reset(token)
            _request_error_headers.reset(header_token)
        return self._record(_response(result, call["questions"]))

    async def _aclassify(self, request: ClassifierRequest) -> ClassifierResponse:
        batch = _batch_context.get()
        current = batch if batch is not None and batch.owner is self else None
        if current is not None:
            current.check()
        call = self._prepare(request)
        if current is not None:
            current.check()
        header_token = _request_error_headers.set([])
        try:
            result = await self._aclient.system_one(**call)
        except nace_sdk.NaceError as error:
            raise convert_error(error, self._provider, secrets=self._error_secrets()) from None
        finally:
            _request_error_headers.reset(header_token)
        return self._record(_response(result, call["questions"]))

    def _traced(self, config: RunnableConfig | None) -> RunnableConfig:
        config = ensure_config(config)
        config["metadata"] = {
            **(config.get("metadata") or {}),
            "ls_provider": "nace",
            "ls_model_name": self.model,
            "ls_model_type": "chat",
        }
        return config

    def _record(self, response: ClassifierResponse) -> ClassifierResponse:
        """Attach usage and the request id to the current trace, if any."""
        input_tokens = response.usage.input_tokens or 0
        output_tokens = response.usage.output_tokens or 0
        try:
            run = get_current_run_tree()
            if run is not None:
                metadata = run.extra.setdefault("metadata", {})
                metadata["usage_metadata"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
                metadata["nace_request_id"] = response.request_id
        except Exception:
            logger.debug("Could not attach Drex usage to the trace.", exc_info=True)
        return response


def _response(result: nace_sdk.SystemOneResponse, asked: dict[str, Any]) -> ClassifierResponse:
    raw = result.raw_http_response
    try:
        body = raw.json()
    except ValueError as error:
        message = "Drex returned a body that isn't JSON."
        raise NaceError(message) from error
    return ClassifierResponse.from_body(body, raw.headers.get("x-request-id"), asked)


def _openrouter_state(state: Any, images: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenRouter takes images as `image_url` parts of a `state` content array."""
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": state if isinstance(state, str) else json.dumps(state)}
    ]
    for image in images:
        if not isinstance(image, str):
            message = "OpenRouter accepts inline images only, not uploaded workspace files."
            raise NaceImageError(message)
        parts.append({"type": "image_url", "image_url": {"url": as_data_url(image)}})
    return parts


__all__ = ["DEFAULT_MODEL", "NaceClassifier", "is_text_only"]
