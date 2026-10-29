"""`NaceClassifier`: Drex decisions as a LangChain Runnable."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx
import nace_sdk
from langchain_core._api import beta
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.utils import from_env, secret_from_env
from langsmith.run_helpers import get_current_run_tree
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr, field_validator
from typing_extensions import override

from langchain_nace._images import as_data_url, to_drex_image
from langchain_nace._state import encode_state
from langchain_nace._version import __version__
from langchain_nace.errors import NaceError, NaceImageError, NaceImageModelError, convert_error
from langchain_nace.types import ClassifierRequest, ClassifierResponse, to_wire

logger = logging.getLogger(__name__)

# The default model is Drex Max, moved to the newest flagship with each Drex release.
# Users pin a model with `model=` or NACE_DEFAULT_DECISION_MODEL; nothing else in the
# package depends on this value.
DEFAULT_MODEL = "drex-max"
CLIENT_NAME = f"langchain-nace/{__version__}"
MIN_TIMEOUT = 60.0
DEFAULT_MAX_CONCURRENCY = 8

# Facts about released, pinned versions never change, so this list can't go stale.
# Every other model id (new releases, aliases such as drex-latest) is sent as given
# and Drex decides.
TEXT_ONLY_MODELS = frozenset({"drex-v1.5", "drex-v2.0"})


def is_text_only(model: str) -> bool:
    """Whether `model` is a released version known to read text only."""
    return model.rsplit("/", 1)[-1].lstrip("~") in TEXT_ONLY_MODELS


def _is_openrouter(base_url: str | None) -> bool:
    return bool(base_url) and (urlsplit(str(base_url)).hostname or "").endswith("openrouter.ai")


def _set_user_agent(request: httpx.Request) -> None:
    request.headers["User-Agent"] = CLIENT_NAME


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
        api_key: `nace_sk_...` key. Defaults to `NACE_API_KEY`.
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
        default_factory=secret_from_env("NACE_API_KEY", default=None),
        exclude=True,
        repr=False,
    )
    base_url: str | None = Field(
        default_factory=from_env("NACE_BASE_URL", default=None), validate_default=True
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

    @field_validator("base_url")
    @classmethod
    def _blank_base_url_is_default(cls, base_url: str | None) -> str | None:
        # An empty NACE_BASE_URL (common in CI and .env files) means "use the default".
        return base_url.strip() or None if base_url is not None else None

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
        headers = dict(self.default_headers or {})
        if self._openrouter:
            headers.setdefault("HTTP-Referer", "https://nace.ai")
            headers.setdefault("X-OpenRouter-Title", "Drex for LangChain")
        # The SDK's total retry budget defaults to 30 s, which skips the retry after a
        # slow failed attempt (Drex may take 55 s). `timeout` already bounds each
        # attempt and `max_retries` their number, so the budget is disabled, as in
        # deepset's typesafe-haystack and nace-haystack.
        retry = nace_sdk.RetryPolicy(max_retries=self.max_retries, timeout=None)
        key = self.api_key.get_secret_value() if self.api_key else None
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
        return self.openrouter if self.openrouter is not None else _is_openrouter(self.base_url)

    @property
    def _provider(self) -> str:
        return "OpenRouter" if self._openrouter else "Drex"

    @property
    def _base_url(self) -> str | None:
        if self.base_url and self._openrouter:
            # OpenRouter documents `.../api/v1`; the SDK appends `/v1/systemone` itself.
            return self.base_url.rstrip("/").removesuffix("/v1")
        return self.base_url

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
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "NACE_API_KEY"}

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
        return self._call_with_config(self._classify, input, self._traced(config), run_type="llm")

    @override
    async def ainvoke(
        self,
        input: ClassifierRequest,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ClassifierResponse:
        """Answer one request asynchronously."""
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
        """Answer many requests; at most 8 at a time unless `max_concurrency` is set."""
        return super().batch(
            inputs,
            _with_max_concurrency(config),
            return_exceptions=return_exceptions,
            **kwargs,
        )

    @override
    async def abatch(
        self,
        inputs: list[ClassifierRequest],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[ClassifierResponse]:
        """Answer many requests asynchronously; at most 8 at a time by default."""
        return await super().abatch(
            inputs,
            _with_max_concurrency(config),
            return_exceptions=return_exceptions,
            **kwargs,
        )

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

    def _classify(self, request: ClassifierRequest) -> ClassifierResponse:
        call = self._prepare(request)
        try:
            result = self._client.system_one(**call)
        except nace_sdk.NaceError as error:
            raise convert_error(error, self._provider) from error
        return self._record(_response(result))

    async def _aclassify(self, request: ClassifierRequest) -> ClassifierResponse:
        call = self._prepare(request)
        try:
            result = await self._aclient.system_one(**call)
        except nace_sdk.NaceError as error:
            raise convert_error(error, self._provider) from error
        return self._record(_response(result))

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


def _response(result: nace_sdk.SystemOneResponse) -> ClassifierResponse:
    raw = result.raw_http_response
    try:
        body = raw.json()
    except ValueError as error:
        message = "Drex returned a body that isn't JSON."
        raise NaceError(message) from error
    return ClassifierResponse.from_body(body, raw.headers.get("x-request-id"))


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
