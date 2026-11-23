"""Choose a chat model with Drex Router (`POST /v1/route`).

`NaceModelSelector` has no dependency on the LangChain agent framework. The middleware
in `langchain_nace.experimental.middleware` calls the same route exchange.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import math
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

import httpx
import nace_sdk
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langsmith.run_helpers import get_current_run_tree
from nace_sdk.constants import API_KEY_ENV, BASE_URL_ENV, read_env
from pydantic import SecretStr
from typing_extensions import override

from langchain_nace._ndi import _SDK_LOG_REDACTOR, SDK_LOGGER_NAME
from langchain_nace._state import message_text
from langchain_nace.classifier import (
    CLIENT_NAME,
    DEFAULT_BASE_URL,
    NaceClassifier,
    _aset_user_agent,
    _is_openrouter,
    _set_user_agent,
)

logger = logging.getLogger(__name__)

Priority = Literal["quality", "balanced", "cheapest"]
ContextMode = Literal["conversation", "latest_human"]
SelectOn = Literal["run", "model_call"]
Reason = Literal[
    "selected",
    "low_probability",
    "single_candidate",
    "no_input",
    "no_eligible_candidate",
    "timeout",
    "connection_error",
    "rate_limited",
    "unavailable",
    "auth_error",
    "no_funds",
    "invalid_request",
    "invalid_response",
]
_OutcomeKind = Literal[
    "response",
    "no_input",
    "no_eligible_candidate",
    "single_candidate",
    "timeout",
    "connection_error",
    "http_error",
    "invalid_response",
]

_CHAR_BUDGET = 24_000
_MAX_ROUTE_MESSAGES = 1024
_MAX_IMAGE_DEPTH = 8
_MAX_CANDIDATES = 255
_MAX_ID_LENGTH = 256
_MAX_PROFILE_TEXT = 128
_MAX_TAGS = 64
_MAX_CANDIDATE_TEXT = 100_000
_MAX_CONTEXT_WINDOW = 2**53 - 1
_DEFAULT_TIMEOUT = 3.0
_TOKEN_DIVISOR = 3
_IMAGE_BLOCK_TYPES = frozenset({"image", "image_url", "input_image"})
_FILE_BLOCK_TYPES = frozenset({"media", "file"})
_PROFILE_FIELDS = frozenset({"provider", "price_in", "price_out", "context", "tags"})
_PRIORITIES = frozenset({"quality", "balanced", "cheapest"})
_WARN_REASONS = frozenset(
    {
        "low_probability",
        "no_eligible_candidate",
        "timeout",
        "connection_error",
        "rate_limited",
        "unavailable",
        "auth_error",
        "no_funds",
        "invalid_request",
        "invalid_response",
    }
)
_REQUEST_ID = re.compile(r"^req_[A-Za-z0-9]{8,64}$")
# JavaScript `\s`, including U+FEFF, which Python's `str.strip` leaves in place.
_JS_WHITESPACE = frozenset(
    "\t\n\v\f\r \u00a0\u1680\u2028\u2029\u202f\u205f\u3000\ufeff"
    + "".join(chr(code) for code in range(0x2000, 0x200B))
)
_NO_RETRY = nace_sdk.RetryPolicy(max_retries=0)
_TRACE_CONFIG: RunnableConfig = {"run_name": "NaceModelSelector"}


class RouteTurn(TypedDict):
    """One text turn sent to `POST /v1/route`."""

    role: Literal["system", "user", "assistant"]
    content: str


class CandidateProfile(TypedDict):
    """Metadata for a candidate that is not a known Drex Router id.

    Prices are USD per million tokens. `id` may be omitted; the mapping key supplies it.
    """

    id: NotRequired[str]
    provider: str
    price_in: float
    price_out: float
    context: int
    tags: list[str]


class ModelSelection(TypedDict):
    """The selection kept in agent state under `nace_model_selection`.

    A plain mapping, so checkpointers can store it. It holds no message text.
    """

    model: str | None
    """The candidate id in use, or `None` when the agent's own model is in use."""

    selected: str | None
    """The id Drex returned, or `None`."""

    probabilities: dict[str, float]
    """The full map from Drex, or `{}`."""

    fallback: bool
    """Whether `model` is not the Drex selection."""

    reason: Reason
    """Why `model` is in use. See the reason table in the package docs."""

    priority: Priority
    """The `selection_priority` used for the decision."""

    candidates: list[str]
    """The candidate ids sent to Drex, in binding order."""

    request_id: str | None
    """Drex's request id, when the response included a safe one."""

    latency_ms: int | None
    """How long the route call took, or `None` when no call was made."""


@dataclass(frozen=True)
class CandidateBinding:
    """One candidate the caller permits, in binding order."""

    id: str
    profile: CandidateProfile | None = None
    images: bool = True
    tools: bool = True


@dataclass(frozen=True)
class ExchangeOutcome:
    """Everything known after a route attempt, with no message text.

    `selection_from_outcome` is the only function that turns one of these into a
    `ModelSelection`.
    """

    kind: _OutcomeKind
    priority: Priority
    sent_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    fallback_id: str | None = None
    min_probability: float | None = None
    selected: Any = field(default=None, repr=False)
    probabilities: Any = field(default=None, repr=False)
    status: int | None = None
    request_id: str | None = None
    latency_ms: int | None = None
    single_id: str | None = None


class _RouteWireRedactor(logging.Filter):
    """Drop route request and response bodies from the SDK debug wire log."""

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, dict):
            return True
        url = args.get("url")
        if not isinstance(url, str) or "/v1/route" not in url:
            return True
        secrets = tuple(secret for secret in _SDK_LOG_REDACTOR._secrets if secret)
        record.args = _scrub_log_args({**args, "body": b"[redacted]"}, secrets)
        return True


_ROUTE_WIRE_REDACTOR = _RouteWireRedactor()


def require_priority(priority: str) -> Priority:
    """Return `priority` when it is `quality`, `balanced`, or `cheapest`.

    Raises:
        ValueError: `priority` is not one of those three.

    """
    if priority not in _PRIORITIES:
        message = "priority must be 'quality', 'balanced', or 'cheapest'."
        raise ValueError(message)
    return cast("Priority", priority)


def require_timeout(timeout: float) -> float:
    """Return `timeout` when it is a positive finite number of seconds.

    Raises:
        ValueError: `timeout` is not a positive finite number.

    """
    if isinstance(timeout, bool) or not isinstance(timeout, int | float):
        message = "timeout must be a positive number of seconds."
        raise ValueError(message)
    if not math.isfinite(timeout) or timeout <= 0:
        message = "timeout must be a positive number of seconds."
        raise ValueError(message)
    return float(timeout)


def require_min_probability(min_probability: float | None) -> float | None:
    """Return `min_probability` when it is omitted or between 0 and 1.

    Raises:
        ValueError: The value is not a finite number from 0 to 1.

    """
    if min_probability is None:
        return None
    if isinstance(min_probability, bool) or not isinstance(min_probability, int | float):
        message = "min_probability must be between 0 and 1."
        raise ValueError(message)
    if not math.isfinite(min_probability) or not 0 <= min_probability <= 1:
        message = "min_probability must be between 0 and 1."
        raise ValueError(message)
    return float(min_probability)


def require_context(context: str) -> ContextMode:
    """Return `context` when it is `conversation` or `latest_human`.

    Raises:
        ValueError: `context` is neither.

    """
    if context not in {"conversation", "latest_human"}:
        message = "context must be 'conversation' or 'latest_human'."
        raise ValueError(message)
    return cast("ContextMode", context)


def require_select_on(select_on: str) -> SelectOn:
    """Return `select_on` when it is `run` or `model_call`.

    Raises:
        ValueError: `select_on` is neither.

    """
    if select_on not in {"run", "model_call"}:
        message = "select_on must be 'run' or 'model_call'."
        raise ValueError(message)
    return cast("SelectOn", select_on)


def validate_profile(candidate_id: str, profile: Mapping[str, Any]) -> CandidateProfile:
    """Return a profile whose `id` is `candidate_id`.

    Raises:
        ValueError: The profile is missing a field, has an extra field, or a field
            has the wrong type.

    """
    keys = set(profile)
    if "id" in keys and profile.get("id") != candidate_id:
        message = f"Candidate {candidate_id!r} profile id does not match the candidate id."
        raise ValueError(message)
    keys.discard("id")
    if keys != _PROFILE_FIELDS:
        message = f"Candidate {candidate_id!r} has an invalid profile."
        raise ValueError(message)
    provider = profile["provider"]
    if not isinstance(provider, str) or _is_js_blank(provider) or len(provider) > _MAX_PROFILE_TEXT:
        message = f"Candidate {candidate_id!r} has an invalid profile."
        raise ValueError(message)
    return {
        "id": candidate_id,
        "provider": provider,
        "price_in": _price(candidate_id, profile["price_in"]),
        "price_out": _price(candidate_id, profile["price_out"]),
        "context": _context_window(candidate_id, profile["context"]),
        "tags": _tags(candidate_id, profile["tags"]),
    }


def validate_candidate_map(
    candidates: Mapping[str, Mapping[str, Any] | None],
    *,
    fallback: str | None,
) -> tuple[CandidateBinding, ...]:
    """Check candidate ids and profiles and return bindings in mapping order.

    Raises:
        ValueError: There are no candidates, more than 255, a blank or over-long id,
            an id that is not a string, a `fallback` that is not a candidate, or an
            invalid profile.

    """
    if not candidates:
        message = "candidates must contain at least one model."
        raise ValueError(message)
    if len(candidates) > _MAX_CANDIDATES:
        message = "candidates must contain at most 255 models."
        raise ValueError(message)
    if fallback is not None and fallback not in candidates:
        message = f"fallback={fallback!r} is not one of {list(candidates)}."
        raise ValueError(message)
    bindings: list[CandidateBinding] = []
    for candidate_id, profile in candidates.items():
        _check_id(candidate_id)
        parsed = None if profile is None else validate_profile(candidate_id, profile)
        bindings.append(CandidateBinding(id=candidate_id, profile=parsed))
    if _candidate_text(bindings) > _MAX_CANDIDATE_TEXT:
        message = "candidates must contain at most 100000 characters of ids, providers, and tags."
        raise ValueError(message)
    return tuple(bindings)


def request_characters(
    messages: Sequence[BaseMessage], system_message: BaseMessage | None = None
) -> int:
    """Count characters in the model request, for the context-window estimate."""
    total = _content_characters(system_message) if system_message is not None else 0
    return total + sum(_content_characters(message) for message in messages)


def has_image_block(
    messages: Sequence[BaseMessage], system_message: BaseMessage | None = None
) -> bool:
    """Whether the model request contains an image content block."""
    pending: list[BaseMessage] = []
    if system_message is not None:
        pending.append(system_message)
    pending.extend(messages)
    return any(_contains_image(message.content, 0) for message in pending)


def project_route_messages(
    messages: Sequence[BaseMessage],
    *,
    system_message: BaseMessage | None = None,
    context: ContextMode = "conversation",
) -> list[RouteTurn]:
    """Build the text turns `POST /v1/route` accepts.

    Human messages go through `without_media(..., keep_images=False)`. Assistant
    tool calls and tool results are dropped. A human turn with no real text,
    including one that is only a media placeholder, is not sent. When the latest
    human message has no real text, the result is empty in both context modes
    and an earlier human turn is not sent. The latest human message is kept.
    The rest of the 24,000 character and 1,024 message budget is filled with
    the newest other turns first. The turns stay in chronological order.
    `latest_human` sends the system text and that latest human message.
    """
    latest_human = next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage)),
        None,
    )
    if latest_human is None or not message_text(latest_human):
        return []
    turns: list[RouteTurn] = []
    if system_message is not None:
        _append_turn(turns, "system", message_text(system_message))
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, SystemMessage):
            _append_turn(turns, "system", message_text(message))
        elif isinstance(message, HumanMessage):
            if not message_text(message):
                continue
            plain = _without_media(message, keep_images=False)
            _append_turn(turns, "user", message_text(plain))
        elif isinstance(message, AIMessage):
            _append_turn(turns, "assistant", message_text(message))
    if context == "latest_human":
        latest = next(
            (message for message in reversed(messages) if isinstance(message, HumanMessage)),
            None,
        )
        if latest is None:
            return []
        text = message_text(_without_media(latest, keep_images=False))
        if not text:
            return []
        turns = [turn for turn in turns if turn["role"] == "system"]
        turns.append({"role": "user", "content": text})
    return _trim_turns(turns)


def filter_candidates(
    bindings: Sequence[CandidateBinding],
    *,
    has_image: bool,
    has_tools: bool,
    estimated_tokens: float,
) -> list[CandidateBinding]:
    """Return the bindings that may receive this request, in binding order."""
    eligible: list[CandidateBinding] = []
    for binding in bindings:
        if has_image and not binding.images:
            continue
        if has_tools and not binding.tools:
            continue
        profile = binding.profile
        if profile is not None and profile["context"] < estimated_tokens:
            continue
        eligible.append(binding)
    return eligible


def response_is_valid(
    selected: object, probabilities: object, sent_ids: Sequence[str]
) -> dict[str, float] | None:
    """Return the probability map when it matches `sent_ids`, else `None`.

    `selected` must be one of `sent_ids`. The probability keys must be exactly
    those ids. Each probability must be a finite number from 0 to 1.
    """
    if not isinstance(selected, str) or selected not in sent_ids:
        return None
    if not isinstance(probabilities, Mapping):
        return None
    if set(probabilities) != set(sent_ids):
        return None
    parsed: dict[str, float] = {}
    for key, value in probabilities.items():
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int | float)
        ):
            return None
        if not math.isfinite(value) or value < 0 or value > 1:
            return None
        parsed[key] = float(value)
    return parsed


def local_outcome(
    messages: Sequence[RouteTurn],
    eligible: Sequence[CandidateBinding],
    *,
    priority: Priority,
    fallback_id: str | None,
    min_probability: float | None,
    candidate_ids: Sequence[str],
) -> ExchangeOutcome | None:
    """Return an outcome that needs no route call, or `None` when Drex should be asked."""
    shared = (
        priority,
        tuple(candidate_ids),
        fallback_id,
        min_probability,
    )
    if not any(turn["role"] == "user" and turn["content"] for turn in messages):
        return _local("no_input", *shared)
    if not eligible:
        return _local("no_eligible_candidate", *shared)
    if len(eligible) == 1:
        outcome = _local("single_candidate", *shared)
        return ExchangeOutcome(
            kind=outcome.kind,
            priority=outcome.priority,
            candidate_ids=outcome.candidate_ids,
            fallback_id=outcome.fallback_id,
            min_probability=outcome.min_probability,
            single_id=eligible[0].id,
        )
    return None


def selection_from_outcome(outcome: ExchangeOutcome) -> ModelSelection:
    """Map one exchange outcome to a `ModelSelection`.

    This is the only function that builds a `ModelSelection`.
    """
    reason, model, selected, probabilities = _fields(outcome)
    request_id = _safe_request_id(outcome.request_id)
    latency = outcome.latency_ms
    if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
        latency = None
    fallback = selected is None or model != selected
    selection: ModelSelection = {
        "model": model,
        "selected": selected,
        "probabilities": probabilities,
        "fallback": fallback,
        "reason": reason,
        "priority": outcome.priority,
        "candidates": list(outcome.sent_ids),
        "request_id": request_id,
        "latency_ms": latency,
    }
    return selection


def apply_outcome(outcome: ExchangeOutcome) -> ModelSelection:
    """Map `outcome` and log one warning when the result is a fallback."""
    selection = selection_from_outcome(outcome)
    if selection["reason"] in _WARN_REASONS:
        status = "-" if outcome.status is None else str(outcome.status)
        request_id = selection["request_id"] if selection["request_id"] is not None else "-"
        ids = outcome.candidate_ids or outcome.sent_ids
        logger.warning(
            "Drex model selection fell back: reason=%s status=%s request_id=%s candidates=%s",
            selection["reason"],
            status,
            request_id,
            ",".join(ids),
        )
    return selection


def _raise_if_interrupted(error: BaseException) -> None:
    """Raise a `KeyboardInterrupt` or `CancelledError` buried under a cleanup error."""
    interruption = _buried_interruption(error)
    if interruption is not None:
        raise interruption


def _buried_interruption(error: BaseException) -> BaseException | None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(current, KeyboardInterrupt | asyncio.CancelledError):
            return current
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return None


def _raise_if_caller_cancelled(error: BaseException) -> None:
    """Raise when the caller cancelled and the deadline did not consume that cancel.

    `asyncio.timeout` uncancels the cancel it added before it raises `TimeoutError`.
    A remaining cancellation count belongs to the caller.
    """
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError from error


def route_exchange(
    client: nace_sdk.NaceClient,
    messages: Sequence[RouteTurn],
    candidates: Sequence[str | CandidateProfile],
    *,
    priority: Priority,
    timeout: float,
    sent_ids: Sequence[str],
    candidate_ids: Sequence[str],
    fallback_id: str | None,
    min_probability: float | None,
) -> ModelSelection:
    """Send one route request, wait, apply the deadline, and close.

    The sync deadline is the SDK timeout, which bounds each HTTP operation, plus
    the elapsed time after a response returns. A late response is `timeout`. One
    attempt is made (`max_retries=0`). `NaceError` and `httpx.HTTPError` use the
    status table. Any other `Exception` is `connection_error`. Caller cancellation
    does not apply on this path. A response is closed before this function returns.
    """
    started = time.perf_counter()
    response: nace_sdk.RouteResponse | None = None
    try:
        response = client.route(
            cast("Sequence[nace_sdk.RouteMessage]", list(messages)),
            cast("Sequence[nace_sdk.RouteCandidate]", list(candidates)),
            selection_priority=priority,
            retry=_NO_RETRY,
            timeout=timeout,
        )
        if time.perf_counter() - started >= timeout:
            _close_route_response(response)
            response = None
            return apply_outcome(
                _timeout_outcome(
                    priority=priority,
                    sent_ids=sent_ids,
                    candidate_ids=candidate_ids,
                    fallback_id=fallback_id,
                    min_probability=min_probability,
                    latency_ms=_latency_ms(started),
                )
            )
        return apply_outcome(
            _from_response(
                response,
                priority=priority,
                sent_ids=sent_ids,
                candidate_ids=candidate_ids,
                fallback_id=fallback_id,
                min_probability=min_probability,
                latency_ms=_latency_ms(started),
            )
        )
    except (nace_sdk.NaceError, httpx.HTTPError) as error:
        _raise_if_interrupted(error)
        return apply_outcome(
            _outcome_for_failure(
                error,
                started=started,
                timeout=timeout,
                priority=priority,
                sent_ids=sent_ids,
                candidate_ids=candidate_ids,
                fallback_id=fallback_id,
                min_probability=min_probability,
            )
        )
    except Exception as error:
        _raise_if_interrupted(error)
        return apply_outcome(
            _outcome_for_failure(
                error,
                started=started,
                timeout=timeout,
                priority=priority,
                sent_ids=sent_ids,
                candidate_ids=candidate_ids,
                fallback_id=fallback_id,
                min_probability=min_probability,
            )
        )
    finally:
        _close_route_response(response)


async def aroute_exchange(
    client: nace_sdk.AsyncNaceClient,
    messages: Sequence[RouteTurn],
    candidates: Sequence[str | CandidateProfile],
    *,
    priority: Priority,
    timeout: float,
    sent_ids: Sequence[str],
    candidate_ids: Sequence[str],
    fallback_id: str | None,
    min_probability: float | None,
) -> ModelSelection:
    """Async twin of `route_exchange`.

    The wait is bounded by `asyncio.timeout`. A caller cancellation propagates,
    including when it is still pending after a successful return, and no
    selection is returned. `TimeoutError` is `timeout` only when that deadline
    expired. Any other `Exception` is `connection_error`.
    """
    started = time.perf_counter()
    response: nace_sdk.RouteResponse | None = None
    deadline = asyncio.timeout(timeout)
    try:
        async with deadline:
            response = await client.route(
                cast("Sequence[nace_sdk.RouteMessage]", list(messages)),
                cast("Sequence[nace_sdk.RouteCandidate]", list(candidates)),
                selection_priority=priority,
                retry=_NO_RETRY,
                timeout=timeout,
            )
        if deadline.expired() or time.perf_counter() - started >= timeout:
            _close_route_response(response)
            response = None
            return apply_outcome(
                _timeout_outcome(
                    priority=priority,
                    sent_ids=sent_ids,
                    candidate_ids=candidate_ids,
                    fallback_id=fallback_id,
                    min_probability=min_probability,
                    latency_ms=_latency_ms(started),
                )
            )
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            _close_route_response(response)
            response = None
            raise asyncio.CancelledError
        return apply_outcome(
            _from_response(
                response,
                priority=priority,
                sent_ids=sent_ids,
                candidate_ids=candidate_ids,
                fallback_id=fallback_id,
                min_probability=min_probability,
                latency_ms=_latency_ms(started),
            )
        )
    except TimeoutError as caught:
        # A caller cancel in the same loop step as the deadline stays a cancellation.
        # The SDK's own timeout is a TimeoutError too; a hook's TimeoutError is not.
        _raise_if_caller_cancelled(caught)
        if isinstance(caught, nace_sdk.NaceError):
            _raise_if_interrupted(caught)
            return apply_outcome(
                _outcome_for_failure(
                    caught,
                    started=started,
                    timeout=timeout,
                    priority=priority,
                    sent_ids=sent_ids,
                    candidate_ids=candidate_ids,
                    fallback_id=fallback_id,
                    min_probability=min_probability,
                )
            )
        if deadline.expired():
            return apply_outcome(
                _timeout_outcome(
                    priority=priority,
                    sent_ids=sent_ids,
                    candidate_ids=candidate_ids,
                    fallback_id=fallback_id,
                    min_probability=min_probability,
                    latency_ms=_latency_ms(started),
                )
            )
        return apply_outcome(
            _connection_outcome(
                priority=priority,
                sent_ids=sent_ids,
                candidate_ids=candidate_ids,
                fallback_id=fallback_id,
                min_probability=min_probability,
                latency_ms=_latency_ms(started),
            )
        )
    except (nace_sdk.NaceError, httpx.HTTPError) as error:
        _raise_if_caller_cancelled(error)
        _raise_if_interrupted(error)
        return apply_outcome(
            _outcome_for_failure(
                error,
                started=started,
                timeout=timeout,
                priority=priority,
                sent_ids=sent_ids,
                candidate_ids=candidate_ids,
                fallback_id=fallback_id,
                min_probability=min_probability,
            )
        )
    except Exception as error:
        _raise_if_caller_cancelled(error)
        _raise_if_interrupted(error)
        return apply_outcome(
            _outcome_for_failure(
                error,
                started=started,
                timeout=timeout,
                priority=priority,
                sent_ids=sent_ids,
                candidate_ids=candidate_ids,
                fallback_id=fallback_id,
                min_probability=min_probability,
            )
        )
    finally:
        _close_route_response(response)


class NaceModelSelector:
    """Ask Drex Router which candidate should answer a conversation.

    `select` and `aselect` return a `ModelSelection`. They do not call a chat model.
    Build one selector and pass it to `NaceModelSelectorMiddleware`, or call it directly.

    The route call is one attempt. The async path cancels the call at `timeout`
    seconds. The sync path passes that timeout to the SDK, which applies it to each
    HTTP operation. An OpenRouter base URL is refused.

    Args:
        api_key: Nace API key. Falls back to `NACE_API_KEY`.
        base_url: Nace Console URL. Falls back to `NACE_BASE_URL`, then the console.
        candidates: Candidate ids mapped to a profile, or `None` for a known id.
            Omit this when the middleware owns the candidates.
        priority: `quality`, `balanced`, or `cheapest`.
        fallback: Candidate id used when Drex does not return a usable selection.
            `None` means the caller keeps its own model.
        min_probability: Below this probability of the selected id, use `fallback`.
        timeout: Seconds for the route call. Defaults to 3.
        context: `conversation` or `latest_human`.
        http_client: Sync HTTP client. A client is built when omitted.
        http_async_client: Async HTTP client. A client is built when omitted.

    Raises:
        ValueError: The endpoint is OpenRouter or not a usable URL, the key is
            unusable, or the candidates, fallback, priority, or timeout are invalid.

    """

    def __init__(
        self,
        *,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        candidates: Mapping[str, Mapping[str, Any] | None] | None = None,
        priority: str = "balanced",
        fallback: str | None = None,
        min_probability: float | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        context: str = "conversation",
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Resolve credentials and build the SDK clients."""
        self.priority = require_priority(priority)
        self.timeout = require_timeout(timeout)
        self.min_probability = require_min_probability(min_probability)
        self.context = require_context(context)
        self.fallback = fallback
        self._bindings = (
            None if candidates is None else validate_candidate_map(candidates, fallback=fallback)
        )
        self._client, self._aclient = _open_clients(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,
            http_client=http_client,
            http_async_client=http_async_client,
        )

    def select(self, messages: Sequence[BaseMessage]) -> ModelSelection:
        """Select a candidate for `messages`."""
        planned = self._plan(messages)
        if isinstance(planned, _RoutePlan):
            return self._trace(planned)
        return planned

    async def aselect(self, messages: Sequence[BaseMessage]) -> ModelSelection:
        """Async twin of `select`."""
        planned = self._plan(messages)
        if not isinstance(planned, _RoutePlan):
            return planned
        return await self._atrace(planned)

    def exchange(
        self,
        messages: Sequence[RouteTurn],
        bindings: Sequence[CandidateBinding],
        *,
        priority: Priority,
        fallback_id: str | None,
        min_probability: float | None,
        timeout: float,
    ) -> ModelSelection:
        """Route `messages` to `bindings`, or return a local outcome when no call is needed."""
        planned = self._plan_bindings(
            messages,
            bindings,
            priority=priority,
            fallback_id=fallback_id,
            min_probability=min_probability,
            timeout=timeout,
        )
        if isinstance(planned, _RoutePlan):
            return self._trace(planned)
        return planned

    async def aexchange(
        self,
        messages: Sequence[RouteTurn],
        bindings: Sequence[CandidateBinding],
        *,
        priority: Priority,
        fallback_id: str | None,
        min_probability: float | None,
        timeout: float,
    ) -> ModelSelection:
        """Async twin of `exchange`."""
        planned = self._plan_bindings(
            messages,
            bindings,
            priority=priority,
            fallback_id=fallback_id,
            min_probability=min_probability,
            timeout=timeout,
        )
        if not isinstance(planned, _RoutePlan):
            return planned
        return await self._atrace(planned)

    def _plan(self, messages: Sequence[BaseMessage]) -> ModelSelection | _RoutePlan:
        if self._bindings is None:
            message = "candidates must contain at least one model."
            raise ValueError(message)
        projected = project_route_messages(messages, context=self.context)
        tokens = request_characters(messages) / _TOKEN_DIVISOR
        image = has_image_block(messages)
        eligible = filter_candidates(
            self._bindings, has_image=image, has_tools=False, estimated_tokens=tokens
        )
        fallback_id = _eligible_fallback(
            self.fallback,
            self._bindings,
            has_image=image,
            has_tools=False,
            estimated_tokens=tokens,
        )
        return self._plan_bindings(
            projected,
            eligible,
            priority=self.priority,
            fallback_id=fallback_id,
            min_probability=self.min_probability,
            timeout=self.timeout,
            candidate_ids=tuple(binding.id for binding in self._bindings),
        )

    def _plan_bindings(
        self,
        messages: Sequence[RouteTurn],
        bindings: Sequence[CandidateBinding],
        *,
        priority: Priority,
        fallback_id: str | None,
        min_probability: float | None,
        timeout: float,
        candidate_ids: Sequence[str] | None = None,
    ) -> ModelSelection | _RoutePlan:
        ids = (
            tuple(candidate_ids)
            if candidate_ids is not None
            else tuple(binding.id for binding in bindings)
        )
        outcome = local_outcome(
            messages,
            bindings,
            priority=priority,
            fallback_id=fallback_id,
            min_probability=min_probability,
            candidate_ids=ids,
        )
        if outcome is not None:
            return apply_outcome(outcome)
        return _RoutePlan(
            messages=tuple(messages),
            candidates=tuple(_wire_candidate(binding) for binding in bindings),
            sent_ids=tuple(binding.id for binding in bindings),
            candidate_ids=ids,
            priority=priority,
            fallback_id=fallback_id,
            min_probability=min_probability,
            timeout=timeout,
        )

    def _trace(self, plan: _RoutePlan) -> ModelSelection:
        def run(_: dict[str, Any]) -> ModelSelection:
            selection = route_exchange(
                self._client,
                plan.messages,
                plan.candidates,
                priority=plan.priority,
                timeout=plan.timeout,
                sent_ids=plan.sent_ids,
                candidate_ids=plan.candidate_ids,
                fallback_id=plan.fallback_id,
                min_probability=plan.min_probability,
            )
            _attach_request_id(selection["request_id"])
            return selection

        return RunnableLambda(run).invoke(_trace_inputs(plan), config=_TRACE_CONFIG)

    async def _atrace(self, plan: _RoutePlan) -> ModelSelection:
        async def run(_: dict[str, Any]) -> ModelSelection:
            selection = await aroute_exchange(
                self._aclient,
                plan.messages,
                plan.candidates,
                priority=plan.priority,
                timeout=plan.timeout,
                sent_ids=plan.sent_ids,
                candidate_ids=plan.candidate_ids,
                fallback_id=plan.fallback_id,
                min_probability=plan.min_probability,
            )
            _attach_request_id(selection["request_id"])
            return selection

        result = await RunnableLambda(run).ainvoke(_trace_inputs(plan), config=_TRACE_CONFIG)
        return cast("ModelSelection", result)


@dataclass(frozen=True)
class _RoutePlan:
    """A route call that still has to be sent."""

    messages: tuple[RouteTurn, ...] = field(repr=False)
    candidates: tuple[str | CandidateProfile, ...]
    sent_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    priority: Priority
    fallback_id: str | None
    min_probability: float | None
    timeout: float


def _trace_inputs(plan: _RoutePlan) -> dict[str, Any]:
    return {
        "candidates": list(plan.sent_ids),
        "priority": plan.priority,
        "message_count": len(plan.messages),
        "character_count": sum(len(turn["content"]) for turn in plan.messages),
    }


def _attach_request_id(request_id: str | None) -> None:
    if not request_id:
        return
    try:
        run = get_current_run_tree()
        if run is not None:
            metadata = run.extra.setdefault("metadata", {})
            metadata["nace_request_id"] = request_id
    except Exception:
        logger.debug("Could not attach the Drex request id to the trace.")


def _wire_candidate(binding: CandidateBinding) -> str | CandidateProfile:
    return binding.id if binding.profile is None else binding.profile


def _eligible_fallback(
    fallback: str | None,
    bindings: Sequence[CandidateBinding],
    *,
    has_image: bool,
    has_tools: bool,
    estimated_tokens: float,
) -> str | None:
    if fallback is None:
        return None
    binding = next(item for item in bindings if item.id == fallback)
    eligible = filter_candidates(
        (binding,),
        has_image=has_image,
        has_tools=has_tools,
        estimated_tokens=estimated_tokens,
    )
    return fallback if eligible else None


def _local(
    kind: Literal["no_input", "no_eligible_candidate", "single_candidate"],
    priority: Priority,
    candidate_ids: tuple[str, ...],
    fallback_id: str | None,
    min_probability: float | None,
) -> ExchangeOutcome:
    return ExchangeOutcome(
        kind=kind,
        priority=priority,
        candidate_ids=candidate_ids,
        fallback_id=fallback_id,
        min_probability=min_probability,
    )


def _fields(
    outcome: ExchangeOutcome,
) -> tuple[Reason, str | None, str | None, dict[str, float]]:
    if outcome.kind == "no_input":
        return "no_input", None, None, {}
    if outcome.kind == "no_eligible_candidate":
        return "no_eligible_candidate", outcome.fallback_id, None, {}
    if outcome.kind == "single_candidate":
        return "single_candidate", outcome.single_id, None, {}
    if outcome.kind == "timeout":
        return "timeout", outcome.fallback_id, None, {}
    if outcome.kind == "connection_error":
        return "connection_error", outcome.fallback_id, None, {}
    if outcome.kind == "invalid_response":
        return "invalid_response", outcome.fallback_id, None, {}
    if outcome.kind == "http_error":
        return _reason_for_status(outcome.status), outcome.fallback_id, None, {}
    parsed = response_is_valid(outcome.selected, outcome.probabilities, outcome.sent_ids)
    if parsed is None or not isinstance(outcome.selected, str):
        return "invalid_response", outcome.fallback_id, None, {}
    if outcome.min_probability is not None and parsed[outcome.selected] < outcome.min_probability:
        return "low_probability", outcome.fallback_id, outcome.selected, parsed
    return "selected", outcome.selected, outcome.selected, parsed


def _reason_for_status(status: int | None) -> Reason:
    if status == 429:
        return "rate_limited"
    if status is not None and status >= 500:
        return "unavailable"
    if status in {401, 403}:
        return "auth_error"
    if status == 402:
        return "no_funds"
    return "invalid_request"


def _from_response(
    response: nace_sdk.RouteResponse,
    *,
    priority: Priority,
    sent_ids: Sequence[str],
    candidate_ids: Sequence[str],
    fallback_id: str | None,
    min_probability: float | None,
    latency_ms: int,
) -> ExchangeOutcome:
    try:
        request_id: str | None = response.request_id
    except nace_sdk.NaceError:
        request_id = None
    probabilities: Any
    try:
        probabilities = dict(response.probabilities)
    except (TypeError, ValueError):
        probabilities = response.probabilities
    return ExchangeOutcome(
        kind="response",
        priority=priority,
        sent_ids=tuple(sent_ids),
        candidate_ids=tuple(candidate_ids),
        fallback_id=fallback_id,
        min_probability=min_probability,
        selected=response.selected,
        probabilities=probabilities,
        status=200,
        request_id=request_id,
        latency_ms=latency_ms,
    )


def _from_error(
    error: nace_sdk.NaceError,
    *,
    priority: Priority,
    sent_ids: Sequence[str],
    candidate_ids: Sequence[str],
    fallback_id: str | None,
    min_probability: float | None,
    latency_ms: int,
) -> ExchangeOutcome:
    status: int | None = None
    request_id: str | None = None
    if isinstance(error, nace_sdk.NaceAPITimeoutError):
        kind: _OutcomeKind = "timeout"
    elif isinstance(error, nace_sdk.NaceAPIConnectionError):
        kind = "connection_error"
    elif isinstance(error, nace_sdk.NaceAPIResponseValidationError):
        kind = "invalid_response"
        status = error.status
        request_id = error.request_id
    elif isinstance(error, nace_sdk.NaceAPIError):
        kind = "http_error"
        status = error.status
        request_id = error.request_id
    else:
        kind = "invalid_response"
    return ExchangeOutcome(
        kind=kind,
        priority=priority,
        sent_ids=tuple(sent_ids),
        candidate_ids=tuple(candidate_ids),
        fallback_id=fallback_id,
        min_probability=min_probability,
        status=status,
        request_id=request_id,
        latency_ms=latency_ms,
    )


def _connection_outcome(
    *,
    priority: Priority,
    sent_ids: Sequence[str],
    candidate_ids: Sequence[str],
    fallback_id: str | None,
    min_probability: float | None,
    latency_ms: int,
) -> ExchangeOutcome:
    return ExchangeOutcome(
        kind="connection_error",
        priority=priority,
        sent_ids=tuple(sent_ids),
        candidate_ids=tuple(candidate_ids),
        fallback_id=fallback_id,
        min_probability=min_probability,
        latency_ms=latency_ms,
    )


def _timeout_outcome(
    *,
    priority: Priority,
    sent_ids: Sequence[str],
    candidate_ids: Sequence[str],
    fallback_id: str | None,
    min_probability: float | None,
    latency_ms: int,
) -> ExchangeOutcome:
    return ExchangeOutcome(
        kind="timeout",
        priority=priority,
        sent_ids=tuple(sent_ids),
        candidate_ids=tuple(candidate_ids),
        fallback_id=fallback_id,
        min_probability=min_probability,
        latency_ms=latency_ms,
    )


def _close_route_response(response: object) -> None:
    if response is None:
        return
    try:
        raw = response.raw_http_response  # type: ignore[attr-defined]
    except (nace_sdk.NaceError, AttributeError):
        return
    close = getattr(raw, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:
        logger.debug("Could not close the route response.")


def _latency_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _scrub_log_args(value: Any, secrets: tuple[str, ...]) -> Any:
    """Remove configured secrets from a route wire-log argument."""
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: _scrub_log_args(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_log_args(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_log_args(item, secrets) for item in value)
    return value


def _is_js_blank(value: str) -> bool:
    """True when every character is JavaScript whitespace, including U+FEFF."""
    return all(char in _JS_WHITESPACE for char in value)


def _safe_request_id(value: object) -> str | None:
    """Keep a request id of the form `req_` plus 8-64 letters or digits, with no secret."""
    if not isinstance(value, str) or _REQUEST_ID.fullmatch(value) is None:
        return None
    for secret in _SDK_LOG_REDACTOR._secrets:
        if secret and secret in value:
            return None
    return value


def _contains_image(value: object, depth: int) -> bool:
    """Whether `value` is an image block or nests one, up to `_MAX_IMAGE_DEPTH`."""
    if _is_image_block(value):
        return True
    if depth >= _MAX_IMAGE_DEPTH:
        return False
    if isinstance(value, list):
        return any(_contains_image(item, depth + 1) for item in value)
    if isinstance(value, dict):
        nested = value.get("content")
        if isinstance(nested, list):
            return any(_contains_image(item, depth + 1) for item in nested)
    return False


def _is_image_block(block: object) -> bool:
    """Whether `block` is an image the candidate filter must see."""
    if not isinstance(block, dict):
        return False
    if "image" in block:
        return True
    kind = block.get("type")
    if kind in _IMAGE_BLOCK_TYPES:
        return True
    if kind not in _FILE_BLOCK_TYPES:
        return False
    if _mime_is_image(block.get("mime_type")):
        return True
    source = block.get("source")
    if not isinstance(source, dict):
        return False
    return any(_mime_is_image(source.get(key)) for key in ("mime_type", "mime", "media_type"))


def _mime_is_image(value: object) -> bool:
    return isinstance(value, str) and value.lower().startswith("image/")


def _candidate_text(bindings: Sequence[CandidateBinding]) -> int:
    """Code points in the ids, providers, and tags the server counts."""
    total = 0
    for binding in bindings:
        total += len(binding.id)
        profile = binding.profile
        if profile is None:
            continue
        total += len(profile["provider"])
        total += sum(len(tag) for tag in profile["tags"])
    return total


def _outcome_for_failure(
    error: BaseException,
    *,
    started: float,
    timeout: float,
    priority: Priority,
    sent_ids: Sequence[str],
    candidate_ids: Sequence[str],
    fallback_id: str | None,
    min_probability: float | None,
) -> ExchangeOutcome:
    """Map a failed exchange. Past the deadline, the reason is `timeout`."""
    latency_ms = _latency_ms(started)
    if time.perf_counter() - started >= timeout:
        return _timeout_outcome(
            priority=priority,
            sent_ids=sent_ids,
            candidate_ids=candidate_ids,
            fallback_id=fallback_id,
            min_probability=min_probability,
            latency_ms=latency_ms,
        )
    if isinstance(error, httpx.HTTPStatusError):
        header = error.response.headers.get("x-request-id")
        return ExchangeOutcome(
            kind="http_error",
            priority=priority,
            sent_ids=tuple(sent_ids),
            candidate_ids=tuple(candidate_ids),
            fallback_id=fallback_id,
            min_probability=min_probability,
            status=error.response.status_code,
            request_id=header if isinstance(header, str) else None,
            latency_ms=latency_ms,
        )
    if isinstance(error, httpx.HTTPError):
        return ExchangeOutcome(
            kind="connection_error",
            priority=priority,
            sent_ids=tuple(sent_ids),
            candidate_ids=tuple(candidate_ids),
            fallback_id=fallback_id,
            min_probability=min_probability,
            latency_ms=latency_ms,
        )
    if isinstance(error, nace_sdk.NaceError):
        return _from_error(
            error,
            priority=priority,
            sent_ids=sent_ids,
            candidate_ids=candidate_ids,
            fallback_id=fallback_id,
            min_probability=min_probability,
            latency_ms=latency_ms,
        )
    return ExchangeOutcome(
        kind="connection_error",
        priority=priority,
        sent_ids=tuple(sent_ids),
        candidate_ids=tuple(candidate_ids),
        fallback_id=fallback_id,
        min_probability=min_probability,
        latency_ms=latency_ms,
    )


def _without_media(message: BaseMessage, *, keep_images: bool) -> BaseMessage:
    """Call `_media.without_media` without loading the middleware package."""
    return _load_without_media()(message, keep_images=keep_images)


def _load_without_media() -> Callable[..., BaseMessage]:
    global _WITHOUT_MEDIA
    if _WITHOUT_MEDIA is not None:
        return _WITHOUT_MEDIA
    # Importing the middleware package loads the agent framework. Load the helper
    # module on its own so `NaceModelSelector` stays usable without that extra.
    loaded = sys.modules.get("langchain_nace.experimental.middleware._media")
    if loaded is not None:
        _WITHOUT_MEDIA = loaded.without_media
        return _WITHOUT_MEDIA
    path = Path(__file__).resolve().parent / "experimental" / "middleware" / "_media.py"
    spec = importlib.util.spec_from_file_location("_langchain_nace_route_media", path)
    if spec is None or spec.loader is None:
        message = "Could not load langchain_nace message-media helper."
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _WITHOUT_MEDIA = module.without_media
    return _WITHOUT_MEDIA


_WITHOUT_MEDIA: Callable[..., BaseMessage] | None = None


def _append_turn(
    turns: list[RouteTurn], role: Literal["system", "user", "assistant"], text: str
) -> None:
    if text:
        turns.append({"role": role, "content": text})


def _trim_turns(turns: list[RouteTurn]) -> list[RouteTurn]:
    """Keep the latest human turn, then the newest other turns that fit.

    The result stays in chronological order. A human turn at least as long as
    the budget is cut to the budget and sent alone.
    """
    user_at = None
    for index, turn in enumerate(turns):
        if turn["role"] == "user" and turn["content"]:
            user_at = index
    if user_at is None:
        return []
    user = turns[user_at]
    if len(user["content"]) >= _CHAR_BUDGET:
        return [{"role": "user", "content": user["content"][:_CHAR_BUDGET]}]
    chosen = [user_at]
    remaining = _CHAR_BUDGET - len(user["content"])
    slots = _MAX_ROUTE_MESSAGES - 1
    for index in range(len(turns) - 1, -1, -1):
        if slots == 0 or remaining == 0:
            break
        if index == user_at:
            continue
        size = len(turns[index]["content"])
        if size == 0 or size > remaining:
            continue
        chosen.append(index)
        remaining -= size
        slots -= 1
    chosen.sort()
    return [turns[index] for index in chosen]


def _content_characters(message: BaseMessage) -> int:
    content = message.content
    if isinstance(content, str):
        return len(content)
    try:
        return len(json.dumps(content, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(content))


def _check_id(candidate_id: object) -> None:
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        message = "Candidate ids must not be blank."
        raise ValueError(message)
    if len(candidate_id) > _MAX_ID_LENGTH:
        message = "Candidate ids must be at most 256 characters."
        raise ValueError(message)


def _price(candidate_id: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"Candidate {candidate_id!r} has an invalid profile."
        raise ValueError(message)
    number = float(value)
    if not math.isfinite(number) or number < 0:
        message = f"Candidate {candidate_id!r} has an invalid profile."
        raise ValueError(message)
    return number


def _context_window(candidate_id: str, value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > _MAX_CONTEXT_WINDOW
    ):
        message = f"Candidate {candidate_id!r} has an invalid profile."
        raise ValueError(message)
    return value


def _tags(candidate_id: str, value: object) -> list[str]:
    if not isinstance(value, list) or len(value) > _MAX_TAGS:
        message = f"Candidate {candidate_id!r} has an invalid profile."
        raise ValueError(message)
    tags: list[str] = []
    for tag in value:
        if not isinstance(tag, str) or _is_js_blank(tag) or len(tag) > _MAX_PROFILE_TEXT:
            message = f"Candidate {candidate_id!r} has an invalid profile."
            raise ValueError(message)
        tags.append(tag)
    return tags


def _open_clients(
    *,
    api_key: SecretStr | str | None,
    base_url: str | None,
    timeout: float,
    http_client: httpx.Client | None,
    http_async_client: httpx.AsyncClient | None,
) -> tuple[nace_sdk.NaceClient, nace_sdk.AsyncNaceClient]:
    explicit = NaceClassifier._blank_base_url_is_default(base_url)
    endpoint = (
        explicit
        or NaceClassifier._blank_base_url_is_default(read_env(BASE_URL_ENV))
        or DEFAULT_BASE_URL
    )
    try:
        url = httpx.URL(endpoint)
    except httpx.InvalidURL as error:
        message = "base_url is not a valid URL."
        raise ValueError(message) from error
    if url.scheme not in {"http", "https"} or not url.host:
        message = "base_url must be an absolute http(s) URL."
        raise ValueError(message)
    if _is_openrouter(endpoint) or url.host.removesuffix(".") == "openrouter.ai":
        message = "Drex Router is not available through OpenRouter; use a Nace endpoint."
        raise ValueError(message)
    if api_key is None:
        key: str | None = read_env(API_KEY_ENV)
    else:
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    http = http_client or httpx.Client(timeout=timeout)
    if _set_user_agent not in http.event_hooks["request"]:
        http.event_hooks["request"].append(_set_user_agent)
    async_http = http_async_client or httpx.AsyncClient(timeout=timeout)
    if _aset_user_agent not in async_http.event_hooks["request"]:
        async_http.event_hooks["request"].append(_aset_user_agent)
    secrets = tuple(
        secret.strip() for secret in (key, read_env(API_KEY_ENV)) if secret and secret.strip()
    )
    _SDK_LOG_REDACTOR.add_secrets(secrets)
    sdk_logger = logging.getLogger(SDK_LOGGER_NAME)
    if _SDK_LOG_REDACTOR not in sdk_logger.filters:
        sdk_logger.addFilter(_SDK_LOG_REDACTOR)
    if _ROUTE_WIRE_REDACTOR not in sdk_logger.filters:
        sdk_logger.addFilter(_ROUTE_WIRE_REDACTOR)
    options: dict[str, Any] = {
        "api_key": key,
        "base_url": endpoint,
        "retry": _NO_RETRY,
        "timeout": timeout,
        "client_info": CLIENT_NAME,
    }
    try:
        sync = nace_sdk.NaceClient(**options, http_client=http)
        async_client = nace_sdk.AsyncNaceClient(**options, http_client=async_http)
    except nace_sdk.NaceError as error:
        raise ValueError(str(error)) from error
    return sync, async_client


__all__ = [
    "CandidateBinding",
    "CandidateProfile",
    "ExchangeOutcome",
    "ModelSelection",
    "NaceModelSelector",
    "RouteTurn",
    "apply_outcome",
    "aroute_exchange",
    "filter_candidates",
    "has_image_block",
    "local_outcome",
    "project_route_messages",
    "request_characters",
    "response_is_valid",
    "route_exchange",
    "selection_from_outcome",
    "validate_candidate_map",
    "validate_profile",
]
