"""Experimental model-selection middleware powered by Drex Router."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NotRequired, cast

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        AgentState,
        ContextT,
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
        ResponseT,
        TracePolicy,
        omit_payload,
    )
    from langchain.chat_models import init_chat_model
    from langgraph.runtime import Runtime
    from langgraph.types import Command
except ImportError as error:  # pragma: no cover
    message = (
        "NaceModelSelectorMiddleware needs the LangChain agent framework: "
        "pip install 'langchain-nace[experimental]'"
    )
    raise ImportError(message) from error

from langchain_core.language_models import BaseChatModel
from typing_extensions import TypedDict, override

from langchain_nace.selection import (
    CandidateBinding,
    ExchangeOutcome,
    ModelSelection,
    NaceModelSelector,
    Priority,
    RouteTurn,
    SelectOn,
    apply_outcome,
    filter_candidates,
    has_image_block,
    local_outcome,
    project_route_messages,
    request_characters,
    require_context,
    require_min_probability,
    require_priority,
    require_select_on,
    require_timeout,
    validate_candidate_map,
)

_TOKEN_DIVISOR = 3


@dataclass(frozen=True)
class ModelCandidate:
    """A model the selector may run, and when it may run it.

    Args:
        model: A chat model, or a model string accepted by `init_chat_model`.
        profile: Optional price and context metadata. The candidate id is added as `id`.
        images: Whether this model may answer a request that contains an image.
        tools: Whether this model may answer a request that contains tools.

    """

    model: str | BaseChatModel = field(repr=False)
    profile: Mapping[str, Any] | None = None
    images: bool = True
    tools: bool = True


class _SelectorState(AgentState[Any]):
    nace_model_selection: NotRequired[ModelSelection | None]


class _Prepared(TypedDict):
    outcome: ExchangeOutcome | None
    messages: tuple[RouteTurn, ...]
    bindings: tuple[CandidateBinding, ...]
    eligible: tuple[CandidateBinding, ...]
    fallback_id: str | None


class NaceModelSelectorMiddleware(AgentMiddleware[_SelectorState, ContextT, ResponseT]):
    """Pick the chat model for an agent with Drex Router.

    On the first model call of a run, the conversation and the eligible candidates
    are sent to `POST /v1/route`. A later model call reuses the stored selection
    when that model is still eligible for the current request, or when the agent's
    own model is in use. Otherwise it selects again. `before_agent` only clears
    the selection, so a new run asks again. A resumed run reuses the checkpointed
    selection while that model is still eligible and selects again otherwise.

    !!! warning
        Experimental: the API may change without notice.

    Example:
        ```python
        selector = NaceModelSelectorMiddleware(
            candidates={
                "fast": ModelCandidate("openai:gpt-5-mini"),
                "strong": ModelCandidate("anthropic:claude-opus-5-5"),
            },
            priority="balanced",
            fallback="fast",
        )
        agent = create_agent("openai:gpt-5-mini", tools=tools, middleware=[selector])
        ```

    Args:
        candidates: Candidate ids mapped to the models they may run.
        priority: `quality`, `balanced`, or `cheapest`.
        fallback: Candidate id used when Drex does not return a usable selection.
            `None` keeps the agent's own model.
        min_probability: Below this probability of the selected id, use `fallback`.
        select_on: `run` selects once per run. `model_call` selects on every model call.
        context: `conversation` sends the projected conversation. `latest_human` sends
            the system message and the latest human message.
        timeout: Seconds for the route call. The async path enforces this as a
            deadline. The sync path passes it to the SDK for each HTTP operation.
        selector: A configured `NaceModelSelector`. When omitted, one is built from
            the environment.

    Raises:
        ValueError: The candidates, fallback, priority, timeout, or profile are invalid.

    """

    state_schema = _SelectorState
    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(
        self,
        *,
        candidates: Mapping[str, ModelCandidate],
        priority: str = "balanced",
        fallback: str | None = None,
        min_probability: float | None = None,
        select_on: str = "run",
        context: str = "conversation",
        timeout: float = 3.0,
        selector: NaceModelSelector | None = None,
    ) -> None:
        """Validate the candidates and keep the selector used for route calls."""
        super().__init__()
        profiles = {
            candidate_id: candidate.profile for candidate_id, candidate in candidates.items()
        }
        validated = validate_candidate_map(profiles, fallback=fallback)
        self._bindings = tuple(
            CandidateBinding(
                id=binding.id,
                profile=binding.profile,
                images=candidates[binding.id].images,
                tools=candidates[binding.id].tools,
            )
            for binding in validated
        )
        self._models = {binding.id: _chat_model(candidates[binding.id]) for binding in validated}
        self.priority: Priority = require_priority(priority)
        self.fallback = fallback
        self.min_probability = require_min_probability(min_probability)
        self.select_on: SelectOn = require_select_on(select_on)
        self.context = require_context(context)
        self.timeout = require_timeout(timeout)
        self._selector = (
            selector if selector is not None else NaceModelSelector(timeout=self.timeout)
        )

    @override
    def before_agent(self, state: _SelectorState, runtime: Runtime[ContextT]) -> dict[str, Any]:
        """Start the run with no stored selection. This makes no network call."""
        return {"nace_model_selection": None}

    @override
    async def abefore_agent(
        self, state: _SelectorState, runtime: Runtime[ContextT]
    ) -> dict[str, Any]:
        """Start the run with no stored selection. This makes no network call."""
        return {"nace_model_selection": None}

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ExtendedModelResponse[ResponseT]:
        """Select a model when needed and call the handler once."""
        selection = self._selection(request)
        return _with_selection(self._call_model(request, handler, selection), selection)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ExtendedModelResponse[ResponseT]:
        """Select a model when needed and call the handler once."""
        selection = await self._aselection(request)
        response = await self._call_model(request, handler, selection)
        return _with_selection(response, selection)

    def _selection(self, request: ModelRequest[ContextT]) -> ModelSelection:
        prepared = self._prepare(request)
        stored = self._reusable(request, prepared["eligible"])
        if stored is not None:
            return stored
        if prepared["outcome"] is not None:
            return apply_outcome(prepared["outcome"])
        return self._selector.exchange(
            prepared["messages"],
            prepared["bindings"],
            priority=self.priority,
            fallback_id=prepared["fallback_id"],
            min_probability=self.min_probability,
            timeout=self.timeout,
        )

    async def _aselection(self, request: ModelRequest[ContextT]) -> ModelSelection:
        prepared = self._prepare(request)
        stored = self._reusable(request, prepared["eligible"])
        if stored is not None:
            return stored
        if prepared["outcome"] is not None:
            return apply_outcome(prepared["outcome"])
        return await self._selector.aexchange(
            prepared["messages"],
            prepared["bindings"],
            priority=self.priority,
            fallback_id=prepared["fallback_id"],
            min_probability=self.min_probability,
            timeout=self.timeout,
        )

    def _reusable(
        self, request: ModelRequest[ContextT], eligible: Sequence[CandidateBinding]
    ) -> ModelSelection | None:
        """Return the stored selection when this request can still use it."""
        if self.select_on != "run":
            return None
        value = request.state.get("nace_model_selection")
        try:
            if not isinstance(value, Mapping) or "model" not in value:
                return None
            model = value["model"]
        except (KeyError, TypeError):
            return None
        # None means the agent's own model. It has no candidate capability flags.
        if model is None:
            return cast("ModelSelection", value)
        configured = {binding.id for binding in self._bindings}
        eligible_ids = {binding.id for binding in eligible}
        if isinstance(model, str) and model in configured and model in eligible_ids:
            return cast("ModelSelection", value)
        return None

    def _prepare(self, request: ModelRequest[ContextT]) -> _Prepared:
        messages = project_route_messages(
            request.messages, system_message=request.system_message, context=self.context
        )
        has_image = has_image_block(request.messages, request.system_message)
        has_tools = bool(request.tools) or request.response_format is not None
        tokens = request_characters(request.messages, request.system_message) / _TOKEN_DIVISOR
        eligible = filter_candidates(
            self._bindings,
            has_image=has_image,
            has_tools=has_tools,
            estimated_tokens=tokens,
        )
        fallback_id = _fallback_id(
            self.fallback,
            self._bindings,
            has_image=has_image,
            has_tools=has_tools,
            estimated_tokens=tokens,
        )
        outcome = local_outcome(
            messages,
            eligible,
            priority=self.priority,
            fallback_id=fallback_id,
            min_probability=self.min_probability,
            candidate_ids=tuple(binding.id for binding in self._bindings),
        )
        if outcome is not None:
            return {
                "outcome": outcome,
                "messages": (),
                "bindings": (),
                "eligible": tuple(eligible),
                "fallback_id": fallback_id,
            }
        return {
            "outcome": None,
            "messages": tuple(messages),
            "bindings": tuple(eligible),
            "eligible": tuple(eligible),
            "fallback_id": fallback_id,
        }

    def _call_model(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Any],
        selection: ModelSelection,
    ) -> Any:
        """Call `handler` once. This is the only place the middleware does so."""
        model_id = selection["model"]
        bound = self._models.get(model_id) if isinstance(model_id, str) else None
        prepared = request if bound is None else request.override(model=bound)
        return handler(prepared)


def _with_selection(
    response: ModelResponse[ResponseT], selection: ModelSelection
) -> ExtendedModelResponse[ResponseT]:
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={"nace_model_selection": dict(selection)}),
    )


def _fallback_id(
    fallback: str | None,
    bindings: Sequence[CandidateBinding],
    *,
    has_image: bool,
    has_tools: bool,
    estimated_tokens: float,
) -> str | None:
    if fallback is None:
        return None
    matched = next(binding for binding in bindings if binding.id == fallback)
    eligible = filter_candidates(
        (matched,),
        has_image=has_image,
        has_tools=has_tools,
        estimated_tokens=estimated_tokens,
    )
    return fallback if eligible else None


def _chat_model(candidate: ModelCandidate) -> BaseChatModel:
    model = candidate.model
    if isinstance(model, str):
        if not model.strip():
            message = "A candidate model name must not be blank."
            raise ValueError(message)
        return init_chat_model(model)
    if isinstance(model, BaseChatModel):
        return model
    message = "A candidate model must be a chat model or a model name."
    raise ValueError(message)


__all__ = ["ModelCandidate", "NaceModelSelectorMiddleware"]
