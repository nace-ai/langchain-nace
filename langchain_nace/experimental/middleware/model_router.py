"""Experimental model-routing middleware powered by Drex."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

try:
    from langchain.agents.middleware.types import (
        AgentMiddleware,
        AgentState,
        ContextT,
        ModelRequest,
        ModelResponse,
        ResponseT,
        TracePolicy,
        omit_payload,
    )
    from langchain.chat_models import init_chat_model
    from langgraph.runtime import Runtime
except ImportError as error:  # pragma: no cover
    message = (
        "NaceModelRouterMiddleware needs the LangChain agent framework: "
        "pip install 'langchain-nace[experimental]'"
    )
    raise ImportError(message) from error

from typing import NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict, override

from langchain_nace.classifier import NaceClassifier, is_text_only
from langchain_nace.errors import NaceAPIError, NaceError
from langchain_nace.experimental.middleware._media import without_media
from langchain_nace.types import Choice, ChoiceAnswer, ClassifierRequest

logger = logging.getLogger(__name__)

_QUESTION_ID = "model_route"
# Statuses after which routing is retried once with images replaced by placeholders.
_IMAGE_RETRY_STATUSES = frozenset({400, 413, 422})


@dataclass(frozen=True)
class ModelChoice:
    """A model the router can pick, and when to pick it.

    Args:
        model: A chat model, or a model string accepted by `init_chat_model`.
        criteria: The kind of task this model suits.

    """

    model: str | BaseChatModel
    criteria: str


class NaceModelRoute(TypedDict):
    """The routing decision kept in agent state under `nace_model_route`.

    A plain mapping, so checkpointers can store it without custom types.
    """

    route: str
    """The model choice used for this run."""

    choice: str | None
    """What Drex picked, or `None` when Drex could not be asked."""

    confidence: float | None
    probabilities: dict[str, float]
    escalated: bool
    """Whether `route` is `escalate_to` instead of Drex's pick."""

    reason: str
    """`"confident"`, `"low_confidence"` or `"error"`."""


class _RouterState(AgentState[Any]):
    nace_model_route: NotRequired[NaceModelRoute | None]


class NaceModelRouterMiddleware(AgentMiddleware[_RouterState, ContextT, ResponseT]):
    """Pick the model for each agent run with a Drex `Choice`.

    Before each run, Drex reads the latest user message and picks one of `choices`.
    When Drex's confidence is below `min_confidence`, or Drex can't be reached, the
    run uses `escalate_to` (by default the last choice, so list choices from cheapest
    to strongest). With no user message, the agent's own model is used. The decision
    and its probabilities are kept in state under `nace_model_route`.

    !!! warning
        Experimental: the API may change without notice.

    Example:
        ```python
        router = NaceModelRouterMiddleware(
            choices={
                "fast": ModelChoice("openai:gpt-5-mini", "Simple, well-scoped tasks."),
                "strong": ModelChoice("anthropic:claude-opus-5-5", "Hard, multi-step work."),
            },
            instructions="Pick the cheapest model that can do this task well.",
        )
        agent = create_agent("openai:gpt-5-mini", tools=tools, middleware=[router])
        ```

    Args:
        choices: Route names mapped to models, cheapest first.
        instructions: The routing question.
        model: Drex model id or a configured `NaceClassifier`. Defaults to `drex-max`.
        min_confidence: Below this, escalate.
        escalate_to: The route to use when escalating. Defaults to the last choice.

    """

    state_schema = _RouterState
    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(
        self,
        *,
        choices: Mapping[str, ModelChoice],
        instructions: str,
        model: str | NaceClassifier | None = None,
        min_confidence: float = 0.6,
        escalate_to: str | None = None,
    ) -> None:
        """Set up the router.

        Raises:
            ValueError: `choices` is empty, `instructions` is blank, or `escalate_to`
                isn't one of `choices`.

        """
        super().__init__()
        if not choices:
            message = "choices must contain at least one model."
            raise ValueError(message)
        if not instructions.strip():
            message = "instructions must not be empty."
            raise ValueError(message)
        self.escalate_to = escalate_to or list(choices)[-1]
        if self.escalate_to not in choices:
            message = f"escalate_to={self.escalate_to!r} is not one of {list(choices)}."
            raise ValueError(message)
        self.min_confidence = min_confidence
        self.question = Choice(
            instructions=instructions,
            criteria={route: choice.criteria for route, choice in choices.items()},
        )
        self.models: dict[str, BaseChatModel] = {
            route: init_chat_model(choice.model) if isinstance(choice.model, str) else choice.model
            for route, choice in choices.items()
        }
        if isinstance(model, NaceClassifier):
            self.classifier = model
        else:
            self.classifier = NaceClassifier(model=model) if model else NaceClassifier()

    def _requests(self, state: _RouterState) -> list[ClassifierRequest]:
        message = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
        )
        if message is None:
            return []
        keep = not is_text_only(self.classifier.model)
        states = [without_media(message, keep_images=keep)]
        if keep:
            text = without_media(message, keep_images=False)
            if text.content != states[0].content:
                states.append(text)
        return [ClassifierRequest(state=s, questions={_QUESTION_ID: self.question}) for s in states]

    def _config(self) -> RunnableConfig:
        return {
            "run_name": "NaceModelRouter",
            "metadata": {
                "nace_min_confidence": self.min_confidence,
                "nace_escalate_to": self.escalate_to,
            },
        }

    def _route(self, answer: ChoiceAnswer | None) -> NaceModelRoute:
        if answer is None:
            return NaceModelRoute(
                route=self.escalate_to,
                choice=None,
                confidence=None,
                probabilities={},
                escalated=True,
                reason="error",
            )
        confident = answer.confidence >= self.min_confidence and answer.choice in self.models
        return NaceModelRoute(
            route=answer.choice if confident else self.escalate_to,
            choice=answer.choice,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
            escalated=not confident,
            reason="confident" if confident else "low_confidence",
        )

    @override
    def before_agent(self, state: _RouterState, runtime: Runtime[ContextT]) -> dict[str, Any]:
        """Ask Drex once per run and store the decision."""
        requests = self._requests(state)
        if not requests:
            return {"nace_model_route": None}
        answer: ChoiceAnswer | None = None
        for index, request in enumerate(requests):
            try:
                answer = self.classifier.invoke(request, self._config()).choices.get(_QUESTION_ID)
                break
            except NaceAPIError as error:
                if index + 1 < len(requests) and error.status in _IMAGE_RETRY_STATUSES:
                    continue
                logger.warning("Drex routing failed; escalating: %s", error)
                break
            except NaceError as error:
                logger.warning("Drex routing failed; escalating: %s", error)
                break
        return {"nace_model_route": self._route(answer)}

    @override
    async def abefore_agent(
        self, state: _RouterState, runtime: Runtime[ContextT]
    ) -> dict[str, Any]:
        """Ask Drex once per run and store the decision."""
        requests = self._requests(state)
        if not requests:
            return {"nace_model_route": None}
        answer: ChoiceAnswer | None = None
        for index, request in enumerate(requests):
            try:
                response = await self.classifier.ainvoke(request, self._config())
                answer = response.choices.get(_QUESTION_ID)
                break
            except NaceAPIError as error:
                if index + 1 < len(requests) and error.status in _IMAGE_RETRY_STATUSES:
                    continue
                logger.warning("Drex routing failed; escalating: %s", error)
                break
            except NaceError as error:
                logger.warning("Drex routing failed; escalating: %s", error)
                break
        return {"nace_model_route": self._route(answer)}

    def _routed(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        route = cast("NaceModelRoute | None", request.state.get("nace_model_route"))
        if route is None:
            return request
        return request.override(model=self.models[route["route"]])

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Call the routed model."""
        return handler(self._routed(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Call the routed model."""
        return await handler(self._routed(request))


__all__ = ["ModelChoice", "NaceModelRoute", "NaceModelRouterMiddleware"]
