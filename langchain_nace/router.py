"""`NaceRouter`: route LCEL inputs with a Drex `Choice`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnableSerializable
from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from langchain_nace.classifier import NaceClassifier
from langchain_nace.errors import NaceError
from langchain_nace.types import Choice, ChoiceAnswer, ClassifierRequest

_QUESTION_ID = "route"


class NaceLowConfidenceError(NaceError):
    """Drex was less confident than `min_confidence` and no fallback was set."""


class RouteDecision(BaseModel):
    """Which route Drex picked, with the full answer."""

    route: str
    """The selected label, or the fallback when confidence was too low."""

    answer: ChoiceAnswer
    """Drex's answer, with every label's probability."""

    fell_back: bool = False
    """Whether `route` is the fallback because confidence was below the threshold."""


class NaceRouter(RunnableSerializable[Any, RouteDecision]):
    """Pick a route for an input with a Drex `Choice` question.

    Use it on its own, or with `branch()` in place of `RunnableBranch` conditions or
    an output parser that routes on generated text.

    Example:
        ```python
        router = NaceRouter(
            classifier=NaceClassifier(),
            question=Choice(
                instructions="Which team should answer?",
                criteria={"billing": "Payments", "technical": "Bugs"},
            ),
            state=lambda x: x["text"],
            min_confidence=0.6,
            fallback="human",
        )
        chain = router.branch(
            {"billing": billing_chain, "technical": tech_chain, "human": escalate},
            default=escalate,
        )
        chain.invoke({"text": "I was charged twice"})
        ```

    """

    classifier: NaceClassifier
    question: Choice
    state: Callable[[Any], Any] | None = None
    """Turns the input into Drex state. Defaults to the input itself."""
    min_confidence: float = 0.0
    fallback: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _request(self, value: Any) -> ClassifierRequest:
        state = self.state(value) if self.state else value
        return {"state": state, "questions": {_QUESTION_ID: self.question}}

    def _decide(self, answer: ChoiceAnswer) -> RouteDecision:
        if answer.confidence >= self.min_confidence:
            return RouteDecision(route=answer.choice, answer=answer)
        if self.fallback is None:
            message = (
                f"Drex picked {answer.choice!r} with confidence {answer.confidence:.2f}, "
                f"below min_confidence={self.min_confidence}, and no fallback is set."
            )
            raise NaceLowConfidenceError(message)
        return RouteDecision(route=self.fallback, answer=answer, fell_back=True)

    def _config(self, config: RunnableConfig | None) -> RunnableConfig:
        base: RunnableConfig = config or {}
        metadata = {
            **(base.get("metadata") or {}),
            "nace_min_confidence": self.min_confidence,
            "nace_fallback": self.fallback,
        }
        return {**base, "metadata": metadata}

    @override
    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> RouteDecision:
        response = self.classifier.invoke(self._request(input), self._config(config))
        return self._decide(response.choices[_QUESTION_ID])

    @override
    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> RouteDecision:
        response = await self.classifier.ainvoke(self._request(input), self._config(config))
        return self._decide(response.choices[_QUESTION_ID])

    def branch(
        self, routes: Mapping[str, Runnable[Any, Any]], default: Runnable[Any, Any]
    ) -> Runnable[Any, Any]:
        """Run `routes[decision.route]` (or `default`) on the original input."""

        def run(value: Any, config: RunnableConfig) -> Any:
            decision = self.invoke(value, config)
            return routes.get(decision.route, default).invoke(value, config)

        async def arun(value: Any, config: RunnableConfig) -> Any:
            decision = await self.ainvoke(value, config)
            return await routes.get(decision.route, default).ainvoke(value, config)

        return RunnableLambda(run, afunc=arun, name="NaceRouterBranch")


__all__ = ["NaceLowConfidenceError", "NaceRouter", "RouteDecision"]
