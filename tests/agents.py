"""A scripted chat model that emits tool calls, for agent tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedModel(GenericFakeChatModel):
    """Returns the scripted messages in order; `bind_tools` is a no-op."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedModel:  # type: ignore[override]
        return self


def scripted(*messages: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(messages))


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args}])


def score(level: float, levels: int = 5) -> dict[str, Any]:
    legend = [f"level {i}" for i in range(levels)]
    probs = {str(i): (1.0 if i == round(level) else 0.0) for i in range(levels)}
    return {
        "type": "score",
        "score": level,
        "legend": legend,
        "probabilities": probs,
        "confidence": 0.9,
    }


def noul(p: float) -> dict[str, Any]:
    return {"type": "noul", "noul": p}
