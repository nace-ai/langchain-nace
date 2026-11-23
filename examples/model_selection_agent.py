"""Pick the chat model for one agent with Drex Router.

`NaceModelSelectorMiddleware` sends the conversation to `POST /v1/route` and runs
the candidate Drex selects. The chat models are scripted. The route call is real.

    export NACE_API_KEY=nace_sk_...
    uv run python examples/model_selection_agent.py
"""

from __future__ import annotations

import warnings
from typing import Any

from langchain.agents import create_agent
from langchain_core._api import LangChainBetaWarning
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain_nace.experimental.middleware import ModelCandidate, NaceModelSelectorMiddleware

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

# The default route deadline is 3 seconds. A live router call can take longer.
ROUTE_TIMEOUT = 60.0


class _Scripted(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> _Scripted:  # type: ignore[override]
        return self


def main() -> None:
    fast = _Scripted(messages=iter([AIMessage("Reply from the fast candidate.")]))
    strong = _Scripted(messages=iter([AIMessage("Reply from the strong candidate.")]))
    selector = NaceModelSelectorMiddleware(
        candidates={
            "openai/gpt-4o-mini": ModelCandidate(fast),
            "anthropic/claude-haiku-5.5": ModelCandidate(strong),
        },
        priority="balanced",
        fallback="openai/gpt-4o-mini",
        timeout=ROUTE_TIMEOUT,
    )
    agent = create_agent(fast, tools=[], middleware=[selector])
    result = agent.invoke(
        {"messages": [HumanMessage("I was charged twice for March. Please refund one charge.")]}
    )
    selection = result["nace_model_selection"]
    print(f"model={selection['model']}")
    print(f"reason={selection['reason']}")
    print(f"probabilities={selection['probabilities']}")


if __name__ == "__main__":
    main()
