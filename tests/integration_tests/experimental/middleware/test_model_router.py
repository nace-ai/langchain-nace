"""Live model-router check."""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from langchain_nace.experimental.middleware import ModelChoice, NaceModelRouterMiddleware
from tests.agents import scripted


def test_router_picks_and_records_probabilities(live) -> None:
    router = NaceModelRouterMiddleware(
        choices={
            "fast": ModelChoice(scripted(AIMessage("fast")), "Short factual lookups."),
            "strong": ModelChoice(
                scripted(AIMessage("strong")), "Long multi-step engineering work."
            ),
        },
        instructions="Pick the cheapest model that can do this well.",
        model=live("drex-v2.0"),
    )
    agent = create_agent(scripted(AIMessage("default")), tools=[], middleware=[router])
    easy = agent.invoke({"messages": [HumanMessage("What's the capital of France?")]})[
        "nace_model_route"
    ]
    hard = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "Migrate our 40-service monorepo from REST to gRPC, keep backwards "
                    "compatibility, and write the rollout plan."
                )
            ]
        }
    )["nace_model_route"]
    print(f"\neasy -> {easy}\nhard -> {hard}")
    assert easy["probabilities"]["fast"] > hard["probabilities"]["fast"]
    assert abs(sum(easy["probabilities"].values()) - 1) < 1e-2
