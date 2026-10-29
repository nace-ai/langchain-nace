import base64
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from langchain_nace import NaceClassifier
from langchain_nace.experimental.middleware import ModelChoice, NaceModelRouterMiddleware
from tests.agents import scripted
from tests.fakes import FakeDrex, error

RED = base64.b64encode((Path(__file__).parents[3] / "fixtures" / "red.png").read_bytes()).decode()


def router(classifier: NaceClassifier, **kwargs: object) -> NaceModelRouterMiddleware:
    return NaceModelRouterMiddleware(
        choices={
            "fast": ModelChoice(scripted(AIMessage("from fast")), "Simple tasks."),
            "strong": ModelChoice(scripted(AIMessage("from strong")), "Hard tasks."),
        },
        instructions="Pick the cheapest model that can do this.",
        model=classifier,
        **kwargs,  # type: ignore[arg-type]
    )


def run(middleware: NaceModelRouterMiddleware, text: object = "What is 2+2?") -> dict:
    agent = create_agent(scripted(AIMessage("from default")), tools=[], middleware=[middleware])
    return agent.invoke({"messages": [HumanMessage(text)]})  # type: ignore[arg-type]


def test_uses_the_model_drex_picks(classifier: NaceClassifier, drex: FakeDrex) -> None:
    result = run(router(classifier))
    assert result["messages"][-1].content == "from fast"
    route = result["nace_model_route"]
    assert route == {
        "route": "fast",
        "choice": "fast",
        "confidence": 0.8,
        "probabilities": {"fast": 0.8, "strong": 0.2},
        "escalated": False,
        "reason": "confident",
    }
    assert drex.bodies[0]["questions"]["model_route"]["criteria"] == {
        "fast": "Simple tasks.",
        "strong": "Hard tasks.",
    }


def test_escalates_below_threshold(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.overrides["model_route"] = {
        "type": "choice",
        "choice": "fast",
        "confidence": 0.3,
        "probabilities": {"fast": 0.55, "strong": 0.45},
    }
    result = run(router(classifier))
    assert result["messages"][-1].content == "from strong"
    assert result["nace_model_route"]["reason"] == "low_confidence"
    assert result["nace_model_route"]["choice"] == "fast"


def test_escalates_when_drex_fails(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.replies.extend([error(529, "overloaded", kind="overloaded_error")] * 3)
    result = run(router(classifier))
    assert result["messages"][-1].content == "from strong"
    assert result["nace_model_route"]["reason"] == "error"


async def test_async(classifier: NaceClassifier) -> None:
    agent = create_agent(
        scripted(AIMessage("from default")), tools=[], middleware=[router(classifier)]
    )
    result = await agent.ainvoke({"messages": [HumanMessage("hi")]})
    assert result["messages"][-1].content == "from fast"


def test_no_user_message_keeps_the_agents_model(classifier: NaceClassifier, drex: FakeDrex) -> None:
    middleware = router(classifier)
    assert middleware.before_agent({"messages": []}, None) == {"nace_model_route": None}  # type: ignore[arg-type]
    assert drex.requests == []


def test_escalate_to_must_be_a_choice(classifier: NaceClassifier) -> None:
    with pytest.raises(ValueError, match="escalate_to"):
        router(classifier, escalate_to="medium")


def test_image_on_text_model_becomes_placeholder(make_classifier, drex: FakeDrex) -> None:
    run(
        router(make_classifier(model="drex-v2.0")),
        [
            {"type": "text", "text": "what is this"},
            {"type": "image", "base64": RED, "mime_type": "image/png"},
        ],
    )
    body = drex.bodies[0]
    assert "images" not in body
    assert body["state"]["content"] == "what is this\n[image omitted]"


def test_image_rejected_then_retried_without_it(make_classifier, drex: FakeDrex) -> None:
    drex.replies.append(
        error(422, "bad", issues=[{"path": "images.0", "message": "unreadable image"}])
    )
    result = run(
        router(make_classifier(model="drex-max")),
        [
            {"type": "text", "text": "what is this"},
            {"type": "image", "base64": RED, "mime_type": "image/png"},
        ],
    )
    assert len(drex.requests) == 2
    assert drex.bodies[0]["images"] == [f"data:image/png;base64,{RED}"]
    assert "images" not in drex.bodies[1]
    assert result["nace_model_route"]["reason"] == "confident"


async def test_async_escalates_when_drex_fails(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.replies.extend([error(529, "overloaded", kind="overloaded_error")] * 3)
    agent = create_agent(
        scripted(AIMessage("from default")), tools=[], middleware=[router(classifier)]
    )
    result = await agent.ainvoke({"messages": [HumanMessage("hi")]})
    assert result["nace_model_route"]["reason"] == "error"


async def test_async_image_retry(make_classifier, drex: FakeDrex) -> None:
    drex.replies.append(error(422, "bad", issues=[{"path": "images.0", "message": "unreadable"}]))
    agent = create_agent(
        scripted(AIMessage("x")), tools=[], middleware=[router(make_classifier(model="drex-max"))]
    )
    await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "what"},
                        {"type": "image", "base64": RED, "mime_type": "image/png"},
                    ]
                )
            ]
        }
    )
    assert len(drex.requests) == 2


def test_model_given_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_test")
    middleware = NaceModelRouterMiddleware(
        choices={"a": ModelChoice(scripted(AIMessage("a")), "A")},
        instructions="pick",
        model="drex-max",
    )
    assert middleware.classifier.model == "drex-max"
    with pytest.raises(ValueError, match="instructions"):
        NaceModelRouterMiddleware(
            choices={"a": ModelChoice(scripted(AIMessage("a")), "A")}, instructions=" "
        )
    with pytest.raises(ValueError, match="choices"):
        NaceModelRouterMiddleware(choices={}, instructions="pick")
