import base64
import json
import threading
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from langchain_nace import (
    Choice,
    NaceClassifier,
    NaceImageModelError,
    Noul,
    NoulCriteria,
    Score,
    __version__,
)
from tests.fakes import FakeDrex

FIXTURES = Path(__file__).parents[1] / "fixtures"
RED = base64.b64encode((FIXTURES / "red.png").read_bytes()).decode()
BLUE = base64.b64encode((FIXTURES / "blue.png").read_bytes()).decode()

QUESTIONS = {
    "refund": Noul(instructions="Is the customer asking for a refund?"),
    "team": Choice(instructions="Which team?", criteria={"billing": "Payments", "technical": None}),
    "anger": Score(instructions="How upset?", criteria=["calm", "upset", "furious"]),
}


def test_default_model_is_drex_max(classifier: NaceClassifier, drex: FakeDrex) -> None:
    classifier.invoke({"state": "hi", "questions": {"q": Noul(instructions="Greeting?")}})
    assert classifier.model == "drex-max"
    assert drex.bodies[0]["model"] == "drex-max"


def test_images_work_without_naming_a_model(classifier: NaceClassifier, drex: FakeDrex) -> None:
    classifier.invoke(
        {"state": "x", "questions": {"q": Noul(instructions="Red?")}, "images": [RED]}
    )
    assert drex.bodies[0]["model"] == "drex-max"
    assert drex.bodies[0]["images"] == [RED]


def test_default_model_from_env(monkeypatch: pytest.MonkeyPatch, make_classifier) -> None:
    monkeypatch.setenv("NACE_DEFAULT_DECISION_MODEL", "drex-v9.9")
    assert make_classifier().model == "drex-v9.9"


def test_any_model_id_is_sent_as_given(make_classifier, drex: FakeDrex) -> None:
    make_classifier(model="drex-v7.0-preview").invoke(
        {"state": "x", "questions": {"q": Noul(instructions="?")}}
    )
    assert drex.bodies[0]["model"] == "drex-v7.0-preview"


@pytest.mark.parametrize("name", ["refund", "team", "anger"])
def test_single_question_each_type(classifier: NaceClassifier, name: str) -> None:
    response = classifier.invoke(
        {"state": "I was charged twice", "questions": {name: QUESTIONS[name]}}
    )
    answer = response.answers[name]
    assert answer.type == QUESTIONS[name].type
    if name == "team":
        assert response.choices["team"].probabilities == {"billing": 0.8, "technical": 0.2}
    if name == "anger":
        assert set(response.scores["anger"].probabilities) == {0, 1, 2}
    assert response.usage.input_tokens == 120
    assert response.request_id == "req_test"
    assert response.evaluation_time_ms == 81.5


def test_mixed_questions_go_in_one_request(classifier: NaceClassifier, drex: FakeDrex) -> None:
    response = classifier.invoke({"state": "x", "questions": QUESTIONS})
    assert len(drex.requests) == 1
    assert set(response.answers) == set(QUESTIONS)


def test_wire_body(classifier: NaceClassifier, drex: FakeDrex) -> None:
    classifier.invoke(
        {
            "state": [HumanMessage("refund please")],
            "questions": {
                "n": Noul(criteria=NoulCriteria(true="Asks for money back")),
                "c": Choice(criteria={"a": None, "b": "B"}),
            },
        }
    )
    body = drex.bodies[0]
    assert body["state"] == [{"role": "user", "content": "refund please"}]
    assert body["questions"]["n"] == {
        "type": "noul",
        "instructions": "",
        "criteria": {"true": "Asks for money back"},
    }
    assert body["questions"]["c"] == {
        "type": "choice",
        "instructions": "",
        "criteria": {"a": None, "b": "B"},
    }
    assert "images" not in body


async def test_ainvoke(classifier: NaceClassifier, drex: FakeDrex) -> None:
    response = await classifier.ainvoke({"state": "x", "questions": QUESTIONS})
    assert response.nouls["refund"].noul == 0.9
    assert len(drex.requests) == 1


def test_attribution_headers(classifier: NaceClassifier, drex: FakeDrex) -> None:
    classifier.invoke({"state": "x", "questions": {"q": Noul(instructions="?")}})
    headers = drex.requests[0].headers
    assert headers["user-agent"] == f"langchain-nace/{__version__}"
    assert headers["x-nace-client"] == f"langchain-nace/{__version__}"
    assert headers["authorization"] == "Bearer nace_sk_test"
    assert str(drex.requests[0].url) == "https://drex.test/v1/systemone"


async def test_attribution_headers_async(classifier: NaceClassifier, drex: FakeDrex) -> None:
    await classifier.ainvoke({"state": "x", "questions": {"q": Noul(instructions="?")}})
    assert drex.requests[0].headers["user-agent"] == f"langchain-nace/{__version__}"


def test_batch_runs_at_most_8_at_once(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.delay = 0.05
    results = classifier.batch(
        [{"state": f"s{i}", "questions": {"q": Noul(instructions="?")}} for i in range(20)]
    )
    assert len(results) == 20
    assert 1 < drex.peak_in_flight <= 8


def test_batch_respects_explicit_concurrency(classifier: NaceClassifier, drex: FakeDrex) -> None:
    drex.delay = 0.02
    classifier.batch(
        [{"state": "s", "questions": {"q": Noul(instructions="?")}}] * 6,
        config={"max_concurrency": 2},
    )
    assert drex.peak_in_flight <= 2


def test_timeout_below_60_rejected() -> None:
    with pytest.raises(ValueError, match="60"):
        NaceClassifier(api_key="nace_sk_test", timeout=30)


def test_missing_key_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="NACE_API_KEY"):
        NaceClassifier()


def test_images_with_v2_refused_before_sending(make_classifier, drex: FakeDrex) -> None:
    classifier = make_classifier(model="drex-v2.0")
    with pytest.raises(NaceImageModelError, match="drex-max"):
        classifier.invoke(
            {"state": "x", "questions": {"q": Noul(instructions="?")}, "images": [RED]}
        )
    with pytest.raises(NaceImageModelError):
        classifier.invoke(
            {
                "state": HumanMessage(
                    content=[{"type": "image", "base64": RED, "mime_type": "image/png"}]
                ),
                "questions": {"q": Noul(instructions="?")},
            }
        )
    assert drex.requests == []


def test_images_order_is_kept(make_classifier, drex: FakeDrex) -> None:
    max_ = make_classifier(model="drex-max")
    q = {
        "which": Choice(
            instructions="Which image is red?", criteria={"Image 1": None, "Image 2": None}
        )
    }
    max_.invoke({"state": "Two images.", "questions": q, "images": [RED, BLUE]})
    max_.invoke({"state": "Two images.", "questions": q, "images": [BLUE, RED]})
    assert drex.bodies[0]["images"] == [RED, BLUE]
    assert drex.bodies[1]["images"] == [BLUE, RED]


def test_message_images_follow_explicit_images(make_classifier, drex: FakeDrex) -> None:
    make_classifier(model="drex-max").invoke(
        {
            "state": HumanMessage(
                content=[
                    {"type": "text", "text": "and this"},
                    {"type": "image", "base64": BLUE, "mime_type": "image/png"},
                ]
            ),
            "questions": {"q": Noul(instructions="?")},
            "images": [RED],
        }
    )
    body = drex.bodies[0]
    assert body["images"] == [RED, f"data:image/png;base64,{BLUE}"]
    assert body["state"]["content"][1] == {"type": "text", "text": "[Image 2]"}


def test_dict_message_images_are_sent_as_images(make_classifier, drex: FakeDrex) -> None:
    make_classifier(model="drex-max").invoke(
        {
            "state": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour is this?"},
                        {"type": "image", "base64": RED, "mime_type": "image/png"},
                    ],
                }
            ],
            "questions": {"q": Noul(instructions="Is Image 1 red?")},
        }
    )
    body = drex.bodies[0]
    assert body["images"] == [f"data:image/png;base64,{RED}"]
    assert body["state"][0]["content"][1] == {"type": "text", "text": "[Image 1]"}
    assert RED not in json.dumps(body["state"])


def test_unknown_model_with_images_is_sent(make_classifier, drex: FakeDrex) -> None:
    make_classifier(model="drex-max-v2").invoke(
        {"state": "x", "questions": {"q": Noul(instructions="?")}, "images": [RED]}
    )
    assert drex.bodies[0]["images"] == [RED]


def test_openrouter_format(make_classifier, drex: FakeDrex) -> None:
    router = make_classifier(
        model="nace-ai/drex-max", base_url="https://openrouter.ai/api/v1", api_key="sk-or-test"
    )
    router.invoke(
        {
            "state": {"ticket": 1},
            "questions": {"q": Noul(criteria=NoulCriteria(true="red"))},
            "images": [RED],
        }
    )
    request, body = drex.requests[0], drex.bodies[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/systemone"
    assert request.headers["http-referer"] == "https://nace.ai"
    assert request.headers["x-openrouter-title"] == "Drex for LangChain"
    assert "images" not in body
    assert body["questions"]["q"]["instructions"] == ""
    assert body["state"][0] == {"type": "text", "text": '{"ticket": 1}'}
    assert body["state"][1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{RED}"},
    }


def test_openrouter_text_only_model_id_still_refuses_images(make_classifier) -> None:
    with pytest.raises(NaceImageModelError):
        make_classifier(model="nace-ai/drex-v2.0", base_url="https://openrouter.ai/api").invoke(
            {"state": "x", "questions": {"q": Noul(instructions="?")}, "images": [RED]}
        )


def test_model_kwargs_reach_the_body(make_classifier, drex: FakeDrex) -> None:
    make_classifier(model_kwargs={"future_option": True}).invoke(
        {"state": "x", "questions": {"q": Noul(instructions="?")}}
    )
    assert drex.bodies[0]["future_option"] is True


def test_no_questions_refused(classifier: NaceClassifier) -> None:
    with pytest.raises(ValueError, match="question"):
        classifier.invoke({"state": "x", "questions": {}})


def test_shared_http_client_gets_one_hook(drex: FakeDrex) -> None:
    import httpx

    client = httpx.Client(transport=drex.transport())
    for _ in range(3):
        NaceClassifier(api_key="nace_sk_test", http_client=client)
    assert len(client.event_hooks["request"]) == 1


def test_thread_safe_invoke(classifier: NaceClassifier, drex: FakeDrex) -> None:
    threads = [
        threading.Thread(
            target=classifier.invoke,
            args=({"state": "x", "questions": {"q": Noul(instructions="?")}},),
        )
        for _ in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(drex.requests) == 10


async def test_abatch(classifier: NaceClassifier, drex: FakeDrex) -> None:
    results = await classifier.abatch(
        [{"state": f"s{i}", "questions": {"q": Noul(instructions="?")}} for i in range(5)]
    )
    assert len(results) == 5
    assert len(drex.requests) == 5


def test_batch_with_config_list(classifier: NaceClassifier) -> None:
    out = classifier.batch(
        [{"state": "a", "questions": {"q": Noul(instructions="?")}}] * 2,
        config=[{"tags": ["a"]}, {"tags": ["b"], "max_concurrency": 1}],
    )
    assert len(out) == 2


def test_openrouter_refuses_workspace_files(make_classifier) -> None:
    from langchain_nace import NaceImageError

    with pytest.raises(NaceImageError, match="inline"):
        make_classifier(model="nace-ai/drex-max", base_url="https://openrouter.ai/api").invoke(
            {
                "state": "x",
                "questions": {"q": Noul(instructions="?")},
                "images": [{"type": "workspace_file", "workspace_id": "w", "file_id": "f"}],
            }
        )


def test_serializable(classifier: NaceClassifier) -> None:
    from langchain_core.load import dumpd

    data = dumpd(classifier)
    assert data["id"] == ["langchain", "classifiers", "nace", "NaceClassifier"]
    assert "nace_sk_test" not in str(data)


def test_blank_model_rejected() -> None:
    with pytest.raises(ValueError, match="model"):
        NaceClassifier(api_key="nace_sk_test", model="  ")


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_base_url_means_default(
    monkeypatch: pytest.MonkeyPatch, drex: FakeDrex, value: str
) -> None:
    import httpx

    monkeypatch.setenv("NACE_BASE_URL", value)
    classifier = NaceClassifier(
        api_key="nace_sk_test", http_client=httpx.Client(transport=drex.transport())
    )
    assert classifier.base_url is None
    classifier.invoke({"state": "x", "questions": {"q": Noul(instructions="?")}})
    assert str(drex.requests[0].url) == "https://console.nace.ai/v1/systemone"
    assert NaceClassifier(api_key="nace_sk_test", base_url=value).base_url is None
