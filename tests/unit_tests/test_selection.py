"""Pure selection rules and NaceModelSelector. Case 6 is JavaScript only."""

import json
import logging
import math
import time

import httpx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tracers.context import collect_runs

from langchain_nace._ndi import _SDK_LOG_REDACTOR
from langchain_nace.classifier import CLIENT_NAME
from langchain_nace.selection import (
    CandidateBinding,
    CandidateProfile,
    ExchangeOutcome,
    NaceModelSelector,
    _RoutePlan,
    aroute_exchange,
    filter_candidates,
    has_image_block,
    project_route_messages,
    response_is_valid,
    route_exchange,
    selection_from_outcome,
    validate_candidate_map,
)
from tests.fakes import FakeRoute, error, route_ok

CANARY = "CANARY route secret 9f3a"
PROFILE: CandidateProfile = {
    "id": "custom",
    "provider": "openai",
    "price_in": 1.0,
    "price_out": 2.0,
    "context": 8000,
    "tags": ["chat"],
}


def selector(route: FakeRoute, **kwargs: object) -> NaceModelSelector:
    options: dict[str, object] = {
        "api_key": "nace_sk_test",
        "base_url": "https://drex.test",
        "http_client": httpx.Client(transport=route.transport()),
        "http_async_client": httpx.AsyncClient(transport=route.async_transport()),
        "candidates": {"fast": None, "strong": None},
        "fallback": "strong",
    }
    options.update(kwargs)
    return NaceModelSelector(**options)  # type: ignore[arg-type]


def outcome(**kwargs: object) -> ExchangeOutcome:
    base: dict[str, object] = {
        "kind": "response",
        "priority": "balanced",
        "sent_ids": ("fast", "strong"),
        "candidate_ids": ("fast", "strong"),
        "fallback_id": "strong",
    }
    base.update(kwargs)
    return ExchangeOutcome(**base)  # type: ignore[arg-type]


def test_selection_from_outcome_covers_the_reason_table() -> None:
    # Case 2, the mapping half: each outcome kind becomes that reason and model.
    rows: list[tuple[ExchangeOutcome, str, str | None]] = [
        (
            outcome(
                selected="fast",
                probabilities={"fast": 0.8, "strong": 0.2},
                min_probability=0.6,
            ),
            "selected",
            "fast",
        ),
        (
            outcome(
                selected="fast",
                probabilities={"fast": 0.4, "strong": 0.6},
                min_probability=0.6,
            ),
            "low_probability",
            "strong",
        ),
        (
            outcome(kind="single_candidate", single_id="fast", sent_ids=()),
            "single_candidate",
            "fast",
        ),
        (outcome(kind="no_input", sent_ids=(), fallback_id="strong"), "no_input", None),
        (
            outcome(kind="no_eligible_candidate", sent_ids=(), fallback_id=None),
            "no_eligible_candidate",
            None,
        ),
        (
            outcome(kind="no_eligible_candidate", sent_ids=(), fallback_id="strong"),
            "no_eligible_candidate",
            "strong",
        ),
        (outcome(kind="timeout"), "timeout", "strong"),
        (outcome(kind="connection_error"), "connection_error", "strong"),
        (outcome(kind="http_error", status=429), "rate_limited", "strong"),
        (outcome(kind="http_error", status=500), "unavailable", "strong"),
        (outcome(kind="http_error", status=503), "unavailable", "strong"),
        (outcome(kind="http_error", status=504), "unavailable", "strong"),
        (outcome(kind="http_error", status=529), "unavailable", "strong"),
        (outcome(kind="http_error", status=401), "auth_error", "strong"),
        (outcome(kind="http_error", status=403), "auth_error", "strong"),
        (outcome(kind="http_error", status=402), "no_funds", "strong"),
        (outcome(kind="http_error", status=413), "invalid_request", "strong"),
        (outcome(kind="http_error", status=422), "invalid_request", "strong"),
        (outcome(kind="invalid_response"), "invalid_response", "strong"),
    ]
    for item, reason, model in rows:
        selection = selection_from_outcome(item)
        assert selection["reason"] == reason
        assert selection["model"] == model
        assert selection["fallback"] is (reason != "selected")
        assert CANARY not in json.dumps(selection)


def test_invalid_selected_id_is_invalid_response() -> None:
    # Case 3.
    selection = selection_from_outcome(
        outcome(selected="other", probabilities={"fast": 0.5, "strong": 0.5})
    )
    assert selection["reason"] == "invalid_response"
    assert selection["selected"] is None
    assert selection["model"] == "strong"
    assert selection["probabilities"] == {}


def test_probability_keys_must_match_sent_ids() -> None:
    # Case 4.
    selection = selection_from_outcome(
        outcome(selected="fast", probabilities={"fast": 1.0, "extra": 0.0})
    )
    assert selection["reason"] == "invalid_response"


@pytest.mark.parametrize("value", [math.nan, -0.1, 1.1])
def test_probability_out_of_range_is_invalid_response(value: float) -> None:
    # Case 5.
    assert response_is_valid("fast", {"fast": value, "strong": 0.0}, ("fast", "strong")) is None
    selection = selection_from_outcome(
        outcome(selected="fast", probabilities={"fast": value, "strong": 0.0})
    )
    assert selection["reason"] == "invalid_response"


def test_route_body_has_only_the_documented_fields() -> None:
    # Case 9.
    route = FakeRoute()
    selector(
        route,
        candidates={
            "fast": None,
            "custom": {key: value for key, value in PROFILE.items() if key != "id"},
        },
        fallback="fast",
    ).select(
        [
            SystemMessage("Be brief."),
            HumanMessage("Need a refund."),
            AIMessage("I can help."),
        ]
    )
    body = route.bodies[0]
    assert set(body) == {"messages", "candidates", "selection_priority"}
    assert body["selection_priority"] == "balanced"
    for message in body["messages"]:
        assert set(message) == {"role", "content"}
        assert message["role"] in {"system", "user", "assistant"}
        assert isinstance(message["content"], str)
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert body["candidates"][0] == "fast"
    assert body["candidates"][1]["id"] == "custom"
    assert set(body["candidates"][1]) == {
        "id",
        "provider",
        "price_in",
        "price_out",
        "context",
        "tags",
    }


def test_image_tool_call_and_tool_result_are_projected() -> None:
    # Case 10.
    projected = project_route_messages(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "look"},
                    {"type": "image", "base64": "IMGSECRET", "mime_type": "image/png"},
                ]
            ),
            AIMessage(
                content="checking",
                tool_calls=[{"id": "c1", "name": "lookup", "args": {"q": "CALLSECRET"}}],
            ),
            ToolMessage(content="TOOLSECRET result", tool_call_id="c1"),
            HumanMessage("thanks"),
        ],
        system_message=SystemMessage("SYS-RULE"),
    )
    blob = json.dumps(projected)
    assert "IMGSECRET" not in blob
    assert "CALLSECRET" not in blob
    assert "TOOLSECRET" not in blob
    assert "[image omitted]" in blob
    assert [turn["role"] for turn in projected] == ["system", "user", "assistant", "user"]
    assert projected[1]["content"] == "look[image omitted]"
    assert projected[2]["content"] == "checking"


def test_latest_human_keeps_system_and_the_latest_user() -> None:
    projected = project_route_messages(
        [HumanMessage("first"), AIMessage("middle"), HumanMessage("last")],
        system_message=SystemMessage("rule"),
        context="latest_human",
    )
    assert projected == [
        {"role": "system", "content": "rule"},
        {"role": "user", "content": "last"},
    ]


def test_long_latest_human_is_cut_to_the_budget() -> None:
    projected = project_route_messages(
        [SystemMessage("rule"), HumanMessage("x" * 30_000)],
    )
    assert projected == [{"role": "user", "content": "x" * 24_000}]


def test_projection_reaches_both_caps() -> None:
    many = project_route_messages([HumanMessage("h")] + [AIMessage("a")] * 2000)
    assert len(many) == 1024
    packed = project_route_messages(
        [HumanMessage("h"), AIMessage("a" * 20_000), AIMessage("b" * 20_000)]
    )
    total = sum(len(turn["content"]) for turn in packed)
    assert total <= 24_000
    assert "b" * 20_000 in [turn["content"] for turn in packed]
    assert "a" * 20_000 not in [turn["content"] for turn in packed]


def test_message_cap_keeps_the_latest_human() -> None:
    projected = project_route_messages([HumanMessage("keep")] + [AIMessage("a")] * 1100)
    assert len(projected) == 1024
    assert projected[0] == {"role": "user", "content": "keep"}


def test_no_human_text_projects_to_nothing() -> None:
    assert project_route_messages([AIMessage("only a model turn")]) == []
    assert project_route_messages([HumanMessage("")]) == []


def test_filter_drops_image_tool_and_small_context_candidates() -> None:
    bindings = [
        CandidateBinding("vision", images=True, tools=True),
        CandidateBinding("blind", images=False, tools=True),
        CandidateBinding("plain", images=True, tools=False),
        CandidateBinding(
            "small", images=True, tools=True, profile={**PROFILE, "id": "small", "context": 2}
        ),
        CandidateBinding(
            "roomy", images=True, tools=True, profile={**PROFILE, "id": "roomy", "context": 100}
        ),
    ]
    chosen = filter_candidates(bindings, has_image=True, has_tools=True, estimated_tokens=10)
    assert [binding.id for binding in chosen] == ["vision", "roomy"]
    assert has_image_block(
        [HumanMessage(content=[{"type": "image_url", "image_url": "https://example.test/a.png"}])]
    )


def test_route_slower_than_timeout_uses_fallback_and_discards_the_body() -> None:
    # Case 7, sync: the SDK timeout is best-effort per HTTP operation.
    route = FakeRoute(delay=30)
    chosen = selector(route, timeout=0.2).select([HumanMessage("hello")])
    assert chosen["reason"] == "timeout"
    assert chosen["model"] == "strong"
    assert route.delivered == 0
    assert route.withheld
    assert route.withheld[0]["selected"] not in {chosen["model"], chosen["selected"]}
    assert len(route.requests) == 1


async def test_async_deadline_discards_a_late_response() -> None:
    # Case 7, async.
    route = FakeRoute(delay=5)
    chosen = await selector(route, timeout=0.2).aselect([HumanMessage("hello")])
    assert chosen["reason"] == "timeout"
    assert chosen["model"] == "strong"
    assert route.delivered == 0
    assert route.in_flight == 0
    assert chosen["selected"] is None


def test_http_failures_map_to_reasons_and_one_attempt(caplog: pytest.LogCaptureFixture) -> None:
    route = FakeRoute(replies=[error(429, "CANARY route secret 9f3a", kind="rate_limit_error")])
    with caplog.at_level(logging.DEBUG):
        chosen = selector(route).select([HumanMessage(CANARY)])
    assert chosen["reason"] == "rate_limited"
    assert chosen["model"] == "strong"
    assert len(route.requests) == 1
    assert CANARY not in caplog.text
    assert "rate_limited" in caplog.text
    assert "nace_sk_test" not in caplog.text


def test_negative_probability_from_the_server_is_invalid_response() -> None:
    route = FakeRoute(replies=[route_ok("fast", {"fast": -0.1, "strong": 0.5})])
    chosen = selector(route).select([HumanMessage("hello")])
    assert chosen["reason"] == "invalid_response"
    assert chosen["probabilities"] == {}


def test_probability_above_one_from_the_server_is_invalid_response() -> None:
    route = FakeRoute(replies=[route_ok("fast", {"fast": 1.4, "strong": 0.2})])
    chosen = selector(route).select([HumanMessage("hello")])
    assert chosen["reason"] == "invalid_response"


def test_canary_stays_out_of_selection_logs_and_trace_inputs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Case 17.
    route = FakeRoute(replies=[error(422, CANARY)])
    with caplog.at_level(logging.DEBUG), collect_runs() as tracer:
        chosen = selector(route).select([HumanMessage(CANARY)])
    assert chosen["reason"] == "invalid_request"
    assert CANARY not in json.dumps(chosen)
    assert CANARY not in caplog.text
    traced = [run for root in tracer.traced_runs for run in _walk(root)]
    named = [run for run in traced if run.name == "NaceModelSelector"]
    assert named
    assert CANARY not in json.dumps(named[0].inputs, default=str)
    assert "nace_sk_test" not in json.dumps(named[0].inputs, default=str)


def test_headers_name_this_package() -> None:
    # Case 19.
    route = FakeRoute()
    selector(route).select([HumanMessage("hello")])
    headers = route.requests[0].headers
    assert headers["user-agent"] == CLIENT_NAME
    assert headers["x-nace-client"] == CLIENT_NAME


@pytest.mark.parametrize(
    "base_url",
    [
        "https://openrouter.ai/api/v1",
        "https://openrouter.ai./api/v1",
        "https://openrouter。ai/api",
    ],
)
def test_openrouter_is_refused(base_url: str) -> None:
    # Case 18.
    with pytest.raises(ValueError, match="OpenRouter"):
        NaceModelSelector(
            api_key="nace_sk_test",
            base_url=base_url,
            candidates={"fast": None},
        )


def test_openrouter_from_the_environment_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NACE_BASE_URL", "https://openrouter.ai/api")
    with pytest.raises(ValueError, match="OpenRouter"):
        NaceModelSelector(api_key="nace_sk_test", candidates={"fast": None})


@pytest.mark.parametrize(
    ("candidates", "fallback", "match"),
    [
        ({}, None, "at least one"),
        ({f"c{i}": None for i in range(256)}, None, "255"),
        ({"": None}, None, "blank"),
        ({"   ": None}, None, "blank"),
        ({"x" * 257: None}, None, "256"),
        ({"fast": None}, "missing", "fallback"),
        ({"fast": {"provider": "openai"}}, None, "invalid profile"),
        ({"fast": {**PROFILE, "id": "other"}}, None, "does not match"),
        ({"fast": {**PROFILE, "id": "fast", "note": "x"}}, None, "invalid profile"),
        ({"fast": {**PROFILE, "id": "fast", "price_in": -1}}, None, "invalid profile"),
        ({"fast": {**PROFILE, "id": "fast", "context": 0}}, None, "invalid profile"),
        ({"fast": {**PROFILE, "id": "fast", "tags": [1]}}, None, "invalid profile"),
    ],
)
def test_construction_refuses_bad_candidates(
    candidates: dict, fallback: str | None, match: str
) -> None:
    # Case 20.
    with pytest.raises(ValueError, match=match):
        NaceModelSelector(api_key="nace_sk_test", candidates=candidates, fallback=fallback)


def test_validate_candidate_map_accepts_a_full_set() -> None:
    candidates = {f"c{index}": None for index in range(255)}
    assert len(validate_candidate_map(candidates, fallback="c0")) == 255


def _walk(run):  # type: ignore[no-untyped-def]
    yield run
    for child in run.child_runs:
        yield from _walk(child)


_TEXT = st.text(alphabet="abc ", min_size=0, max_size=24)


def _messages() -> st.SearchStrategy:
    human = st.one_of(
        _TEXT.map(HumanMessage),
        st.tuples(_TEXT, st.booleans()).map(
            lambda pair: HumanMessage(
                content=[
                    {"type": "text", "text": pair[0]},
                    *(
                        [{"type": "image", "base64": "IMGSECRET", "mime_type": "image/png"}]
                        if pair[1]
                        else []
                    ),
                ]
            )
        ),
    )
    assistant = _TEXT.map(
        lambda text: AIMessage(
            content=text,
            tool_calls=[{"id": "c1", "name": "lookup", "args": {"q": "CALLSECRET"}}],
        )
    )
    tool = st.just(ToolMessage(content="TOOLSECRET", tool_call_id="c1"))
    system = _TEXT.map(SystemMessage)
    return st.lists(st.one_of(human, assistant, tool, system), max_size=8)


@given(_messages(), st.sampled_from(["conversation", "latest_human"]))
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_projection_properties(messages: list, context: str) -> None:
    projected = project_route_messages(messages, context=context)  # type: ignore[arg-type]
    assert len(projected) <= 1024
    assert sum(len(turn["content"]) for turn in projected) <= 24_000
    blob = json.dumps(projected)
    assert "TOOLSECRET" not in blob
    assert "IMGSECRET" not in blob
    assert "CALLSECRET" not in blob
    for turn in projected:
        assert turn["role"] in {"system", "user", "assistant"}
        assert isinstance(turn["content"], str)
    if context == "latest_human":
        assert [turn["role"] for turn in projected].count("user") <= 1


@given(
    st.lists(
        st.tuples(
            st.booleans(),
            st.booleans(),
            st.one_of(st.none(), st.integers(min_value=1, max_value=100)),
        ),
        min_size=0,
        max_size=6,
    ),
    st.booleans(),
    st.booleans(),
    st.floats(min_value=0, max_value=200, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_filter_properties(
    rows: list[tuple[bool, bool, int | None]],
    has_image: bool,
    has_tools: bool,
    estimated: float,
) -> None:
    bindings = []
    for index, (images, tools, context) in enumerate(rows):
        profile = None if context is None else {**PROFILE, "id": f"c{index}", "context": context}
        bindings.append(CandidateBinding(f"c{index}", profile=profile, images=images, tools=tools))
    chosen = filter_candidates(
        bindings, has_image=has_image, has_tools=has_tools, estimated_tokens=estimated
    )
    assert [binding.id for binding in chosen] == [
        binding.id
        for binding in bindings
        if not (has_image and not binding.images)
        and not (has_tools and not binding.tools)
        and not (binding.profile is not None and binding.profile["context"] < estimated)
    ]


@given(
    st.lists(st.from_regex(r"[a-z]{1,8}", fullmatch=True), min_size=1, max_size=4, unique=True),
    st.floats(min_value=-1, max_value=2, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_response_validation_properties(ids: list[str], probability: float) -> None:
    sent = tuple(ids)
    good = {
        candidate_id: (0.0 if candidate_id != ids[0] else min(max(probability, 0.0), 1.0))
        for candidate_id in ids
    }
    if 0 <= probability <= 1:
        parsed = response_is_valid(ids[0], good, sent)
        assert parsed is not None
        assert set(parsed) == set(sent)
    broken = {**good, ids[0]: probability}
    if not 0 <= probability <= 1:
        assert response_is_valid(ids[0], broken, sent) is None
    # "missing" can be drawn as an id. Use a token the strategy cannot produce.
    assert response_is_valid("ABSENT", good, sent) is None
    assert response_is_valid(ids[0], {ids[0]: 1.0}, sent) is None or len(sent) == 1


def test_media_and_file_image_blocks_count_as_images() -> None:
    image = {"type": "image", "base64": "AAAA", "mime_type": "image/png"}
    url = {"type": "image_url", "image_url": "https://example.test/a.png"}
    media = {"type": "media", "mime_type": "image/webp"}
    file_block = {"type": "file", "source": {"mime": "Image/jpeg"}}
    audio = {"type": "media", "mime_type": "audio/mpeg"}
    assert has_image_block([HumanMessage(content=[image])])
    assert has_image_block([HumanMessage(content=[url])])
    assert has_image_block([HumanMessage(content=[media])])
    assert has_image_block([HumanMessage(content=[file_block])])
    assert not has_image_block([HumanMessage(content=[audio])])


def test_empty_latest_human_does_not_send_an_earlier_turn() -> None:
    projected = project_route_messages(
        [HumanMessage("earlier secret"), HumanMessage("")],
        system_message=SystemMessage("rule"),
        context="latest_human",
    )
    assert projected == []
    assert "earlier secret" not in json.dumps(projected)


def test_profile_text_limits() -> None:
    base = {key: value for key, value in PROFILE.items() if key != "id"}
    with pytest.raises(ValueError, match="invalid profile"):
        validate_candidate_map({"fast": {**base, "provider": "p" * 129}}, fallback=None)
    with pytest.raises(ValueError, match="invalid profile"):
        validate_candidate_map({"fast": {**base, "tags": [""]}}, fallback=None)
    with pytest.raises(ValueError, match="invalid profile"):
        validate_candidate_map({"fast": {**base, "tags": ["t"] * 65}}, fallback=None)
    with pytest.raises(ValueError, match="invalid profile"):
        validate_candidate_map({"fast": {**base, "tags": ["t" * 129]}}, fallback=None)
    packed = {
        f"{index:0256d}": {
            "provider": "p" * 128,
            "price_in": 0.0,
            "price_out": 0.0,
            "context": 8,
            "tags": ["t" * 128] * 64,
        }
        for index in range(12)
    }
    assert len(validate_candidate_map(dict(list(packed.items())[:11]), fallback=None)) == 11
    with pytest.raises(ValueError, match="100000"):
        validate_candidate_map(packed, fallback=None)


def test_request_id_equal_to_the_api_key_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    key = "req_Secret12345"
    route = FakeRoute(replies=[route_ok("fast", {"fast": 0.7, "strong": 0.3}, request_id=key)])
    with caplog.at_level(logging.DEBUG):
        chosen = selector(route, api_key=key).select([HumanMessage("hello")])
    assert chosen["request_id"] is None
    assert key not in json.dumps(chosen)
    assert key not in caplog.text
    safe = FakeRoute(
        replies=[route_ok("fast", {"fast": 0.7, "strong": 0.3}, request_id="req_Abcdefgh")]
    )
    assert selector(safe).select([HumanMessage("hello")])["request_id"] == "req_Abcdefgh"


def _route_args() -> dict[str, object]:
    return {
        "priority": "balanced",
        "timeout": 0.01,
        "sent_ids": ("fast", "strong"),
        "candidate_ids": ("fast", "strong"),
        "fallback_id": "strong",
        "min_probability": None,
    }


def test_sync_error_after_the_deadline_is_timeout() -> None:
    class Late:
        def route(self, *_args: object, **_kwargs: object) -> None:
            time.sleep(0.05)
            raise httpx.ConnectError("down")

    selection = route_exchange(
        Late(),  # type: ignore[arg-type]
        [{"role": "user", "content": "hi"}],
        ["fast", "strong"],
        **_route_args(),  # type: ignore[arg-type]
    )
    assert selection["reason"] == "timeout"
    assert selection["model"] == "strong"
    assert "down" not in json.dumps(selection)


async def test_async_error_after_the_deadline_is_timeout() -> None:
    class Late:
        async def route(self, *_args: object, **_kwargs: object) -> None:
            deadline = time.perf_counter() + 0.05
            while time.perf_counter() < deadline:
                pass
            raise httpx.ConnectError("down")

    selection = await aroute_exchange(
        Late(),  # type: ignore[arg-type]
        [{"role": "user", "content": "hi"}],
        ["fast", "strong"],
        **_route_args(),  # type: ignore[arg-type]
    )
    assert selection["reason"] == "timeout"
    assert "down" not in json.dumps(selection)


def test_plain_word_request_id_is_dropped() -> None:
    reply = route_ok("fast", {"fast": 0.7, "strong": 0.3}, request_id="request")
    route = FakeRoute(replies=[reply])
    assert selector(route).select([HumanMessage("hello")])["request_id"] is None


def test_provider_native_image_blocks_count_as_images() -> None:
    bedrock = {"image": {"format": "png", "source": {"bytes": "AAAA"}}}
    native = {"type": "input_image", "image_url": "https://example.test/a.png"}
    assert has_image_block([HumanMessage(content=[bedrock])])
    assert has_image_block([HumanMessage(content=[native])])


def test_image_only_latest_human_sends_nothing() -> None:
    messages = [
        HumanMessage("earlier secret"),
        HumanMessage(content=[{"type": "image", "base64": "AAAA", "mime_type": "image/png"}]),
    ]
    assert project_route_messages(messages) == []
    assert project_route_messages(messages, context="latest_human") == []
    route = FakeRoute()
    chosen = selector(route).select(messages)
    assert chosen["reason"] == "no_input"
    assert route.requests == []
    kept = project_route_messages(
        [
            HumanMessage(content=[{"type": "image", "base64": "AAAA", "mime_type": "image/png"}]),
            HumanMessage("thanks"),
        ]
    )
    assert kept == [{"role": "user", "content": "thanks"}]


def test_budget_keeps_the_newer_assistant_turn() -> None:
    projected = project_route_messages(
        [HumanMessage("hi"), AIMessage("a" * 12_000), AIMessage("b" * 12_000)]
    )
    assert [turn["content"] for turn in projected] == ["hi", "b" * 12_000]


def test_profile_rejects_js_blank_text_and_huge_context() -> None:
    base = {key: value for key, value in PROFILE.items() if key != "id"}
    with pytest.raises(ValueError, match="invalid profile"):
        validate_candidate_map({"fast": {**base, "provider": "\ufeff"}}, fallback=None)
    with pytest.raises(ValueError, match="invalid profile"):
        validate_candidate_map({"fast": {**base, "tags": ["\ufeff"]}}, fallback=None)
    with pytest.raises(ValueError, match="invalid profile"):
        validate_candidate_map({"fast": {**base, "context": 2**53}}, fallback=None)
    assert len(validate_candidate_map({"fast": {**base, "context": 2**53 - 1}}, fallback=None)) == 1


def test_secrets_registry_ignores_duplicates() -> None:
    secret = "round4-secret-unique"
    assert secret not in _SDK_LOG_REDACTOR._secrets
    before = len(_SDK_LOG_REDACTOR._secrets)
    _SDK_LOG_REDACTOR.add_secrets([secret, secret])
    _SDK_LOG_REDACTOR.add_secrets([secret])
    assert _SDK_LOG_REDACTOR._secrets.count(secret) == 1
    assert len(_SDK_LOG_REDACTOR._secrets) == before + 1


def test_route_plan_repr_hides_the_conversation() -> None:
    plan = _RoutePlan(
        messages=({"role": "user", "content": "SECRET-TURN"},),
        candidates=("fast",),
        sent_ids=("fast",),
        candidate_ids=("fast",),
        priority="balanced",
        fallback_id=None,
        min_probability=None,
        timeout=3.0,
    )
    assert "SECRET-TURN" not in repr(plan)


def _late_body() -> object:
    return type(
        "LateBody",
        (),
        {
            "selected": "fast",
            "probabilities": {"fast": 0.7, "strong": 0.3},
            "request_id": "req_Abcdefgh",
        },
    )()


def test_sync_response_after_the_deadline_is_timeout() -> None:
    class Late:
        def route(self, *_args: object, **_kwargs: object) -> object:
            time.sleep(0.05)
            return _late_body()

    selection = route_exchange(
        Late(),  # type: ignore[arg-type]
        [{"role": "user", "content": "hi"}],
        ["fast", "strong"],
        **_route_args(),  # type: ignore[arg-type]
    )
    assert selection["reason"] == "timeout"
    assert selection["model"] == "strong"
    assert selection["selected"] is None


async def test_async_response_after_the_deadline_is_timeout() -> None:
    class Late:
        async def route(self, *_args: object, **_kwargs: object) -> object:
            deadline = time.perf_counter() + 0.05
            while time.perf_counter() < deadline:
                pass
            return _late_body()

    selection = await aroute_exchange(
        Late(),  # type: ignore[arg-type]
        [{"role": "user", "content": "hi"}],
        ["fast", "strong"],
        **_route_args(),  # type: ignore[arg-type]
    )
    assert selection["reason"] == "timeout"
    assert selection["model"] == "strong"
    assert selection["selected"] is None


def test_exchange_outcome_repr_hides_the_raw_selection() -> None:
    outcome = ExchangeOutcome(
        kind="response",
        priority="balanced",
        selected="SECRET-SELECTED",
        probabilities={"fast": "SECRET-PROB"},
    )
    text = repr(outcome)
    assert "SECRET-SELECTED" not in text
    assert "SECRET-PROB" not in text


def test_nested_images_count() -> None:
    image = {"type": "image", "base64": "AAAA", "mime_type": "image/png"}
    tool_result = {"type": "tool_result", "tool_use_id": "c1", "content": [image]}
    assert has_image_block([HumanMessage(content=[{"type": "text", "text": "see"}, tool_result])])
    assert has_image_block(
        [ToolMessage(content=[{"type": "tool_result", "content": [image]}], tool_call_id="c1")]
    )
    assert has_image_block([ToolMessage(content=[image], tool_call_id="c1")])

    def nest(depth: int) -> dict[str, object]:
        block: dict[str, object] = dict(image)
        for _ in range(depth):
            block = {"type": "tool_result", "content": [block]}
        return block

    assert has_image_block([HumanMessage(content=[nest(3)])])
    assert not has_image_block([HumanMessage(content=[nest(20)])])


def test_keyboard_interrupt_is_not_hidden_by_a_close_error() -> None:
    class Stream(httpx.SyncByteStream):
        def __iter__(self) -> object:
            raise KeyboardInterrupt

        def close(self) -> None:
            raise httpx.ReadError("closed")

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=Stream())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    chosen = NaceModelSelector(
        api_key="nace_sk_test",
        base_url="https://drex.test",
        candidates={"fast": None, "strong": None},
        fallback="strong",
        timeout=30,
        http_client=client,
    )
    with pytest.raises(KeyboardInterrupt):
        chosen.select([HumanMessage("hello")])


def _raise_timeout(request: httpx.Request) -> None:
    del request
    raise TimeoutError


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_hook_timeout_before_the_deadline_is_connection_error(asynchronous: bool) -> None:
    route = FakeRoute()
    chosen = NaceModelSelector(
        api_key="nace_sk_test",
        base_url="https://drex.test",
        candidates={"fast": None, "strong": None},
        fallback="strong",
        timeout=30,
        http_client=httpx.Client(
            transport=route.transport(), event_hooks={"request": [_raise_timeout]}
        ),
        http_async_client=httpx.AsyncClient(
            transport=route.async_transport(), event_hooks={"request": [_raise_timeout]}
        ),
    )
    messages = [HumanMessage("hello")]
    selection = await chosen.aselect(messages) if asynchronous else chosen.select(messages)
    assert selection["reason"] == "connection_error"
    assert selection["model"] == "strong"
    assert "TimeoutError" not in json.dumps(selection)
