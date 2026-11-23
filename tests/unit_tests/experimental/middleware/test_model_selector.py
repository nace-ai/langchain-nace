"""Agent tests for NaceModelSelectorMiddleware. Case 6 is JavaScript only."""

import asyncio
import json
import logging
import time
from typing import Any

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeCancelledError
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from langchain_nace.experimental.middleware import (
    ModelCandidate,
    NaceModelSelectorMiddleware,
)
from langchain_nace.selection import NaceModelSelector
from tests.agents import ScriptedModel, scripted, tool_call
from tests.fakes import FakeRoute, error, route_ok

CANARY = "CANARY route secret 9f3a"


class RouteAnswer(BaseModel):
    """Structured answer used to keep response_format on the model request."""

    answer: str


class Probe(AgentMiddleware):
    def __init__(self, name: str = "Probe") -> None:
        super().__init__()
        self.requests: list[Any] = []
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def wrap_model_call(self, request, handler):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return await handler(request)


def route_selector(route: FakeRoute, **kwargs: object) -> NaceModelSelector:
    return NaceModelSelector(
        api_key="nace_sk_test",
        base_url="https://drex.test",
        http_client=httpx.Client(transport=route.transport()),
        http_async_client=httpx.AsyncClient(transport=route.async_transport()),
        timeout=kwargs.pop("timeout", 3.0),  # type: ignore[arg-type]
    )


def middleware(route: FakeRoute, **kwargs: object) -> NaceModelSelectorMiddleware:
    fast = kwargs.pop("fast", scripted(AIMessage("from fast")))
    strong = kwargs.pop("strong", scripted(AIMessage("from strong")))
    options: dict[str, object] = {
        "candidates": {
            "fast": ModelCandidate(fast),
            "strong": ModelCandidate(strong),
        },
        "fallback": "strong",
        "selector": route_selector(route, timeout=kwargs.get("timeout", 3.0)),
    }
    options.update(kwargs)
    return NaceModelSelectorMiddleware(**options)  # type: ignore[arg-type]


def agent(route: FakeRoute, **kwargs: object):
    default = kwargs.pop("default", scripted(AIMessage("from default")))
    tools = kwargs.pop("tools", [])
    checkpointer = kwargs.pop("checkpointer", None)
    system_prompt = kwargs.pop("system_prompt", None)
    response_format = kwargs.pop("response_format", None)
    extra = kwargs.pop("middleware", [])
    mw = kwargs.pop("mw", None) or middleware(route, **kwargs)
    return create_agent(
        default,
        tools=tools,  # type: ignore[arg-type]
        system_prompt=system_prompt,  # type: ignore[arg-type]
        middleware=[mw, *extra],  # type: ignore[list-item]
        checkpointer=checkpointer,  # type: ignore[arg-type]
        response_format=response_format,  # type: ignore[arg-type]
    )


def invoke(route: FakeRoute, text: object = "What is 2+2?", **kwargs: object) -> dict:
    graph = agent(route, **kwargs)
    return graph.invoke({"messages": [HumanMessage(text)]})  # type: ignore[arg-type]


def test_valid_response_selects_the_bound_model() -> None:
    # Case 1.
    route = FakeRoute(replies=[route_ok("strong", {"fast": 0.2, "strong": 0.8})])
    result = invoke(route)
    assert result["messages"][-1].content == "from strong"
    selection = result["nace_model_selection"]
    assert selection["reason"] == "selected"
    assert selection["model"] == "strong"
    assert selection["selected"] == "strong"
    assert selection["probabilities"] == {"fast": 0.2, "strong": 0.8}
    assert selection["fallback"] is False
    assert "nace_model_route" not in result


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("reason", "fault", "reply"),
    [
        ("timeout", "timeout", None),
        ("connection_error", "connect", None),
        ("rate_limited", None, error(429, "slow", kind="rate_limit_error")),
        ("unavailable", None, error(500, "down")),
        ("unavailable", None, error(503, "down")),
        ("unavailable", None, error(504, "down")),
        ("unavailable", None, error(529, "down", kind="overloaded_error")),
        ("auth_error", None, error(401, "no")),
        ("auth_error", None, error(403, "no")),
        ("no_funds", None, error(402, "empty")),
        ("invalid_request", None, error(413, "big")),
        ("invalid_request", None, error(422, "bad")),
        (
            "invalid_response",
            None,
            (200, {}, {"selected": "nope", "probabilities": {"fast": 0.5, "strong": 0.5}}),
        ),
    ],
)
async def test_http_reason_uses_the_fallback_model(asynchronous, reason, fault, reply) -> None:
    # Case 2, route outcomes. The handler runs the fallback model.
    route = FakeRoute(fault=fault, replies=[reply] if reply is not None else [])
    graph = agent(route)
    payload = {"messages": [HumanMessage("hello")]}
    result = await graph.ainvoke(payload) if asynchronous else graph.invoke(payload)
    assert result["messages"][-1].content == "from strong"
    assert result["nace_model_selection"]["reason"] == reason
    assert result["nace_model_selection"]["model"] == "strong"
    assert len(route.requests) == 1


def test_low_probability_uses_fallback() -> None:
    # Case 2.
    route = FakeRoute(replies=[route_ok("fast", {"fast": 0.4, "strong": 0.6})])
    result = invoke(route, min_probability=0.6)
    assert result["messages"][-1].content == "from strong"
    assert result["nace_model_selection"]["reason"] == "low_probability"
    assert result["nace_model_selection"]["selected"] == "fast"
    assert result["nace_model_selection"]["model"] == "strong"


def test_probability_at_the_minimum_is_selected() -> None:
    route = FakeRoute(replies=[route_ok("fast", {"fast": 0.6, "strong": 0.4})])
    result = invoke(route, min_probability=0.6)
    assert result["nace_model_selection"]["reason"] == "selected"
    assert result["messages"][-1].content == "from fast"


def test_single_candidate_skips_the_route_call() -> None:
    # Case 2.
    route = FakeRoute()
    fast = scripted(AIMessage("only fast"))
    result = invoke(
        route,
        candidates={"fast": ModelCandidate(fast)},
        fallback=None,
        selector=route_selector(route),
    )
    assert result["messages"][-1].content == "only fast"
    assert result["nace_model_selection"]["reason"] == "single_candidate"
    assert result["nace_model_selection"]["model"] == "fast"
    assert route.requests == []


def test_no_input_keeps_the_agent_model() -> None:
    # Case 2.
    route = FakeRoute()
    result = invoke(route, text="")
    assert result["messages"][-1].content == "from default"
    assert result["nace_model_selection"]["reason"] == "no_input"
    assert result["nace_model_selection"]["model"] is None
    assert route.requests == []


def test_no_eligible_candidate_keeps_the_agent_model() -> None:
    # Case 2. Every candidate rejects images, so the fallback is ineligible too.
    route = FakeRoute()
    fast = scripted(AIMessage("from fast"))
    strong = scripted(AIMessage("from strong"))
    result = invoke(
        route,
        text=[
            {"type": "text", "text": "see"},
            {"type": "image", "base64": "AAAA", "mime_type": "image/png"},
        ],
        candidates={
            "fast": ModelCandidate(fast, images=False),
            "strong": ModelCandidate(strong, images=False),
        },
        fallback="strong",
        selector=route_selector(route),
    )
    assert result["messages"][-1].content == "from default"
    assert result["nace_model_selection"]["reason"] == "no_eligible_candidate"
    assert result["nace_model_selection"]["model"] is None
    assert route.requests == []


def test_sync_timeout_gives_the_handler_the_fallback() -> None:
    # Case 7.
    route = FakeRoute(delay=30)
    result = invoke(route, timeout=0.2)
    assert result["messages"][-1].content == "from strong"
    assert result["nace_model_selection"]["reason"] == "timeout"
    assert route.delivered == 0
    assert route.withheld[0]["selected"] == "fast"


async def test_async_timeout_discards_the_late_response() -> None:
    # Case 7.
    route = FakeRoute(delay=5)
    graph = agent(route, timeout=0.2)
    result = await graph.ainvoke({"messages": [HumanMessage("hello")]})
    assert result["messages"][-1].content == "from strong"
    assert result["nace_model_selection"]["reason"] == "timeout"
    assert route.delivered == 0
    assert route.in_flight == 0
    held = dict(result["nace_model_selection"])
    await asyncio.sleep(0.05)
    assert result["nace_model_selection"] == held


async def test_caller_cancellation_does_not_call_the_handler() -> None:
    # Case 8.
    route = FakeRoute(delay=5)
    probe = Probe()
    graph = agent(route, timeout=30, middleware=[probe])
    task = asyncio.create_task(graph.ainvoke({"messages": [HumanMessage("hello")]}))
    for _ in range(100):
        if route.requests:
            break
        await asyncio.sleep(0.01)
    assert route.requests
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert probe.requests == []
    assert route.cancelled >= 1
    assert route.in_flight == 0
    assert route.delivered == 0


def test_handler_request_changes_only_the_model() -> None:
    # Case 11.
    route = FakeRoute(replies=[route_ok("fast", {"fast": 0.9, "strong": 0.1})])
    outer, inner = Probe("outer"), Probe("inner")
    graph = agent(
        route,
        system_prompt="Be brief.",
        response_format=RouteAnswer,
        middleware=[outer, inner],
    )
    # The outer probe is listed after the selector in `middleware=[mw, *extra]`,
    # so build the order explicitly.
    fast = scripted(tool_call("RouteAnswer", {"answer": "ok"}, "s1"))
    strong = scripted(AIMessage("from strong"))
    default = scripted(AIMessage("from default"))
    mw = middleware(route, fast=fast, strong=strong)
    graph = create_agent(
        default,
        tools=[],
        system_prompt="Be brief.",
        response_format=RouteAnswer,
        middleware=[outer, mw, inner],
    )
    result = graph.invoke({"messages": [HumanMessage("hello")]})
    assert outer.requests
    assert inner.requests
    original, sent = outer.requests[0], inner.requests[0]
    assert sent.messages is original.messages
    assert all(left is right for left, right in zip(sent.messages, original.messages, strict=True))
    assert sent.system_message is original.system_message
    assert sent.tools is original.tools
    assert sent.tool_choice is original.tool_choice
    assert sent.response_format is original.response_format
    assert sent.model_settings is original.model_settings
    assert sent.state is original.state
    assert sent.runtime is original.runtime
    assert sent.model is fast
    assert sent is not original
    assert result["structured_response"] == RouteAnswer(answer="ok")


def test_no_input_passes_the_same_request_object() -> None:
    route = FakeRoute()
    outer, inner = Probe("outer"), Probe("inner")
    mw = middleware(route)
    graph = create_agent(
        scripted(AIMessage("from default")),
        tools=[],
        middleware=[outer, mw, inner],
    )
    graph.invoke({"messages": [HumanMessage("")]})
    assert inner.requests[0] is outer.requests[0]


def test_streaming_yields_selected_model_tokens_in_order() -> None:
    # Case 12.
    route = FakeRoute(replies=[route_ok("fast", {"fast": 0.8, "strong": 0.2})])
    fast = scripted(AIMessage("alpha beta"))
    graph = agent(route, fast=fast, strong=scripted(AIMessage("gamma delta")))

    def tokens(chunks: list) -> str:
        parts = []
        for chunk in chunks:
            message = chunk[0] if isinstance(chunk, tuple) else chunk
            if isinstance(message, AIMessageChunk) and isinstance(message.content, str):
                parts.append(message.content)
        return "".join(parts)

    streamed = list(graph.stream({"messages": [HumanMessage("hi")]}, stream_mode="messages"))
    assert tokens(streamed) == "alpha beta"


async def test_async_streaming_yields_selected_model_tokens_in_order() -> None:
    # Case 12.
    route = FakeRoute(replies=[route_ok("fast", {"fast": 0.8, "strong": 0.2})])
    graph = agent(route, fast=scripted(AIMessage("alpha beta")))
    parts: list[str] = []
    async for chunk in graph.astream({"messages": [HumanMessage("hi")]}, stream_mode="messages"):
        message = chunk[0] if isinstance(chunk, tuple) else chunk
        if isinstance(message, AIMessageChunk) and isinstance(message.content, str):
            parts.append(message.content)
    assert "".join(parts) == "alpha beta"


def test_image_and_tool_limits_are_not_sent() -> None:
    # Case 13.
    from langchain_core.tools import tool

    @tool
    def lookup(q: str) -> str:
        """Look up a fact."""
        return "found"

    route = FakeRoute()
    fast = scripted(AIMessage("from fast"))
    strong = scripted(AIMessage("from strong"))
    blind = scripted(AIMessage("from blind"))
    plain = scripted(AIMessage("from plain"))
    invoke(
        route,
        tools=[lookup],
        text=[
            {"type": "text", "text": "see this"},
            {"type": "image", "base64": "AAAA", "mime_type": "image/png"},
        ],
        candidates={
            "fast": ModelCandidate(fast),
            "strong": ModelCandidate(strong),
            "blind": ModelCandidate(blind, images=False),
            "plain": ModelCandidate(plain, tools=False),
        },
        fallback="fast",
        selector=route_selector(route),
    )
    sent = []
    for item in route.bodies[0]["candidates"]:
        sent.append(item if isinstance(item, str) else item["id"])
    assert sent == ["fast", "strong"]


def test_conversation_with_media_and_tools_is_what_is_sent() -> None:
    # Case 10, through the agent.
    from langchain_core.tools import tool

    @tool
    def lookup(q: str) -> str:
        """Look up a fact."""
        return "found"

    route = FakeRoute()
    graph = agent(
        route,
        tools=[lookup],
        system_prompt="SYS-RULE",
        fast=scripted(AIMessage("from fast")),
    )
    graph.invoke(
        {
            "messages": [
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
            ]
        }
    )
    body = route.bodies[-1]
    blob = json.dumps(body)
    assert "IMGSECRET" not in blob
    assert "CALLSECRET" not in blob
    assert "TOOLSECRET" not in blob
    assert "[image omitted]" in blob
    assert {"role": "system", "content": "SYS-RULE"} in body["messages"]
    assert {"role": "assistant", "content": "checking"} in body["messages"]
    assert all(message["role"] != "tool" for message in body["messages"])


def _lookup():
    from langchain_core.tools import tool

    @tool
    def lookup(q: str) -> str:
        """Look something up."""
        return "found"

    return lookup


def test_run_mode_uses_one_route_call_for_three_model_calls() -> None:
    # Case 14.
    route = FakeRoute()
    fast = scripted(
        tool_call("lookup", {"q": "a"}, "1"),
        tool_call("lookup", {"q": "b"}, "2"),
        AIMessage("done"),
    )
    result = invoke(route, tools=[_lookup()], fast=fast)
    assert result["messages"][-1].content == "done"
    assert len(route.requests) == 1
    assert result["nace_model_selection"]["model"] == "fast"


def test_resumed_run_does_not_route_again() -> None:
    # Case 14.
    from langchain_core.tools import tool

    @tool
    def lookup(q: str) -> str:
        """Look something up."""
        return str(interrupt("wait"))

    route = FakeRoute()
    fast = scripted(
        tool_call("lookup", {"q": "a"}, "1"),
        tool_call("lookup", {"q": "b"}, "2"),
        AIMessage("done"),
    )
    graph = agent(route, tools=[lookup], fast=fast, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "run-1"}}
    paused = graph.invoke({"messages": [HumanMessage("go")]}, config)
    assert "__interrupt__" in paused
    assert len(route.requests) == 1
    paused = graph.invoke(Command(resume="one"), config)
    assert "__interrupt__" in paused
    assert len(route.requests) == 1
    finished = graph.invoke(Command(resume="two"), config)
    assert finished["messages"][-1].content == "done"
    assert len(route.requests) == 1
    assert finished["nace_model_selection"]["model"] == "fast"


def test_model_call_mode_routes_every_model_call() -> None:
    # Case 15.
    route = FakeRoute()
    fast = scripted(
        tool_call("lookup", {"q": "a"}, "1"),
        tool_call("lookup", {"q": "b"}, "2"),
        AIMessage("done"),
    )
    result = invoke(route, tools=[_lookup()], fast=fast, select_on="model_call")
    assert result["messages"][-1].content == "done"
    assert len(route.requests) == 3


def test_second_run_on_the_same_thread_selects_again() -> None:
    # Case 16.
    route = FakeRoute(
        replies=[
            route_ok("fast", {"fast": 0.8, "strong": 0.2}),
            route_ok("strong", {"fast": 0.2, "strong": 0.8}),
        ]
    )
    graph = agent(
        route,
        fast=scripted(AIMessage("from fast"), AIMessage("fast again")),
        strong=scripted(AIMessage("from strong")),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "thread-1"}}
    first = graph.invoke({"messages": [HumanMessage("one")]}, config)
    second = graph.invoke({"messages": [HumanMessage("two")]}, config)
    assert first["messages"][-1].content == "from fast"
    assert second["messages"][-1].content == "from strong"
    assert len(route.requests) == 2
    assert second["nace_model_selection"]["model"] == "strong"


def test_canary_is_absent_from_the_stored_selection(caplog: pytest.LogCaptureFixture) -> None:
    # Case 17.
    route = FakeRoute(replies=[error(422, CANARY)])
    with caplog.at_level(logging.DEBUG):
        result = invoke(route, text=CANARY)
    blob = json.dumps(result["nace_model_selection"])
    assert CANARY not in blob
    assert CANARY not in caplog.text
    assert "nace_sk_test" not in caplog.text
    assert "nace_sk_test" not in blob


@pytest.mark.parametrize(
    ("candidates", "fallback", "match"),
    [
        ({}, None, "at least one"),
        ({f"c{i}": ModelCandidate(scripted(AIMessage("x"))) for i in range(256)}, None, "255"),
        ({"": ModelCandidate(scripted(AIMessage("x")))}, None, "blank"),
        ({"x" * 257: ModelCandidate(scripted(AIMessage("x")))}, None, "256"),
        ({"fast": ModelCandidate(scripted(AIMessage("x")))}, "missing", "fallback"),
        (
            {"fast": ModelCandidate(scripted(AIMessage("x")), profile={"provider": ""})},
            None,
            "invalid profile",
        ),
    ],
)
def test_middleware_construction_refuses_bad_candidates(candidates, fallback, match) -> None:
    # Case 20.
    with pytest.raises(ValueError, match=match):
        NaceModelSelectorMiddleware(candidates=candidates, fallback=fallback)


class Boom(ScriptedModel):
    def __init__(self) -> None:
        super().__init__(messages=iter([AIMessage("unused")]))
        object.__setattr__(self, "calls", 0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
        object.__setattr__(self, "calls", self.calls + 1)
        raise RuntimeError("model failed")


def _request(model: Any, messages: list, **kwargs: object) -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=messages,
        tools=kwargs.get("tools", []),  # type: ignore[arg-type]
        response_format=kwargs.get("response_format"),  # type: ignore[arg-type]
        state=kwargs.get("state", {"messages": []}),  # type: ignore[arg-type]
    )


def test_tool_result_image_drops_a_stored_blind_candidate() -> None:
    route = FakeRoute(replies=[route_ok("blind", {"blind": 0.8, "vision": 0.2})])
    blind = scripted(AIMessage("from blind"))
    vision = scripted(AIMessage("from vision"))
    mw = middleware(
        route,
        candidates={
            "blind": ModelCandidate(blind, images=False),
            "vision": ModelCandidate(vision),
        },
        fallback="vision",
        selector=route_selector(route),
    )
    seen: list[Any] = []

    def handler(request: ModelRequest) -> AIMessage:
        seen.append(request.model)
        return AIMessage("ok")

    opening = _request(scripted(AIMessage("default")), [HumanMessage("look")])
    first = mw.wrap_model_call(opening, handler)
    stored = first.command.update["nace_model_selection"]  # type: ignore[index]
    assert stored["model"] == "blind"
    assert seen == [blind]
    second = _request(
        scripted(AIMessage("default")),
        [
            HumanMessage("look"),
            AIMessage(content="", tool_calls=[{"id": "c1", "name": "look", "args": {}}]),
            ToolMessage(
                content=[{"type": "image", "base64": "AAAA", "mime_type": "image/png"}],
                tool_call_id="c1",
            ),
        ],
        state={"messages": [], "nace_model_selection": stored},
    )
    mw.wrap_model_call(second, handler)
    assert seen[1] is vision
    assert seen[1] is not blind


def test_response_format_counts_as_tools_for_the_fallback() -> None:
    route = FakeRoute()
    plain = scripted(AIMessage("from plain"))
    toolish = scripted(AIMessage("from toolish"))
    mw = middleware(
        route,
        candidates={
            "plain": ModelCandidate(plain, tools=False),
            "toolish": ModelCandidate(toolish),
        },
        fallback="plain",
        selector=route_selector(route),
    )
    seen: list[Any] = []

    def handler(request: ModelRequest) -> AIMessage:
        seen.append(request.model)
        return AIMessage("ok")

    result = mw.wrap_model_call(
        _request(
            scripted(AIMessage("default")),
            [HumanMessage("hi")],
            response_format=RouteAnswer,
        ),
        handler,
    )
    assert seen == [toolish]
    assert result.command.update["nace_model_selection"]["reason"] == "single_candidate"  # type: ignore[index]
    assert result.command.update["nace_model_selection"]["model"] == "toolish"  # type: ignore[index]
    assert route.requests == []


def test_status_hook_uses_the_fallback_once(caplog: pytest.LogCaptureFixture) -> None:
    def raise_status(response: httpx.Response) -> None:
        response.raise_for_status()

    route = FakeRoute(replies=[error(429, CANARY, kind="rate_limit_error")])
    hooked = NaceModelSelector(
        api_key="nace_sk_test",
        base_url="https://drex.test",
        http_client=httpx.Client(
            transport=route.transport(), event_hooks={"response": [raise_status]}
        ),
        http_async_client=httpx.AsyncClient(
            transport=route.async_transport(), event_hooks={"response": [raise_status]}
        ),
    )
    probe = Probe("hook")
    with caplog.at_level(logging.DEBUG):
        result = invoke(route, selector=hooked, middleware=[probe])
    assert result["nace_model_selection"]["reason"] == "rate_limited"
    assert result["messages"][-1].content == "from strong"
    assert len(probe.requests) == 1
    assert len(route.requests) == 1
    assert CANARY not in caplog.text


async def _invoke(graph: Any, asynchronous: bool) -> None:
    payload = {"messages": [HumanMessage("hello")]}
    if asynchronous:
        await graph.ainvoke(payload)
    else:
        graph.invoke(payload)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_handler_error_is_not_retried(asynchronous: bool) -> None:
    route = FakeRoute(replies=[route_ok("fast", {"fast": 0.9, "strong": 0.1})])
    fast, strong = Boom(), Boom()
    graph = agent(route, fast=fast, strong=strong)
    with pytest.raises(RuntimeError, match="model failed"):
        await _invoke(graph, asynchronous)
    assert fast.calls == 1
    assert strong.calls == 0


async def test_cancellation_same_step_as_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Both:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            raise TimeoutError

    monkeypatch.setattr(asyncio, "timeout", lambda _delay: _Both())
    route = FakeRoute()
    probe = Probe("cancel")
    graph = agent(route, timeout=30, middleware=[probe])
    with pytest.raises(NodeCancelledError) as caught:
        await graph.ainvoke({"messages": [HumanMessage("hello")]})
    assert isinstance(caught.value.__cause__, asyncio.CancelledError)
    assert probe.requests == []


def test_before_agent_does_not_call_the_network() -> None:
    route = FakeRoute()
    mw = middleware(route)
    assert mw.before_agent({"messages": []}, None) == {"nace_model_selection": None}  # type: ignore[arg-type]
    assert route.requests == []


def test_candidate_repr_hides_the_model() -> None:
    candidate = ModelCandidate("provider:SECRET-MODEL")
    assert "SECRET-MODEL" not in repr(candidate)


def test_filtered_candidate_uses_the_model_drex_selected() -> None:
    route = FakeRoute(replies=[route_ok("vision-b", {"vision-a": 0.2, "vision-b": 0.8})])
    vision_a = scripted(AIMessage("from a"))
    vision_b = scripted(AIMessage("from b"))
    blind = scripted(AIMessage("from blind"))
    mw = middleware(
        route,
        candidates={
            "blind": ModelCandidate(blind, images=False),
            "vision-a": ModelCandidate(vision_a),
            "vision-b": ModelCandidate(vision_b),
        },
        fallback="vision-a",
        selector=route_selector(route),
    )
    seen: list[Any] = []

    def handler(request: ModelRequest) -> AIMessage:
        seen.append(request.model)
        return AIMessage("ok")

    picture = HumanMessage(
        content=[
            {"type": "text", "text": "see"},
            {"type": "image", "base64": "AAAA", "mime_type": "image/png"},
        ]
    )
    result = mw.wrap_model_call(_request(scripted(AIMessage("own")), [picture]), handler)
    stored = result.command.update["nace_model_selection"]  # type: ignore[index]
    assert seen == [vision_b]
    assert stored["model"] == "vision-b"
    assert stored["reason"] == "selected"


def test_structured_output_429_keeps_the_agent_model() -> None:
    route = FakeRoute(replies=[error(429, "slow", kind="rate_limit_error")])
    own = scripted(AIMessage("from own"))
    fast = scripted(AIMessage("from fast"))
    strong = scripted(AIMessage("from strong"))
    plain = scripted(AIMessage("from plain"))
    mw = middleware(
        route,
        candidates={
            "fast": ModelCandidate(fast),
            "strong": ModelCandidate(strong),
            "plain": ModelCandidate(plain, tools=False),
        },
        fallback="plain",
        selector=route_selector(route),
    )
    seen: list[Any] = []

    def handler(request: ModelRequest) -> AIMessage:
        seen.append(request.model)
        return AIMessage("ok")

    result = mw.wrap_model_call(
        _request(own, [HumanMessage("hi")], response_format=RouteAnswer),
        handler,
    )
    stored = result.command.update["nace_model_selection"]  # type: ignore[index]
    assert seen == [own]
    assert stored["model"] is None
    assert stored["reason"] == "rate_limited"


def test_malformed_stored_selection_selects_again() -> None:
    route = FakeRoute()
    own = scripted(AIMessage("own"))
    fast = scripted(AIMessage("from fast"))
    strong = scripted(AIMessage("from strong"))
    mw = middleware(route, fast=fast, strong=strong, selector=route_selector(route))
    seen: list[Any] = []

    def handler(request: ModelRequest) -> AIMessage:
        seen.append(request.model)
        return AIMessage("ok")

    shapes: list[object] = [
        {"reason": "selected"},
        {"model": "missing"},
        {"model": 1},
        ["fast"],
    ]
    for shape in shapes:
        mw.wrap_model_call(
            _request(
                own,
                [HumanMessage("hi")],
                state={"messages": [], "nace_model_selection": shape},
            ),
            handler,
        )
    assert seen == [fast, fast, fast, fast]
    incomplete = {"model": None}
    mw.wrap_model_call(
        _request(
            own,
            [HumanMessage("hi")],
            state={"messages": [], "nace_model_selection": incomplete},
        ),
        handler,
    )
    assert seen[-1] is fast
    written = mw.wrap_model_call(
        _request(
            own,
            [HumanMessage("again")],
            state={
                "messages": [],
                "nace_model_selection": _complete(model=None, reason="no_input"),
            },
        ),
        handler,
    ).command.update["nace_model_selection"]  # type: ignore[index]
    assert seen[-1] is own
    assert written is not None
    assert written["model"] is None


def _complete(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "model": "fast",
        "selected": "fast",
        "probabilities": {"fast": 1.0},
        "fallback": False,
        "reason": "selected",
        "priority": "balanced",
        "candidates": ["fast", "strong"],
        "request_id": "req_Abcdefgh",
        "latency_ms": 3,
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    "bad",
    [
        {"request_id": "plaintext"},
        {"reason": "nope"},
        {"fallback": "yes"},
        {"probabilities": {"fast": 2}},
        {"selected": "other"},
        {"candidates": "fast"},
    ],
)
def test_ill_formed_stored_record_is_not_written_back(bad: dict[str, object]) -> None:
    route = FakeRoute(replies=[route_ok("strong", {"fast": 0.2, "strong": 0.8})])
    own = scripted(AIMessage("own"))
    fast = scripted(AIMessage("from fast"))
    strong = scripted(AIMessage("from strong"))
    mw = middleware(route, fast=fast, strong=strong, selector=route_selector(route))
    original = _complete(**bad)
    seen: list[Any] = []

    def handler(request: ModelRequest) -> AIMessage:
        seen.append(request.model)
        return AIMessage("ok")

    result = mw.wrap_model_call(
        _request(
            own,
            [HumanMessage("hi")],
            state={"messages": [], "nace_model_selection": original},
        ),
        handler,
    )
    stored = result.command.update["nace_model_selection"]  # type: ignore[index]
    assert stored is not original
    assert seen == [strong]
    assert stored["model"] == "strong"
    assert stored["request_id"] != "plaintext"


def test_stored_record_does_not_keep_extra_keys() -> None:
    route = FakeRoute()
    own = scripted(AIMessage("own"))
    fast = scripted(AIMessage("from fast"))
    mw = middleware(route, fast=fast, selector=route_selector(route))
    original = _complete(note="CANARY")
    seen: list[Any] = []

    def handler(request: ModelRequest) -> AIMessage:
        seen.append(request.model)
        return AIMessage("ok")

    result = mw.wrap_model_call(
        _request(
            own,
            [HumanMessage("hi")],
            state={"messages": [], "nace_model_selection": original},
        ),
        handler,
    )
    stored = result.command.update["nace_model_selection"]  # type: ignore[index]
    assert seen == [fast]
    assert route.requests == []
    assert "CANARY" not in json.dumps(stored)
    assert stored is not original


def test_nested_tool_result_image_drops_a_blind_candidate() -> None:
    route = FakeRoute(replies=[route_ok("blind", {"blind": 0.8, "vision": 0.2})])
    blind = scripted(AIMessage("from blind"))
    vision = scripted(AIMessage("from vision"))
    mw = middleware(
        route,
        candidates={
            "blind": ModelCandidate(blind, images=False),
            "vision": ModelCandidate(vision),
        },
        fallback="vision",
        selector=route_selector(route),
    )
    seen: list[Any] = []

    def handler(request: ModelRequest) -> AIMessage:
        seen.append(request.model)
        return AIMessage("ok")

    picture = HumanMessage(
        content=[
            {"type": "text", "text": "see"},
            {
                "type": "tool_result",
                "tool_use_id": "c1",
                "content": [{"type": "image", "base64": "AAAA", "mime_type": "image/png"}],
            },
        ]
    )
    result = mw.wrap_model_call(_request(scripted(AIMessage("own")), [picture]), handler)
    stored = result.command.update["nace_model_selection"]  # type: ignore[index]
    assert seen == [vision]
    assert stored["model"] == "vision"
    assert stored["reason"] == "single_candidate"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_closed_client_is_connection_error(asynchronous: bool) -> None:
    sync_client = httpx.Client()
    async_client = httpx.AsyncClient()
    sync_client.close()
    await async_client.aclose()
    selector = NaceModelSelector(
        api_key="nace_sk_test",
        base_url="https://drex.test",
        http_client=sync_client,
        http_async_client=async_client,
    )
    probe = Probe("closed")
    graph = agent(FakeRoute(), selector=selector, middleware=[probe])
    payload = {"messages": [HumanMessage("hello")]}
    result = await graph.ainvoke(payload) if asynchronous else graph.invoke(payload)
    assert result["nace_model_selection"]["reason"] == "connection_error"
    assert len(probe.requests) == 1


async def test_cancel_during_read_when_close_raises_read_error() -> None:
    started = asyncio.Event()

    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self) -> Any:
            started.set()
            await asyncio.sleep(30)
            yield b"{}"

        async def aclose(self) -> None:
            raise httpx.ReadError("closed during read")

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=_Stream())

    selector = NaceModelSelector(
        api_key="nace_sk_test",
        base_url="https://drex.test",
        timeout=30,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    probe = Probe("read-cancel")
    graph = agent(FakeRoute(), selector=selector, timeout=30, middleware=[probe])
    task = asyncio.create_task(graph.ainvoke({"messages": [HumanMessage("hello")]}))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises((asyncio.CancelledError, NodeCancelledError)) as caught:
        await task
    if isinstance(caught.value, NodeCancelledError):
        assert isinstance(caught.value.__cause__, asyncio.CancelledError)
    assert probe.requests == []


async def test_cancel_after_a_successful_route_skips_the_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Pending:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()

        def expired(self) -> bool:
            return False

    monkeypatch.setattr(asyncio, "timeout", lambda _delay: _Pending())
    route = FakeRoute()
    probe = Probe("after-return")
    graph = agent(route, timeout=30, middleware=[probe])
    with pytest.raises(NodeCancelledError) as caught:
        await graph.ainvoke({"messages": [HumanMessage("hello")]})
    assert isinstance(caught.value.__cause__, asyncio.CancelledError)
    assert probe.requests == []


async def test_deadline_close_error_uses_the_fallback_once() -> None:
    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self) -> Any:
            await asyncio.sleep(30)
            yield b""

        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, stream=_Stream())

    selector = NaceModelSelector(
        api_key="nace_sk_test",
        base_url="https://drex.test",
        timeout=0.05,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    probe = Probe("deadline-close")
    graph = agent(FakeRoute(), selector=selector, timeout=0.05, middleware=[probe])
    result = await graph.ainvoke({"messages": [HumanMessage("hello")]})
    assert result["nace_model_selection"]["reason"] == "timeout"
    assert result["nace_model_selection"]["model"] == "strong"
    assert result["messages"][-1].content == "from strong"
    assert len(probe.requests) == 1
    assert "close failed" not in json.dumps(result["nace_model_selection"])


async def test_caller_cancel_and_late_response_skips_the_handler() -> None:
    started = asyncio.Event()

    class _Body:
        def __init__(self) -> None:
            self.selected = "fast"
            self.probabilities = {"fast": 1.0, "strong": 0.0}
            self.request_id = "req_Abcdefgh"

    class _Late:
        async def route(self, *_args: object, **_kwargs: object) -> _Body:
            mark = time.perf_counter()
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                while time.perf_counter() < mark + 0.2:
                    pass
                return _Body()
            return _Body()

    selector = route_selector(FakeRoute(), timeout=0.2)
    selector._aclient = _Late()  # type: ignore[assignment]
    probe = Probe("late-cancel")
    graph = agent(FakeRoute(), selector=selector, timeout=0.2, middleware=[probe])
    task = asyncio.create_task(graph.ainvoke({"messages": [HumanMessage("hello")]}))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises((asyncio.CancelledError, NodeCancelledError)) as caught:
        await task
    if isinstance(caught.value, NodeCancelledError):
        assert isinstance(caught.value.__cause__, asyncio.CancelledError)
    assert probe.requests == []
