"""A fake Drex server for unit tests: records requests, answers from the questions."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

Reply = tuple[int, dict[str, str], Any]


def answer_for(question: dict[str, Any]) -> dict[str, Any]:
    """A plausible answer for any question type."""
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": 0.9}
    if kind == "choice":
        labels = list(question["criteria"])
        rest = 0.2 / max(len(labels) - 1, 1)
        probs = {label: (0.8 if i == 0 else rest) for i, label in enumerate(labels)}
        return {"type": "choice", "choice": labels[0], "confidence": 0.8, "probabilities": probs}
    if kind == "score":
        levels = question["criteria"]
        probs = {str(i): (1.0 if i == 0 else 0.0) for i in range(len(levels))}
        return {
            "type": "score",
            "score": 0.0,
            "legend": levels,
            "probabilities": probs,
            "confidence": 0.9,
        }
    return {"type": kind, "value": 42}


def ok_body(
    body: dict[str, Any], overrides: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    answers = {name: answer_for(q) for name, q in body["questions"].items()}
    answers.update(overrides or {})
    return {
        "model": body.get("model") or "drex-v2.0",
        "answers": answers,
        "usage": {"input_tokens": 120, "image_tokens": 0, "output_tokens": 3},
        "evaluation_time_ms": 81.5,
        "request_id": "req_test",
    }


@dataclass
class FakeDrex:
    """`replies` are used first, in order; then every request gets a 200 from `ok_body`."""

    replies: list[Reply | Callable[[dict[str, Any]], Reply]] = field(default_factory=list)
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    delay: float = 0.0
    requests: list[httpx.Request] = field(default_factory=list)
    bodies: list[dict[str, Any]] = field(default_factory=list)
    in_flight: int = 0
    peak_in_flight: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        with self._lock:
            self.requests.append(request)
            self.bodies.append(body)
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            reply = self.replies.pop(0) if self.replies else None
        try:
            if self.delay:
                time.sleep(self.delay)
            if callable(reply):
                reply = reply(body)
            if reply is None:
                status, headers, payload = 200, {}, ok_body(body, self.overrides)
            else:
                status, headers, payload = reply
            headers = {"x-request-id": "req_test", **headers}
            return httpx.Response(status, json=payload, headers=headers)
        finally:
            with self._lock:
                self.in_flight -= 1

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    async def _async(self, request: httpx.Request) -> httpx.Response:
        return self(request)

    def async_transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._async)


def error(
    status: int,
    message: str,
    *,
    issues: list[dict[str, str]] | None = None,
    kind: str = "invalid_request_error",
    headers: dict[str, str] | None = None,
) -> Reply:
    payload: dict[str, Any] = {"error": {"type": kind, "message": message}, "request_id": "req_err"}
    if issues is not None:
        payload["error"]["issues"] = issues
    return status, headers or {}, payload
