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


# --- NDI (documents, workspaces, search) -----------------------------------------


NDI_JOB_ID = "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d"
NDI_WS_ID = "2b3c4d5e-6f7a-4b8c-9d0e-1f2a3b4c5d6e"
NDI_FILE_ID = "3c4d5e6f-7a8b-4c9d-0e1f-2a3b4c5d6e7f"
NDI_UPLOAD_URL = "https://upload.ndi.test/uploads"


def _evidence() -> dict[str, Any]:
    return {
        "source_path": "invoice.html",
        "source_file_id": NDI_FILE_ID,
        "page": 1,
        "component": None,
        "quote": "Total due: USD 400.00",
        "relevance": "States the total due.",
    }


def fact_output() -> dict[str, Any]:
    """A fact-shaped search output: the answer fields at the top level."""
    return {
        "answer": "The total due is USD 400.00.",
        "confidence": "high",
        "evidences": [_evidence()],
    }


def agent_output() -> dict[str, Any]:
    """An agent-shaped search output: the same answer fields nested under `answer`."""
    return {
        "route": "agent",
        "answer": {
            "answer": "The total due is USD 400.00.",
            "confidence": "medium",
            "evidences": [_evidence()],
        },
    }


@dataclass
class FakeNdi:
    """A fake NDI document service: records each request, routes on method and path.

    `replies` overrides a route's default answer, in order: append to
    `replies["parse"]` (or "job", "job_delete", "search", ...) using the `Reply`
    tuples of `FakeDrex`, or a callable from the parsed body to one.
    """

    markdown: str = "# Invoice INV-2026-0042\n\nTotal due: USD 400.00"
    full_markdown: str = (
        "# Invoice INV-2026-0042\n\nSeller: Example Supplies Ltd\n\nTotal due: USD 400.00"
    )
    chunks: list[dict[str, Any]] = field(default_factory=list)
    disclosures: list[dict[str, Any]] = field(default_factory=list)
    content_url: str | None = None
    output: dict[str, Any] = field(default_factory=fact_output)
    replies: dict[str, list[Reply | Callable[[dict[str, Any]], Reply]]] = field(
        default_factory=dict
    )
    requests: list[httpx.Request] = field(default_factory=list)
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def queue(self, route: str, reply: Reply | Callable[[dict[str, Any]], Reply]) -> None:
        """Add a reply override for `route`, used (in order) before the default."""
        self.replies.setdefault(route, []).append(reply)

    def parse_job(self) -> dict[str, Any]:
        """The default finished parse job, built from the current attributes."""
        return {
            "job_id": NDI_JOB_ID,
            "kind": "parse",
            "status": "succeeded",
            "result": {
                "document": {
                    "markdown": self.markdown,
                    "page_count": 2,
                    "chunks": self.chunks,
                    "content_url": self.content_url,
                },
                "disclosures": self.disclosures,
            },
        }

    def extract_job(self) -> dict[str, Any]:
        """The default finished extract job."""
        return {
            "job_id": NDI_JOB_ID,
            "kind": "extract",
            "status": "succeeded",
            "result": {
                "data": {"invoice_number": "INV-2026-0042", "total": 400.0},
                "fields": [
                    {
                        "path": "/invoice_number",
                        "value": "INV-2026-0042",
                        "status": "found",
                        "confidence": 0.97,
                        "citations": [
                            {"location": {"page": 1}, "source_text": "Invoice INV-2026-0042"}
                        ],
                    },
                    {
                        "path": "/total",
                        "value": 400.0,
                        "status": "found",
                        "confidence": 0.89,
                        "citations": [
                            {"location": {"page": 1}, "source_text": "Total due: USD 400.00"}
                        ],
                    },
                    {"path": "/po_number", "value": None, "status": "not_found",
                     "confidence": None, "citations": []},
                ],
            },
        }

    def search_job(self, output: dict[str, Any] | None = None) -> dict[str, Any]:
        """A finished automatic-search job with `output` (else the current one)."""
        return {
            "job_id": NDI_JOB_ID,
            "kind": "automatic_search",
            "status": "succeeded",
            "result": {"output": self.output if output is None else output},
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = {}
        if request.content:
            try:
                parsed = json.loads(request.content)
                if isinstance(parsed, dict):
                    body = parsed
            except ValueError:
                body = {}
        self.requests.append(request)
        self.bodies.append(body)
        route = self._route(request.method, request.url.path)
        queue = self.replies.get(route)
        reply = queue.pop(0) if queue else self._default(route)
        if callable(reply):
            reply = reply(body)
        status, headers, payload = reply
        base_headers = {"x-request-id": "req_test", **headers}
        if payload is None:
            return httpx.Response(status, headers=base_headers)
        if isinstance(payload, str):
            return httpx.Response(
                status,
                text=payload,
                headers={**base_headers, "content-type": "text/markdown; charset=utf-8"},
            )
        return httpx.Response(status, json=payload, headers=base_headers)

    def _route(self, method: str, path: str) -> str:
        parts = [part for part in path.split("/") if part]
        if parts[:3] == ["v1", "documents", "jobs"] and len(parts) >= 4:
            if len(parts) == 4 and method == "GET":
                return "job"
            if len(parts) == 4 and method == "DELETE":
                return "job_delete"
            if method == "GET":
                return "job_file"
        if method == "POST" and parts == ["v1", "documents", "upload-grants"]:
            return "grant"
        if method == "POST" and parts == ["uploads"]:
            return "upload"
        if method == "POST" and parts == ["v1", "documents", "parse"]:
            return "parse"
        if method == "POST" and parts == ["v1", "documents", "extract"]:
            return "extract"
        if (
            method == "POST"
            and parts[:2] == ["v1", "workspaces"]
            and parts[-1] == "automatic-search"
        ):
            return "search"
        if (
            method == "DELETE"
            and parts[:3] == ["v1", "documents", "workspaces"]
            and parts[4:5] == ["files"]
            and len(parts) == 6
        ):
            return "file_delete"
        return "unknown"

    def _default(self, route: str) -> Reply:
        if route == "grant":
            return 200, {}, {
                "workspace_id": NDI_WS_ID,
                "upload_url": NDI_UPLOAD_URL,
                "token": "upl_FAKE_FOR_TEST",
                "expires_at": "2026-11-20T12:00:00Z",
            }
        if route == "upload":
            return 200, {}, {
                "job_id": NDI_JOB_ID,
                "kind": "file_upload",
                "status": "succeeded",
                "result": {"file": {"workspace_id": NDI_WS_ID, "file_id": NDI_FILE_ID,
                                    "path": "invoice.html"}},
            }
        if route == "parse":
            return 200, {}, self.parse_job()
        if route == "extract":
            return 200, {}, self.extract_job()
        if route == "search":
            return 200, {}, self.search_job()
        if route == "job":
            return 200, {}, self.parse_job()
        if route == "job_delete":
            return 204, {}, None
        if route == "file_delete":
            return 200, {}, {
                "object": "file_deletion",
                "workspace_id": NDI_WS_ID,
                "file_id": NDI_FILE_ID,
                "status": "deleted",
            }
        if route == "job_file":
            return 200, {}, self.full_markdown
        return 404, {}, {"error": {"type": "not_found", "message": f"no fake route for {route}"}}
