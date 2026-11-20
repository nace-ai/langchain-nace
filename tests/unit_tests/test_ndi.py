"""NDI components against a fake document service: uploads, jobs and clean-up."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.tools import ToolException
from pydantic import ValidationError

from langchain_nace import (
    NaceAPIResponseValidationError,
    NaceAPITimeoutError,
    NaceDocumentLoader,
    NaceError,
    NaceExtractTool,
    NaceInternalServerError,
    NaceSearchTool,
    NaceUnprocessableEntityError,
)
from tests.fakes import (
    NDI_FILE_ID,
    NDI_JOB_ID,
    NDI_WS_ID,
    FakeNdi,
    Reply,
    agent_output,
    error,
    fact_output,
)

KEY = "nace_sk_FAKE_FOR_TEST"
SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total": {"type": "number"},
        "po_number": {"type": "string"},
    },
}
TOOL_CALL = {"name": "nace_extract", "type": "tool_call"}


@pytest.fixture
def ndi() -> FakeNdi:
    return FakeNdi()


def options(fake: FakeNdi) -> dict[str, object]:
    return {
        "api_key": KEY,
        "base_url": "https://ndi.test",
        "http_client": httpx.Client(transport=fake.transport()),
    }


def test_loader_local_file_parses_and_cleans_up(ndi: FakeNdi, tmp_path: Path) -> None:
    path = tmp_path / "invoice.html"
    path.write_text("<html><body>Invoice INV-2026-0042</body></html>")
    loader = NaceDocumentLoader(path, **options(ndi))
    [doc] = list(loader.lazy_load())
    assert "INV-2026-0042" in doc.page_content
    assert doc.metadata == {"source": str(path), "job_id": NDI_JOB_ID, "page_count": 2}
    grant, upload, parse, *rest = ndi.requests
    assert grant.url.path == "/v1/documents/upload-grants"
    assert ndi.bodies[0]["path"].startswith("langchain-nace/")
    assert ndi.bodies[0]["path"].endswith("/invoice.html")
    # A second load of the same file uploads under a new unique path, never the same file.
    list(loader.lazy_load())
    paths = [body["path"] for body in ndi.bodies if "path" in body]
    assert len(paths) == 2
    assert paths[0] != paths[1]
    assert all(
        path.startswith("langchain-nace/") and path.endswith("/invoice.html") for path in paths
    )
    # Plain text output returns the text instead of the Markdown.
    ndi.markdown = None
    ndi.text = "Invoice INV-2026-0042 in plain text"
    text_loader = NaceDocumentLoader(
        path, parse_kwargs={"output": {"formats": ["text"]}}, **options(ndi)
    )
    [text_doc] = list(text_loader.lazy_load())
    assert text_doc.page_content == "Invoice INV-2026-0042 in plain text"
    # Neither rendering requested: an error, not an empty document.
    ndi.text = None
    with pytest.raises(NaceError, match=r"output\.formats"):
        list(text_loader.lazy_load())
    assert upload.url.host == "upload.ndi.test"
    assert upload.headers["x-upload-token"] == "upl_FAKE_FOR_TEST"
    assert "authorization" not in upload.headers
    for request in (grant, parse, *rest):
        assert request.headers["authorization"] == f"Bearer {KEY}"
    assert [(r.method, r.url.path) for r in rest] == [
        ("DELETE", f"/v1/documents/jobs/{NDI_JOB_ID}"),
        ("DELETE", f"/v1/documents/workspaces/{NDI_WS_ID}/files/{NDI_FILE_ID}"),
    ]


def test_loader_url_source_chunks_after_a_queued_job(ndi: FakeNdi) -> None:
    ndi.chunks = [
        {
            "id": "chk_1",
            "content": "# Invoice INV-2026-0042",
            "location": {"kind": "page_region", "page": 1},
        },
        {
            "id": "chk_2",
            "content": "Total due: USD 400.00",
            "location": {"kind": "page_region", "page": None},
        },
    ]
    ndi.queue("parse", (202, {}, {"job_id": NDI_JOB_ID, "kind": "parse", "status": "queued"}))
    ndi.queue("job", (200, {}, ndi.parse_job()))
    loader = NaceDocumentLoader(
        "https://cdn.test/invoice.html", chunking="page", parse_mode="high", **options(ndi)
    )
    iterator = loader.lazy_load()
    first = next(iterator)
    # Clean-up is done before the first document is handed out.
    assert ndi.requests[-1].method == "DELETE"
    docs = [first, *iterator]
    assert [doc.page_content for doc in docs] == [
        "# Invoice INV-2026-0042",
        "Total due: USD 400.00",
    ]
    assert docs[0].metadata["chunk_id"] == "chk_1"
    assert docs[0].metadata["page"] == 1
    assert "page" not in docs[1].metadata
    assert not any(r.url.path in ("/v1/documents/upload-grants", "/uploads") for r in ndi.requests)
    body = ndi.bodies[0]
    assert body["source"] == {
        "type": "url",
        "url": "https://cdn.test/invoice.html",
        "file_name": "invoice.html",
    }
    assert body["chunking"] == {"strategy": "page"}
    assert body["parse_mode"] == "high"


def test_loader_returns_full_markdown_when_truncated(ndi: FakeNdi) -> None:
    ndi.markdown = "# Invoice INV-2026-0042"  # the preview only
    ndi.disclosures = [{"code": "content_truncated", "message": "over 1 MiB"}]
    ndi.content_url = f"https://ndi.test/v1/documents/jobs/{NDI_JOB_ID}/content"
    loader = NaceDocumentLoader(
        "https://cdn.test/invoice", file_name="invoice.html", **options(ndi)
    )
    [doc] = list(loader.lazy_load())
    assert doc.page_content == ndi.full_markdown
    [download] = [r for r in ndi.requests if r.url.path.endswith("/content")]
    assert download.headers["authorization"] == f"Bearer {KEY}"
    # A truncated result without a content_url cannot be completed: say so, not the preview.
    ndi.content_url = None
    loader = NaceDocumentLoader(
        "https://cdn.test/invoice", file_name="invoice.html", **options(ndi)
    )
    with pytest.raises(NaceError, match="content_url"):
        list(loader.lazy_load())
    # A failed download surfaces its error like any other SDK call.
    ndi.content_url = f"https://ndi.test/v1/documents/jobs/{NDI_JOB_ID}/content"
    ndi.queue("job_file", error(500, "download failed"))
    broken = NaceDocumentLoader(
        "https://cdn.test/invoice",
        file_name="invoice.html",
        max_retries=0,
        **options(ndi),
    )
    with pytest.raises(NaceInternalServerError, match="download failed"):
        list(broken.lazy_load())


def test_stalled_content_download_times_out_and_still_cancels(ndi: FakeNdi) -> None:
    # The SDK streams the full content with no read timeout: a stalled server would
    # hold the stream (and the job's clean-up) open forever. The download must stop
    # at the job budget, and the job must still be cancelled.
    ndi.markdown = "# Invoice INV-2026-0042"  # the preview only
    ndi.disclosures = [{"code": "content_truncated", "message": "over 1 MiB"}]
    ndi.content_url = f"https://ndi.test/v1/documents/jobs/{NDI_JOB_ID}/content"

    def stalled(body: dict[str, Any]) -> Reply:
        time.sleep(0.5)
        return 200, {}, ndi.full_markdown

    ndi.queue("job_file", stalled)
    loader = NaceDocumentLoader("https://cdn.test/invoice.html", timeout=0.2, **options(ndi))
    started = time.monotonic()
    with pytest.raises(NaceAPITimeoutError, match="download"):
        list(loader.lazy_load())
    # The stalled transfer is abandoned well before it would have finished.
    assert time.monotonic() - started < 0.45
    last = ndi.requests[-1]
    assert (last.method, last.url.path) == ("DELETE", f"/v1/documents/jobs/{NDI_JOB_ID}")


def test_failed_job_raises_and_still_cleans_up(ndi: FakeNdi, tmp_path: Path) -> None:
    path = tmp_path / "invoice.html"
    path.write_text("not a real invoice")
    ndi.queue(
        "parse",
        (
            200,
            {},
            {
                "job_id": NDI_JOB_ID,
                "kind": "parse",
                "status": "failed",
                "error": {"code": "parser_failed", "message": "The document could not be parsed."},
            },
        ),
    )
    loader = NaceDocumentLoader(path, **options(ndi))
    with pytest.raises(NaceError, match="parser_failed"):
        list(loader.lazy_load())
    assert [(r.method, r.url.path) for r in ndi.requests[-2:]] == [
        ("DELETE", f"/v1/documents/jobs/{NDI_JOB_ID}"),
        ("DELETE", f"/v1/documents/workspaces/{NDI_WS_ID}/files/{NDI_FILE_ID}"),
    ]


def test_wait_timeout_raises_and_cancels_the_job(ndi: FakeNdi) -> None:
    ndi.stuck_queued = True
    ndi.queue("parse", (202, {}, {"job_id": NDI_JOB_ID, "kind": "parse", "status": "queued"}))
    loader = NaceDocumentLoader(
        "https://cdn.test/invoice.html", timeout=0.2, **options(ndi)
    )
    with pytest.raises(NaceAPITimeoutError, match="did not finish"):
        list(loader.lazy_load())
    # The wait budget ran out, but the stuck job is still cancelled.
    last = ndi.requests[-1]
    assert (last.method, last.url.path) == ("DELETE", f"/v1/documents/jobs/{NDI_JOB_ID}")


def test_create_past_the_timeout_raises_and_cancels(ndi: FakeNdi) -> None:
    # The create request answered a finished job, but only after the whole job budget
    # had passed: that is a timeout, not a late success.
    def slow_create(body: dict[str, Any]) -> Reply:
        time.sleep(0.16)
        return 200, {}, ndi.parse_job()

    ndi.queue("parse", slow_create)
    loader = NaceDocumentLoader("https://cdn.test/invoice.html", timeout=0.05, **options(ndi))
    with pytest.raises(NaceAPITimeoutError, match="did not finish"):
        list(loader.lazy_load())
    # The late job is cancelled, not left running.
    last = ndi.requests[-1]
    assert (last.method, last.url.path) == ("DELETE", f"/v1/documents/jobs/{NDI_JOB_ID}")


def test_retries_stop_at_the_job_budget(ndi: FakeNdi) -> None:
    # A retried create waits out its backoff outside the wait loop; the retry budget
    # must stop it before the whole job budget is spent, surfacing the server's error.
    ndi.queue(
        "parse",
        (
            503,
            {},
            {
                "error": {
                    "type": "service_unavailable",
                    "code": "job_pending",
                    "message": "The service is busy; the job may still run.",
                },
                "request_id": "req_busy",
            },
        ),
    )
    loader = NaceDocumentLoader("https://cdn.test/invoice.html", timeout=0.05, **options(ndi))
    with pytest.raises(NaceInternalServerError, match="busy"):
        list(loader.lazy_load())
    creates = [r for r in ndi.requests if r.url.path == "/v1/documents/parse"]
    assert len(creates) == 1


def test_create_hold_stays_within_the_job_budget(ndi: FakeNdi) -> None:
    # The hold the create asks the server for is part of the budget: with less than
    # the one-second hold minimum left, no hold is requested at all.
    loader = NaceDocumentLoader("https://cdn.test/invoice.html", timeout=0.5, **options(ndi))
    list(loader.lazy_load())
    [create] = [r for r in ndi.requests if r.url.path == "/v1/documents/parse"]
    assert "wait_seconds" not in create.url.params
    # A comfortable budget still asks for a hold, capped by the SDK's maximum.
    ndi.requests.clear()
    loader = NaceDocumentLoader("https://cdn.test/invoice.html", timeout=120, **options(ndi))
    list(loader.lazy_load())
    [create] = [r for r in ndi.requests if r.url.path == "/v1/documents/parse"]
    assert create.url.params.get("wait_seconds") == "60"


def test_no_job_read_after_the_deadline(ndi: FakeNdi) -> None:
    # Every poll must start before the budget runs out; the SDK's own wait loop
    # issues one more read after the deadline before raising.
    ndi.stuck_queued = True
    ndi.queue("parse", (202, {}, {"job_id": NDI_JOB_ID, "kind": "parse", "status": "queued"}))
    reads: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if (request.method, request.url.path) == ("GET", f"/v1/documents/jobs/{NDI_JOB_ID}"):
            reads.append(time.monotonic())
        return ndi(request)

    loader = NaceDocumentLoader(
        "https://cdn.test/invoice.html",
        timeout=0.2,
        **{
            **options(ndi),
            "http_client": httpx.Client(transport=httpx.MockTransport(handler)),
        },
    )
    started = time.monotonic()
    with pytest.raises(NaceAPITimeoutError, match="did not finish"):
        list(loader.lazy_load())
    deadline = started + 0.2
    assert reads  # the job was polled, and only before the deadline
    assert all(when < deadline for when in reads)
    # The stuck job is still cancelled.
    last = ndi.requests[-1]
    assert (last.method, last.url.path) == ("DELETE", f"/v1/documents/jobs/{NDI_JOB_ID}")


def test_job_pending_error_keeps_the_idempotency_key(ndi: FakeNdi) -> None:
    # A 503 job_pending may leave the job running server-side; the SDK documents that
    # resending the same body with the same Idempotency-Key resumes it. The raised
    # error must keep that key, or a retry mints a new one and starts a second job.
    ndi.queue(
        "parse",
        (
            503,
            {},
            {
                "error": {
                    "type": "service_unavailable",
                    "code": "job_pending",
                    "message": "The document job is pending; it may still run.",
                },
                "request_id": "req_pending",
            },
        ),
    )
    loader = NaceDocumentLoader(
        "https://cdn.test/invoice.html", max_retries=0, **options(ndi)
    )
    with pytest.raises(NaceInternalServerError, match="pending") as excinfo:
        list(loader.lazy_load())
    key = excinfo.value.idempotency_key
    [create] = [r for r in ndi.requests if r.url.path == "/v1/documents/parse"]
    assert key is not None
    assert key == create.headers["idempotency-key"]
    # No job came back, so there is nothing to delete: the key is the only handle.
    assert not any(r.method == "DELETE" for r in ndi.requests)


def test_job_pending_keeps_the_uploaded_file(ndi: FakeNdi, tmp_path: Path) -> None:
    # The pending job and its documented replay need the uploaded file: it must not
    # be deleted, and the error must say where it is and how to delete it.
    path = tmp_path / "invoice.html"
    path.write_text("<html><body>Invoice INV-2026-0042</body></html>")
    ndi.queue(
        "parse",
        (
            503,
            {},
            {
                "error": {
                    "type": "service_unavailable",
                    "code": "job_pending",
                    "message": "The document job is pending; it may still run.",
                },
                "request_id": "req_pending",
            },
        ),
    )
    loader = NaceDocumentLoader(path, max_retries=0, **options(ndi))
    with pytest.raises(NaceInternalServerError, match="pending") as excinfo:
        list(loader.lazy_load())
    assert excinfo.value.idempotency_key is not None
    text = str(excinfo.value)
    assert "kept" in text
    assert NDI_WS_ID in text
    assert NDI_FILE_ID in text
    assert "delete_file" in text
    assert not any(r.method == "DELETE" for r in ndi.requests)
    # Any other create failure leaves no pending job: the file is still cleaned up.
    ndi.queue("parse", error(422, "chunking.strategy: unknown strategy"))
    loader = NaceDocumentLoader(path, max_retries=0, **options(ndi))
    with pytest.raises(NaceUnprocessableEntityError):
        list(loader.lazy_load())
    assert any(
        (r.method, r.url.path)
        == ("DELETE", f"/v1/documents/workspaces/{NDI_WS_ID}/files/{NDI_FILE_ID}")
        for r in ndi.requests
    )


def test_api_error_is_converted_and_redacted(
    ndi: FakeNdi, caplog: pytest.LogCaptureFixture
) -> None:
    # An opaque key the generic redaction patterns would not match: only the secrets
    # passed to `convert_error` by `NdiClient.call` can redact it.
    opaque = "opaque-FAKE-key-for-test"
    ndi.queue("parse", error(422, f"chunking.strategy: unknown strategy (key {opaque})"))
    loader = NaceDocumentLoader(
        # A key read from a file often ends in a newline; the SDK sends it stripped.
        "https://cdn.test/invoice.html", **{**options(ndi), "api_key": opaque + "\n"}
    )
    with pytest.raises(NaceUnprocessableEntityError) as excinfo:
        list(loader.lazy_load())
    assert "unknown strategy" in str(excinfo.value)
    assert opaque not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)
    # The message is redacted, and so are the other places the echoed key survives:
    # the upstream body kept for inspection, and the raw SDK error the wrapper would
    # still reference as its `__context__`.
    assert opaque not in json.dumps(excinfo.value.body)
    assert "[REDACTED]" in json.dumps(excinfo.value.body)
    assert excinfo.value.__context__ is None
    # The SDK's DEBUG wire log prints the response body verbatim (its own filter masks
    # only credential headers and secret body fields); our keys must not reach it.
    ndi.queue("parse", error(422, f"chunking.strategy: unknown strategy (key {opaque})"))
    with (
        caplog.at_level(logging.DEBUG, logger="nace_sdk"),
        pytest.raises(NaceUnprocessableEntityError),
    ):
        list(loader.lazy_load())
    assert any(record.levelno == logging.DEBUG for record in caplog.records)
    assert opaque not in caplog.text


def test_configuration_rules(ndi: FakeNdi, monkeypatch: pytest.MonkeyPatch) -> None:
    for url in (
        "https://openrouter.ai/api/v1",
        "https://openrouter.ai./api/v1",
        "https://openrouter。ai/api",
    ):
        with pytest.raises(ValueError, match="OpenRouter"):
            NaceDocumentLoader("https://cdn.test/invoice.html", **{**options(ndi), "base_url": url})
    with pytest.raises(ValueError, match="absolute"):
        NaceDocumentLoader("https://cdn.test/invoice.html", **{**options(ndi), "base_url": "/api"})
    monkeypatch.setenv("NACE_API_KEY", KEY)
    monkeypatch.setenv("NACE_BASE_URL", "https://openrouter.ai/api")
    with pytest.raises(ValueError, match="OpenRouter"):
        NaceDocumentLoader(
            "https://cdn.test/invoice.html", http_client=httpx.Client(transport=ndi.transport())
        )
    with pytest.raises(ValueError, match=r"(?i)api key"):
        NaceDocumentLoader(
            "https://cdn.test/invoice.html",
            api_key="",
            base_url="https://ndi.test",
            http_client=httpx.Client(transport=ndi.transport()),
        )
    with pytest.raises(ValueError, match="https"):
        list(NaceDocumentLoader("http://cdn.test/invoice.html", **options(ndi)).lazy_load())
    with pytest.raises(ValueError, match="file_name"):
        list(NaceDocumentLoader("https://cdn.test/invoice", **options(ndi)).lazy_load())
    with pytest.raises(ValueError, match="chunking"):
        NaceDocumentLoader(
            "https://cdn.test/invoice.html",
            parse_kwargs={"chunking": {"strategy": "none"}},
            **options(ndi),
        )
    with pytest.raises(ValueError, match="extra_body"):
        NaceDocumentLoader(
            "https://cdn.test/invoice.html",
            parse_kwargs={"extra_body": {}},
            **options(ndi),
        )
    # `timeout` would collide with the loader's own keyword on the SDK call: a clear
    # rejection in the constructor, not a TypeError from lazy_load.
    with pytest.raises(ValueError, match="timeout"):
        NaceDocumentLoader(
            "https://cdn.test/invoice.html",
            parse_kwargs={"timeout": 30},
            **options(ndi),
        )
    with pytest.raises(ValidationError, match="workspace_id") as as_error:
        NaceSearchTool(api_key=KEY, base_url="https://ndi.test")
    assert KEY not in str(as_error.value)
    assert ndi.requests == []
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_FAKE_OTHER_ENDPOINT")
    monkeypatch.setenv("NACE_BASE_URL", "https://other.ndi.test")
    loader = NaceDocumentLoader(
        "https://cdn.test/invoice.html", http_client=httpx.Client(transport=ndi.transport())
    )
    [doc] = list(loader.lazy_load())
    assert "INV-2026-0042" in doc.page_content
    parse_request, *rest = ndi.requests
    assert str(parse_request.url).startswith("https://other.ndi.test/v1/documents/parse")
    assert parse_request.headers["authorization"] == "Bearer nace_sk_FAKE_OTHER_ENDPOINT"
    assert all(r.url.host == "other.ndi.test" for r in rest)
    # A whitespace-only explicit base_url means "use the environment", like the classifier.
    monkeypatch.setenv("NACE_BASE_URL", "https://blank.ndi.test")
    blank = NaceDocumentLoader(
        "https://cdn.test/invoice.html",
        base_url="   ",
        http_client=httpx.Client(transport=ndi.transport()),
    )
    [doc] = list(blank.lazy_load())
    assert "INV-2026-0042" in doc.page_content
    assert all(r.url.host == "blank.ndi.test" for r in ndi.requests[2:])


def test_extract_tool_returns_fields_quotes_and_artifact(ndi: FakeNdi, tmp_path: Path) -> None:
    tool = NaceExtractTool(
        extraction_schema=SCHEMA,
        instructions="Extract the invoice fields.",
        parse_mode="medium",
        **options(ndi),
    )
    message = tool.invoke(
        {**TOOL_CALL, "args": {"source": "https://cdn.test/invoice.html"}, "id": "call_1"}
    )
    data = json.loads(message.content)
    assert data["data"] == ndi.extract_job()["result"]["data"]
    total = next(field for field in data["fields"] if field["path"] == "/total")
    assert (total["status"], total["quotes"]) == ("found", ["Total due: USD 400.00"])
    assert message.artifact == ndi.extract_job()["result"]
    body = ndi.bodies[0]
    assert body["schema"] == SCHEMA
    assert body["instructions"] == "Extract the invoice fields."
    assert body["parse_mode"] == "medium"
    assert f"/v1/documents/jobs/{NDI_JOB_ID}" in [
        r.url.path for r in ndi.requests if r.method == "DELETE"
    ]
    # The model controls the input: a local path is refused by default, with no request.
    path = tmp_path / "invoice.html"
    path.write_text("<html><body>Invoice INV-2026-0042</body></html>")
    sent = len(ndi.requests)
    with pytest.raises(ToolException):
        tool.invoke({**TOOL_CALL, "args": {"source": str(path)}, "id": "call_2"})
    assert len(ndi.requests) == sent
    # Opting in accepts local files; a missing one is still a ToolException, no request.
    local = NaceExtractTool(extraction_schema=SCHEMA, allow_local_files=True, **options(ndi))
    assert "local file" in local.description
    assert "local file" not in tool.description
    missing = str(tmp_path / "missing.html")
    with pytest.raises(ToolException, match="No such file"):
        local.invoke({**TOOL_CALL, "args": {"source": missing}, "id": "call_3"})
    assert len(ndi.requests) == sent
    # An existing file is uploaded, parsed and deleted after the read.
    message = local.invoke({**TOOL_CALL, "args": {"source": str(path)}, "id": "call_4"})
    assert json.loads(message.content)["data"]["invoice_number"] == "INV-2026-0042"
    assert f"/v1/documents/workspaces/{NDI_WS_ID}/files/{NDI_FILE_ID}" in [
        r.url.path for r in ndi.requests if r.method == "DELETE"
    ]


def test_search_tool_answers_fact_and_agent_shapes(ndi: FakeNdi) -> None:
    ndi.queue("search", (200, {}, ndi.search_job(fact_output())))
    ndi.queue("search", (200, {}, ndi.search_job(agent_output())))
    tool = NaceSearchTool(workspace_id=NDI_WS_ID, mode="fact", top_k=5, **options(ndi))
    call = {"name": "nace_search", "args": {"query": "What is the total due?"}, "type": "tool_call"}
    fact = json.loads(tool.invoke({**call, "id": "call_1"}).content)
    agent = json.loads(tool.invoke({**call, "id": "call_2"}).content)
    assert fact["answer"] == "The total due is USD 400.00."
    assert fact["confidence"] == "high"
    assert fact["evidences"] == [
        {"source_path": "invoice.html", "page": 1, "quote": "Total due: USD 400.00"}
    ]
    assert agent["answer"] == "The total due is USD 400.00."
    assert agent["confidence"] == "medium"
    assert agent["evidences"] == fact["evidences"]
    body = ndi.bodies[0]
    assert body["query"] == "What is the total due?"
    assert body["mode"] == "fact"
    assert body["top_k"] == 5
    assert len([r for r in ndi.requests if r.method == "DELETE"]) == 2


def test_missing_results_raise_response_errors(ndi: FakeNdi) -> None:
    # A succeeded job whose result is missing must not become empty output: the page
    # mode returned no documents, Extract null data and Search a null answer.
    loader = NaceDocumentLoader(
        "https://cdn.test/invoice.html", chunking="page", **options(ndi)
    )
    with pytest.raises(NaceAPIResponseValidationError, match="chunks"):
        list(loader.lazy_load())
    ndi.queue(
        "extract",
        (200, {}, {"job_id": NDI_JOB_ID, "kind": "extract", "status": "succeeded",
                   "result": None}),
    )
    extract = NaceExtractTool(extraction_schema=SCHEMA, **options(ndi))
    with pytest.raises(NaceAPIResponseValidationError, match="data"):
        extract.invoke(
            {**TOOL_CALL, "args": {"source": "https://cdn.test/invoice.html"}, "id": "call_1"}
        )
    ndi.queue(
        "search",
        (200, {}, {"job_id": NDI_JOB_ID, "kind": "automatic_search", "status": "succeeded",
                   "result": None}),
    )
    search = NaceSearchTool(workspace_id=NDI_WS_ID, **options(ndi))
    with pytest.raises(NaceAPIResponseValidationError, match="answer"):
        search.invoke(
            {
                "name": "nace_search",
                "args": {"query": "What is the total due?"},
                "id": "call_2",
                "type": "tool_call",
            }
        )
    # The jobs behind the unusable results are still deleted.
    assert len([r for r in ndi.requests if r.method == "DELETE"]) == 3


def test_validation_errors_do_not_expose_the_api_key() -> None:
    # A missing required field makes pydantic echo the whole input dict in
    # `.errors()` and `.json()`; the key must be masked by then.
    with pytest.raises(ValidationError, match="workspace_id") as search_error:
        NaceSearchTool(api_key=KEY, base_url="https://ndi.test")
    assert KEY not in search_error.value.json()
    assert all(KEY not in str(issue) for issue in search_error.value.errors())
    with pytest.raises(ValidationError, match="extraction_schema") as extract_error:
        NaceExtractTool(api_key=KEY, base_url="https://ndi.test")
    assert KEY not in extract_error.value.json()
    assert all(KEY not in str(issue) for issue in extract_error.value.errors())


def test_empty_results_raise_response_errors(ndi: FakeNdi) -> None:
    # A succeeded job whose answer is empty is as unusable as a missing one.
    ndi.queue(
        "extract",
        (
            200,
            {},
            {
                "job_id": NDI_JOB_ID,
                "kind": "extract",
                "status": "succeeded",
                "result": {"data": None, "fields": []},
            },
        ),
    )
    extract = NaceExtractTool(extraction_schema=SCHEMA, **options(ndi))
    with pytest.raises(NaceAPIResponseValidationError, match="data"):
        extract.invoke(
            {**TOOL_CALL, "args": {"source": "https://cdn.test/invoice.html"}, "id": "call_1"}
        )
    search = NaceSearchTool(workspace_id=NDI_WS_ID, **options(ndi))
    outputs = [
        # An empty fact answer.
        {"answer": "", "confidence": "high", "evidences": []},
        # An agent answer object without an answer text.
        {"route": "agent", "answer": {"answer": "", "confidence": "low", "evidences": []}},
    ]
    for output in outputs:
        ndi.queue("search", (200, {}, ndi.search_job(output)))
        with pytest.raises(NaceAPIResponseValidationError, match="answer"):
            search.invoke(
                {
                    "name": "nace_search",
                    "args": {"query": "What is the total due?"},
                    "id": "call_2",
                    "type": "tool_call",
                }
            )


def test_content_response_format_returns_content_only(ndi: FakeNdi) -> None:
    # With response_format="content" the model must see the JSON answer only; the
    # raw job result is the artifact and stays out of the ToolMessage content.
    extract = NaceExtractTool(
        extraction_schema=SCHEMA, response_format="content", **options(ndi)
    )
    message = extract.invoke(
        {**TOOL_CALL, "args": {"source": "https://cdn.test/invoice.html"}, "id": "call_1"}
    )
    data = json.loads(message.content)
    assert set(data) == {"data", "fields"}
    assert data["data"]["invoice_number"] == "INV-2026-0042"
    assert message.artifact is None
    search = NaceSearchTool(
        workspace_id=NDI_WS_ID, response_format="content", **options(ndi)
    )
    answer = json.loads(
        search.invoke(
            {
                "name": "nace_search",
                "args": {"query": "What is the total due?"},
                "id": "call_2",
                "type": "tool_call",
            }
        ).content
    )
    assert set(answer) == {"answer", "confidence", "evidences"}
    assert answer["answer"] == "The total due is USD 400.00."


def test_cleanup_failure_warns_but_keeps_the_result(
    ndi: FakeNdi, caplog: pytest.LogCaptureFixture
) -> None:
    ndi.queue("job_delete", error(500, "delete failed"))
    tool = NaceExtractTool(extraction_schema=SCHEMA, max_retries=0, **options(ndi))
    with caplog.at_level(logging.WARNING, logger="langchain_nace._ndi"):
        message = tool.invoke(
            {**TOOL_CALL, "args": {"source": "https://cdn.test/invoice.html"}, "id": "call_1"}
        )
    assert json.loads(message.content)["data"]["invoice_number"] == "INV-2026-0042"
    [record] = caplog.records
    assert record.getMessage() == f"Could not delete the NDI job {NDI_JOB_ID}."
