"""NDI components against a fake document service: uploads, jobs and clean-up."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest
from langchain_core.tools import ToolException
from pydantic import ValidationError

from langchain_nace import (
    NaceDocumentLoader,
    NaceError,
    NaceExtractTool,
    NaceSearchTool,
    NaceUnprocessableEntityError,
)
from tests.fakes import (
    NDI_FILE_ID,
    NDI_JOB_ID,
    NDI_WS_ID,
    FakeNdi,
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


def test_api_error_is_converted_and_redacted(ndi: FakeNdi) -> None:
    ndi.queue("parse", error(422, "chunking.strategy: unknown strategy"))
    loader = NaceDocumentLoader("https://cdn.test/invoice.html", **options(ndi))
    with pytest.raises(NaceUnprocessableEntityError) as excinfo:
        list(loader.lazy_load())
    assert "unknown strategy" in str(excinfo.value)
    assert KEY not in str(excinfo.value)


def test_configuration_rules(ndi: FakeNdi, monkeypatch: pytest.MonkeyPatch) -> None:
    for url in ("https://openrouter.ai/api/v1", "https://openrouter.ai./api/v1", "https://openrouter。ai/api"):
        with pytest.raises(ValueError, match="OpenRouter"):
            NaceDocumentLoader("https://cdn.test/invoice.html", **{**options(ndi), "base_url": url})
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
