"""Live NDI checks: parse, extract, workspace search, and an agent using the extract tool."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import nace_sdk
import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage

from langchain_nace import NaceDocumentLoader, NaceExtractTool, NaceSearchTool

FIXTURE = Path(__file__).parents[1] / "fixtures" / "invoice.html"
KEY = os.environ.get("NACE_API_KEY_LIVE")
BASE_URL = os.environ.get("NACE_BASE_URL_LIVE") or "https://console.nace.ai"  # unset = prod
SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total": {"type": "number"},
        "po_number": {"type": "string"},
    },
}
MODEL = os.environ.get("REAL_AGENT_MODEL", "anthropic:claude-sonnet-5-5")
_PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "fireworks": "FIREWORKS_API_KEY"}
_PROVIDER_KEY = _PROVIDER_KEYS.get(MODEL.split(":", 1)[0], "ANTHROPIC_API_KEY")


def ndi_options() -> dict[str, object]:
    return {"api_key": KEY, "base_url": BASE_URL}


def tool_call(name: str, args: dict[str, str], call_id: str) -> dict[str, object]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_loader_parses_a_local_file() -> None:
    loader = NaceDocumentLoader(FIXTURE, **ndi_options())
    [doc] = loader.load()
    assert "INV-2026-0042" in doc.page_content


def test_extract_tool_returns_fields_with_status() -> None:
    tool = NaceExtractTool(extraction_schema=SCHEMA, allow_local_files=True, **ndi_options())
    message = tool.invoke(tool_call("nace_extract", {"source": str(FIXTURE)}, "call_1"))
    data = json.loads(message.content)
    assert data["data"]["invoice_number"] == "INV-2026-0042"
    po_number = next(field for field in data["fields"] if field["path"] == "/po_number")
    assert po_number["status"] == "not_found"


def test_search_tool_answers_with_evidence() -> None:
    with nace_sdk.NaceClient(api_key=KEY, base_url=BASE_URL) as client:
        name = f"langchain-nace-tests-{uuid.uuid4().hex[:8]}"
        workspace = client.workspaces.create(name=name)
        try:
            uploaded = client.documents.upload(FIXTURE, workspace_id=workspace.workspace_id)
            ingestion = client.workspaces.ingest(
                workspace.workspace_id, file_ids=[uploaded.file_id], wait_seconds=60
            )
            client.jobs.wait(ingestion.job_id, timeout=300)
            tool = NaceSearchTool(workspace_id=workspace.workspace_id, mode="fact", **ndi_options())
            message = tool.invoke(
                tool_call("nace_search", {"query": "What is the total due?"}, "call_2")
            )
            data = json.loads(message.content)
            assert "400" in data["answer"]
            assert data["evidences"]
        finally:
            client.workspaces.delete(workspace.workspace_id, confirm_name=name)


def test_real_agent_calls_the_extract_tool() -> None:
    pytest.importorskip(f"langchain_{MODEL.split(':', 1)[0]}")
    if not os.environ.get(_PROVIDER_KEY):
        pytest.skip(f"set {_PROVIDER_KEY}")
    tool = NaceExtractTool(extraction_schema=SCHEMA, allow_local_files=True, **ndi_options())
    agent = create_agent(MODEL, tools=[tool])
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    f"Extract the fields from the invoice at {FIXTURE} with the extract "
                    "tool, then tell me the invoice number."
                )
            ]
        }
    )
    assert any(
        isinstance(message, ToolMessage) and message.name == "nace_extract"
        for message in result["messages"]
    )
    assert "INV-2026-0042" in result["messages"][-1].text
