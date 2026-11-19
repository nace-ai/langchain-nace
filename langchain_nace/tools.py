"""NDI as agent tools: extract structured fields, answer from a workspace."""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from langchain_core._api import beta
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

from langchain_nace._ndi import DEFAULT_JOB_TIMEOUT, NdiClient

EXTRACT_DESCRIPTION = (
    "Extract structured fields from one document. Input: the public https URL of the "
    "document. Returns the extracted data and, for each field, its status (found, "
    "not_found or ambiguous), confidence and the quoted source text."
)
EXTRACT_DESCRIPTION_LOCAL = EXTRACT_DESCRIPTION.replace(
    "Input: the public https URL of the document.",
    "Input: the https URL of the document, or the path of a local file.",
)
SEARCH_DESCRIPTION = (
    "Answer a question from the files in a Nace workspace. Returns the answer, its "
    "confidence and the evidence it cites (file path, page, quote). It searches the "
    "workspace files, not the web."
)


class _ExtractInput(BaseModel):
    source: str = Field(
        description="The https:// URL of the document (or a local file path, if the tool "
        "description permits local files)."
    )


class _SearchInput(BaseModel):
    query: str = Field(description="The question to answer from the workspace files.")


class _NdiTool(BaseTool):
    """Fields the NDI tools share: credentials, endpoint, HTTP and job settings."""

    # Pydantic merges this with BaseTool's config, so arbitrary types stay allowed; the
    # input kwargs (which may hold the api key) stay out of validation error messages.
    model_config = ConfigDict(hide_input_in_errors=True)

    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    base_url: str | None = None
    timeout: float = Field(default=DEFAULT_JOB_TIMEOUT, gt=0)
    max_retries: int = Field(default=2, ge=0)
    http_client: httpx.Client | None = Field(default=None, exclude=True, repr=False)
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    _ndi: NdiClient = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._ndi = NdiClient(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=self.max_retries,
            http_client=self.http_client,
        )


@beta()
class NaceExtractTool(_NdiTool):
    """Extract structured fields from one document with NDI, as a LangChain tool.

    The tool input is the document source, so a model controls it: by default only
    public `https://` URLs are accepted (`allow_local_files=False`); opting in lets the
    model name local files, which are uploaded and deleted after the read.

    Example:
        ```python
        from langchain_nace import NaceExtractTool

        tool = NaceExtractTool(
            extraction_schema={
                "type": "object",
                "properties": {"invoice_number": {"type": "string"}},
            }
        )
        message = tool.invoke(
            {"name": "nace_extract", "args": {"source": "https://example.com/invoice.pdf"},
             "id": "call_1", "type": "tool_call"}
        )
        ```

    Args:
        extraction_schema: The fields to extract, as a JSON Schema `dict` (NDI's subset:
            no `$ref`, `anyOf` or `format`).
        instructions: What to extract and how, in natural language.
        parse_mode: NDI effort: `"low"`, `"medium"` or `"high"`.
        allow_local_files: Accept a local file path as the tool input.
        workspace_id: Upload target for local files; without it they go to the server.
        api_key: Nace API key. Defaults to `NACE_API_KEY` for the configured endpoint.
        base_url: API root. Defaults to `NACE_BASE_URL`, else `https://console.nace.ai`.
        timeout: Seconds to wait for the job to finish.
        max_retries: SDK retries for 408, 429 and 5xx.
        http_client: Optional `httpx.Client` (custom transport, proxy, tests).

    """

    name: str = "nace_extract"
    description: str = EXTRACT_DESCRIPTION
    args_schema: type[_ExtractInput] = _ExtractInput
    extraction_schema: dict[str, Any]
    instructions: str | None = None
    parse_mode: Literal["low", "medium", "high"] = "low"
    allow_local_files: bool = False
    workspace_id: str | None = None

    def model_post_init(self, context: Any, /) -> None:
        """Build the client and make the description match `allow_local_files`."""
        super().model_post_init(context)
        # The model reads the description: it must not say "URL only" when local
        # files are permitted, or the model refuses a path the developer allowed.
        if self.allow_local_files and self.description == EXTRACT_DESCRIPTION:
            self.description = EXTRACT_DESCRIPTION_LOCAL

    def _run(
        self, source: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Extract the fields; returns the JSON answer and the raw result as artifact."""
        if "://" not in source and not self.allow_local_files:
            raise ToolException("source must be a public https:// URL.")
        try:
            with (
                self._ndi.source(source, workspace_id=self.workspace_id) as src,
                self._ndi.job(
                    lambda client, wait: client.documents.extract(
                        src,
                        schema=self.extraction_schema,
                        instructions=self.instructions,
                        parse_mode=self.parse_mode,
                        wait_seconds=wait,
                    ),
                    timeout=self.timeout,
                ) as job,
            ):
                result = job.result if isinstance(job.result, dict) else {}
                fields = []
                for field in result.get("fields") or []:
                    if not isinstance(field, dict):
                        continue
                    fields.append(
                        {
                            "path": field.get("path"),
                            "value": field.get("value"),
                            "status": field.get("status"),
                            "confidence": field.get("confidence"),
                            "quotes": [
                                citation.get("source_text")
                                for citation in field.get("citations") or []
                                if isinstance(citation, dict) and citation.get("source_text")
                            ],
                        }
                    )
                answer = json.dumps(
                    {"data": result.get("data"), "fields": fields}, ensure_ascii=False
                )
                return answer, result
        except ValueError as error:
            raise ToolException(str(error)) from None


@beta()
class NaceSearchTool(_NdiTool):
    """Answer a question from one ingested Nace workspace, as a LangChain tool.

    Example:
        ```python
        from langchain_nace import NaceSearchTool

        tool = NaceSearchTool(workspace_id="...")
        message = tool.invoke(
            {"name": "nace_search", "args": {"query": "What is the total due?"},
             "id": "call_1", "type": "tool_call"}
        )
        ```

    Args:
        workspace_id: The workspace to search; create and ingest one with `nace_sdk`.
        mode: `"auto"`, or `"fact"`/`"agent"` to skip the routing step.
        top_k: Cap on the evidences a fact answer cites (1 to 50).
        path_prefix: Search only the files under this path.
        api_key: Nace API key. Defaults to `NACE_API_KEY` for the configured endpoint.
        base_url: API root. Defaults to `NACE_BASE_URL`, else `https://console.nace.ai`.
        timeout: Seconds to wait for the job to finish.
        max_retries: SDK retries for 408, 429 and 5xx.
        http_client: Optional `httpx.Client` (custom transport, proxy, tests).

    """

    name: str = "nace_search"
    description: str = SEARCH_DESCRIPTION
    args_schema: type[_SearchInput] = _SearchInput
    workspace_id: str
    mode: Literal["auto", "fact", "agent"] = "auto"
    top_k: int | None = Field(default=None, ge=1, le=50)
    path_prefix: str | None = None

    def _run(
        self, query: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Answer the question; returns the JSON answer and the raw result as artifact."""
        with self._ndi.job(
            lambda client, wait: client.search.automatic(
                self.workspace_id,
                query=query,
                mode=self.mode,
                top_k=self.top_k,
                path_prefix=self.path_prefix,
                wait_seconds=wait,
            ),
            timeout=self.timeout,
        ) as job:
            result = job.result if isinstance(job.result, dict) else {}
            raw_output = result.get("output")
            output = raw_output if isinstance(raw_output, dict) else {}
            raw_answer = output.get("answer")
            answer = raw_answer if isinstance(raw_answer, dict) else output
            evidences = [
                {
                    "source_path": evidence.get("source_path"),
                    "page": evidence.get("page"),
                    "quote": evidence.get("quote"),
                }
                for evidence in answer.get("evidences") or []
                if isinstance(evidence, dict)
            ]
            content = json.dumps(
                {
                    "answer": answer.get("answer"),
                    "confidence": answer.get("confidence"),
                    "evidences": evidences,
                },
                ensure_ascii=False,
            )
            return content, result


__all__ = ["NaceExtractTool", "NaceSearchTool"]
