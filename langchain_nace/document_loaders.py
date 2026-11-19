"""`NaceDocumentLoader`: turn one document into LangChain `Document` objects with NDI Parse."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

import httpx
from langchain_core._api import beta
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from nace_sdk import DocumentJob
from pydantic import SecretStr

from langchain_nace._ndi import DEFAULT_JOB_TIMEOUT, NdiClient, Source
from langchain_nace.errors import NaceError

TRUNCATED = "content_truncated"
_PARSE_ONLY = ("source", "chunking", "parse_mode")
_LOADER_SETS = ("wait_seconds", "extra_body")


@beta()
class NaceDocumentLoader(BaseLoader):
    """Parse one document to Markdown (whole, or chunked by page or section) with NDI.

    The source is a local file (uploaded, then deleted after the read) or a public
    `https://` URL. Each job and upload the loader makes is deleted before it returns,
    also after an error; a failed clean-up is logged, not raised.

    Example:
        ```python
        from langchain_nace import NaceDocumentLoader

        loader = NaceDocumentLoader("invoice.pdf", chunking="page")  # reads NACE_API_KEY
        pages = loader.load()
        pages[0].metadata["page"]  # 1
        ```

    Args:
        source: A local file path or a public `https://` URL.
        chunking: `"none"` for one document with the whole Markdown, `"page"` or
            `"section"` for one document per chunk.
        parse_mode: NDI effort: `"low"`, `"medium"` or `"high"`.
        file_name: File type of a URL source whose last path segment has no extension.
        workspace_id: Upload target for a local file; without it the file is uploaded
            to the server.
        parse_kwargs: More Parse body fields; they replace the defaults, so
            `output={"formats": ["text"]}` returns the plain text instead of the Markdown.
        timeout: Seconds to wait for the job to finish.
        api_key: Nace API key. Defaults to `NACE_API_KEY` for the configured endpoint.
        base_url: API root. Defaults to `NACE_BASE_URL`, else `https://console.nace.ai`.
        max_retries: SDK retries for 408, 429 and 5xx.
        http_client: Optional `httpx.Client` (custom transport, proxy, tests).

    """

    def __init__(
        self,
        source: str | os.PathLike[str],
        *,
        chunking: Literal["none", "page", "section"] = "none",
        parse_mode: Literal["low", "medium", "high"] = "low",
        file_name: str | None = None,
        workspace_id: str | None = None,
        parse_kwargs: Mapping[str, Any] | None = None,
        timeout: float = DEFAULT_JOB_TIMEOUT,
        api_key: SecretStr | str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.source: Source = source
        self.chunking = chunking
        self.parse_mode = parse_mode
        self.file_name = file_name
        self.workspace_id = workspace_id
        self.parse_kwargs = dict(parse_kwargs) if parse_kwargs else {}
        for key in _PARSE_ONLY:
            if key in self.parse_kwargs:
                message = f"parse_kwargs must not set {key!r}; use the loader's {key} argument."
                raise ValueError(message)
        for key in _LOADER_SETS:
            if key in self.parse_kwargs:
                message = f"parse_kwargs must not set {key!r}; the loader sets it."
                raise ValueError(message)
        self.timeout = timeout
        self._ndi = NdiClient(
            api_key=api_key, base_url=base_url, max_retries=max_retries, http_client=http_client
        )

    def lazy_load(self) -> Iterator[Document]:
        """Parse the document and yield it, or each of its chunks, as a `Document`."""
        body: dict[str, Any] = {
            "parse_mode": self.parse_mode,
            "chunking": {"strategy": self.chunking},
            "output": {"formats": ["markdown"]},
            **self.parse_kwargs,
        }
        with (
            self._ndi.source(
                self.source, file_name=self.file_name, workspace_id=self.workspace_id
            ) as src,
            self._ndi.job(
                lambda client, wait: client.documents.parse(src, **body, wait_seconds=wait),
                timeout=self.timeout,
            ) as job,
        ):
            documents = self._documents(job)
        # The job and the upload are deleted before the first document is handed out, so
        # a consumer that stops early (or is slow) leaves nothing behind.
        yield from documents

    def _documents(self, job: DocumentJob) -> list[Document]:
        result = job.result if isinstance(job.result, dict) else {}
        raw_document = result.get("document")
        document = raw_document if isinstance(raw_document, dict) else {}
        metadata: dict[str, Any] = {
            "source": self.source if isinstance(self.source, str) else os.fspath(self.source),
            "job_id": job.job_id,
            "page_count": document.get("page_count"),
        }
        if self.chunking == "none":
            if document.get("markdown") is None and document.get("text") is None:
                message = (
                    "The parse result has no Markdown or text; keep 'markdown' or 'text' "
                    "in output.formats."
                )
                raise NaceError(message)
            markdown = document.get("markdown") or document.get("text") or ""
            truncated = any(
                isinstance(item, dict) and item.get("code") == TRUNCATED
                for item in result.get("disclosures") or []
            )
            if truncated:
                if not document.get("content_url"):
                    raise NaceError(
                        "The parse result is truncated and has no content_url for the full text."
                    )
                markdown = self._full_markdown(job.job_id, document["content_url"])
            return [Document(page_content=markdown, metadata=metadata)]
        documents = []
        for chunk in document.get("chunks") or []:
            if not isinstance(chunk, dict):
                continue
            chunk_metadata = {**metadata, "chunk_id": chunk.get("id")}
            location = chunk.get("location")
            page = location.get("page") if isinstance(location, dict) else None
            if page is not None:
                chunk_metadata["page"] = page
            documents.append(
                Document(page_content=chunk.get("content") or "", metadata=chunk_metadata)
            )
        return documents

    def _full_markdown(self, job_id: str, content_url: str) -> str:
        """The full Markdown of a truncated result, downloaded through the job."""
        with tempfile.TemporaryDirectory() as folder:
            target = self._ndi.call(
                lambda client: client.jobs.download(
                    job_id, content_url, Path(folder) / "content.md"
                )
            )
            return target.read_text(encoding="utf-8")


__all__ = ["NaceDocumentLoader"]
