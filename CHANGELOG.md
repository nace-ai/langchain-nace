# Changelog

Notable changes per release. Versions before 0.2.0 have no changelog entry; the
[commit history](https://github.com/nace-ai/langchain-nace/commits/main) covers them.

## 0.2.0 - 2026-11-20

### Added — Nace Document Intelligence (NDI)

- `NaceDocumentLoader`: parse one local file (uploaded, then deleted) or public
  `https://` URL to Markdown, whole or chunked by page or section.
- `NaceExtractTool`: extract structured fields (a JSON Schema subset) from one
  document, with per-field status, confidence and quoted source text.
- `NaceSearchTool`: answer a question from an ingested workspace, with confidence
  and the evidence it cites (file, page, quote).
- Each component deletes the job it created — and the file it uploaded — after the
  read, also on failure; a failed clean-up is logged, not raised. The one exception
  is a `503 job_pending` on a local source: the uploaded file is kept, because the
  pending job and its replay need it, and the error message names the workspace and
  file and shows the `client.documents.delete_file(...)` call that deletes it.
- `timeout` (300 s by default) bounds the job (it starts after any local-file upload): the job's create request, its
  wait and a truncated-content download. The create's hold, every poll's HTTP timeout
  and the SDK retries all stop at the budget, and no poll is sent after it; past it
  the job is cancelled and `NaceAPITimeoutError` is raised. The bound is
  best-effort: a request already in flight, an SDK retry or the clean-up can run
  slightly past the deadline.
- After a `503` with code `job_pending` the raised error keeps the request's
  `idempotency_key` (`NaceAPIError.idempotency_key`): the job may still run, and
  resending the same body with that key resumes it instead of starting another.
- A succeeded job whose result is missing or empty raises
  `NaceAPIResponseValidationError` instead of returning empty output (no chunks,
  null extract data, an empty or missing answer).
- Error redaction covers the message, the upstream body kept on `.body`, the
  exception chain and the SDK's DEBUG wire log, so an echoed API key survives
  nowhere; the tools' pydantic validation errors do not echo the api key either.
- With `response_format="content"` both tools return the JSON answer only; the raw
  job result stays out of the model-facing content.

### Changed

- Requires `nace-sdk>=0.9.0` (from 0.3.0; the last release without this requirement
  is 0.1.5). On nace-sdk 0.9.0 an API redirect to a different scheme, host or port
  is refused — the call raises `NaceError` and nothing is sent to the redirect
  target — where earlier SDKs followed it with credentials, cookies and internal
  deployment headers stripped. Same-origin redirects still retain credentials.
  Point `NACE_BASE_URL` at the final origin, keep redirects same-origin, or pin
  `langchain-nace==0.1.5` together with an older `nace-sdk` (`nace-sdk==0.3.0` is known to follow these redirects) while you migrate.
