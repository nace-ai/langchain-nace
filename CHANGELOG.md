# Changelog

Notable changes per release. Versions before 0.2.0 have no changelog entry; the
[commit history](https://github.com/nace-ai/langchain-nace/commits/main) covers them.

## 0.3.0 - unreleased

### Added

- `NaceModelSelector` asks Drex Router (`POST /v1/route`) which candidate should
  answer a conversation. It does not call a chat model.
- `NaceModelSelectorMiddleware` and `ModelCandidate` select the model on the first
  model call of a run and store a `ModelSelection` in `nace_model_selection`.
  `before_agent` clears that key and makes no network call. A resumed run reuses
  the checkpointed selection while that model is still eligible and selects
  again otherwise.
- The route call is one attempt. Async code enforces `timeout` with
  `asyncio.timeout`. Sync code passes `timeout` to the SDK for each HTTP
  operation. Caller cancellation is raised again and does not call the handler.

### Changed

- Requires `nace-sdk>=0.11.1,<1.0.0` (from `>=0.9.0,<1.0.0`).

### Fixed

- A stored model is reused only when it is still eligible for the current request.
  A tool result that adds an image, a tool, or a longer prompt selects again.
- Image blocks include `input_image`, a block with an `image` key, and `media`
  or `file` blocks whose mime type starts with `image/`. A `response_format`
  counts as tools, including for the fallback.
- Profiles use the server limits: provider and tags are non-blank and at most 128
  characters, at most 64 tags, and ids, providers, and tags together at most
  100000 characters. Blank means no character outside JavaScript whitespace,
  including U+FEFF. `context` is at most 2^53 − 1.
- `httpx.HTTPStatusError` from a caller-supplied client uses the status table.
  Any other `httpx.HTTPError`, and any other exception from the exchange, is
  `connection_error`. An error or a response that arrives after the deadline is
  `timeout` on both the sync and async paths.
- A request id is stored only when it matches `req_` plus 8 to 64 letters or
  digits and does not contain a configured secret.
- The latest human message is kept, and the rest of the character budget is
  filled with the newest other turns first. The turns stay in chronological
  order. A human turn that is only a media placeholder is not text. If the
  latest human message has no real text, including an image-only turn, the
  reason is `no_input` in both context modes and nothing is sent.
- A stored selection is reused only when `model` is `None` or a configured
  candidate id that is still eligible. Any other shape is ignored.
- A caller cancellation propagates, including when it is still pending after
  another error from the route exchange. The handler is not called. An error
  from the handler propagates and is not retried on the fallback.

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
