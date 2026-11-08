<p align="center">
  <a href="https://nace.ai"><picture><source media="(prefers-color-scheme: dark)" srcset="https://console.nace.ai/brand/nace-logo-white.svg"><img src="https://console.nace.ai/brand/nace-logo-black.svg" alt="Nace" height="40"></picture></a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.langchain.com/langchain"><picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6a9942640271f297cc0bd30b_LangChain_OSS%20Lockup_light%201-1.svg"><img src="https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6a994264a64b3afcdd880f66_LangChain_OSS%20Lockup_light%201.svg" alt="LangChain" height="40"></picture></a>
</p>

<h1 align="center">Drex for LangChain</h1>

<p align="center">
  <a href="https://github.com/nace-ai/langchain-nace/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/nace-ai/langchain-nace/ci.yml?branch=main&amp;label=CI&amp;logo=github" alt="CI status"></a>
  <a href="https://github.com/nace-ai/langchain-nace/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

LangChain integration for [Drex](https://console.nace.ai/docs), Nace's decision models. Drex reads text, JSON, LangChain messages and (on Drex Max) images, and returns typed answers with calibrated probabilities instead of generated text. Use it to route requests, pick models, or check tool calls before they run.

```bash
pip install langchain-nace                  # classifier, router, LangGraph edge
pip install "langchain-nace[experimental]"  # + agent middleware
export NACE_API_KEY=nace_sk_...
```

## Quickstart

```python
from langchain_nace import Choice, NaceClassifier, Noul, Score

classifier = NaceClassifier()  # model="drex-max" by default

response = classifier.invoke(
    {
        "state": "I was charged twice for March and I'm annoyed. Please refund one.",
        "questions": {
            "refund": Noul(instructions="Is the customer asking for a refund?"),
            "team": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": "Payments and refunds", "technical": "Bugs"},
            ),
            "mood": Score(
                instructions="How upset is the customer?", criteria=["calm", "annoyed", "furious"]
            ),
        },
    }
)
response.nouls["refund"].noul  # 0.98: probability of yes
response.choices["team"].probabilities  # {"billing": 0.99, "technical": 0.01}
response.scores["mood"].score  # 1.1: expected level
```

All questions about one state go in one request. Answers keep their full probabilities, so thresholds stay in your code. `ainvoke`, `batch` and `abatch` work as on any Runnable. `batch` runs at most 8 requests at once unless you set `max_concurrency`.

Batches stop new SDK dispatch after an unrecovered failure. With
`return_exceptions=True`, only HTTP 401/402/403 stop the batch; unsent inputs return
`NaceBatchStoppedError`. Calls already handed to the SDK finish, including their
SDK retries. Cancelled inputs fail individually. Subclass `invoke`/`ainvoke`
recovery runs before the stop decision, and callbacks report that final outcome.


**Switching from `langchain-typesafe`:** change `langchain_typesafe` to `langchain_nace` in your imports, `TypeSafeClassifier` to `NaceClassifier`, and set `NACE_API_KEY` and, for your own endpoint, `NACE_BASE_URL` (OpenRouter uses `OPENROUTER_API_KEY`; see Backends). Everything else keeps its name and arguments:

| `langchain-typesafe` | `langchain-nace` |
|---|---|
| `TypeSafeClassifier` | `NaceClassifier` (default model `drex-max`) |
| `Noul`, `Choice`, `Score`, `NoulCriteria`, `ClassifierRequest`, `ClassifierResponse` | same names; `nouls`, `choices`, `scores`, `usage`, `request_id` as before |
| `experimental.middleware.ModelRouterMiddleware`, `ModelChoice` | same names (also available as `NaceModelRouterMiddleware`) |
| `experimental.middleware.AutoModeMiddleware(tools=, instructions=, criteria=)` | same names and arguments (also available as `NaceAutoModeMiddleware`) |
| `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL` | `NACE_API_KEY`, `NACE_BASE_URL` |
| `TypeSafe*Error` | `Nace*Error`; both inherit LangChain's `Model*Error` classes |
| `client=`, `async_client=` (`httpx2`) | `http_client=`, `http_async_client=` (`httpx`) |

## Models

The default is **`drex-max`**: it reads images, and takes up to 16 questions per request. For text-only work with more questions per request (up to 512), use `NaceClassifier(model="drex-v2.0")`. Any other model id is accepted and sent as given (`drex-v1.5`, `drex-latest`, and models released later); Drex enforces each model's limits, and errors tell you what to change. Pin a default for your app with `model=` or `NACE_DEFAULT_DECISION_MODEL`. The package default moves to the newest flagship with each Drex release, so pin a model if you need results that never change.

## Images (Drex Max)

```python
max_ = NaceClassifier()  # drex-max
max_.invoke(
    {
        "state": "Two product photos.",
        "images": [red_png_bytes, "data:image/jpeg;base64,..."],
        "questions": {
            "damaged": Choice(
                instructions="Which image shows damage?",
                criteria={"Image 1": None, "Image 2": None},
            )
        },
    }
)
```

Images can be bytes, paths, base64, data URLs, LangChain image blocks (`{"type": "image", "base64": ...}`, `source_type: "base64"`, `image_url`), or files uploaded with `nace_sdk` (`{"type": "workspace_file", ...}`). Images inside LangChain messages in `state`, whether message objects or `{"role": ..., "content": [...]}` dicts, are moved to `images` automatically and referred to as "Image N". Drex doesn't fetch URLs, so remote image URLs are refused. Sending images to a text-only model (`drex-v2.0`, `drex-v1.5`) raises `NaceImageModelError` before anything is sent; images are never dropped silently.

## Routing

```python
from langchain_nace import NaceRouter

router = NaceRouter(
    classifier=classifier,
    question=Choice(instructions="Which team?", criteria={"billing": None, "technical": None}),
    min_confidence=0.6,
    fallback="human",
)
chain = router.branch(
    {"billing": billing_chain, "technical": tech_chain, "human": escalate}, default=escalate
)
```

LangGraph: `nace_conditional_edge(classifier, question, state=..., min_confidence=0.7)` is a path function for `add_conditional_edges`. Below the threshold it returns `"human_review"`, so that node can `interrupt()` for a person. See [`examples/langgraph_router.py`](https://github.com/nace-ai/langchain-nace/blob/main/examples/langgraph_router.py).

## Agent middleware (experimental)

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_nace.experimental.middleware import (
    ModelChoice,
    NaceAutoModeMiddleware,
    NaceModelRouterMiddleware,
)

router = NaceModelRouterMiddleware(
    choices={
        "fast": ModelChoice("openai:gpt-5-mini", "Short lookups"),
        "strong": ModelChoice("anthropic:claude-opus-5-5", "Multi-step changes"),
    },
    instructions="Pick the cheapest model that can do this safely.",
)
gate = NaceAutoModeMiddleware(screenshots=True)  # drex-max reads the screenshots
agent = create_agent(
    "openai:gpt-5-mini", tools=tools, middleware=[router, gate], checkpointer=InMemorySaver()
)
```

**`NaceModelRouterMiddleware`** picks the model once per run. Below `min_confidence` (0.6), or if Drex can't be reached, it uses `escalate_to` (the last choice by default). The decision and probabilities are kept in state under `nace_model_route`.

**`NaceAutoModeMiddleware`** checks each tool call before it runs. In one request Drex reads:
- the user's **original request**, kept however long the run gets;
- the latest user message and the last 20 messages;
- the tool call and its description;
- with `screenshots=True` and Drex Max, the newest screenshots.

It answers three questions: a risk score, "destructive?" and "did the user ask for this?". The default `ThresholdPolicy` then decides:
- **block** at risk ≥ 0.75, or when the call is destructive and the user didn't ask for it;
- **ask a person** at risk ≥ 0.4, or when the call is destructive and the user asked for it;
- **allow** otherwise.

Asking uses a LangGraph `interrupt`. Resume it with `{"type": "approve"}`, `{"type": "reject", "message": ...}` or `{"type": "edit", "args": {...}}`. If Drex fails, the call is blocked (`on_error="block"`). Questions, thresholds and the policy are all replaceable. Write your own `GatePolicy`, or subclass `GateDecision`.

With a checkpointer, each gated call adds one checkpoint write: a JSON decision
record from the `NaceGateDecision` task, visible in traces and `updates` streams.
It contains the call id, a SHA-256 digest of the prepared wire input (including
screenshots), the validated policy response as JSON text, the decision kind, and
the gate scope and number of gates seen for that call. Task inputs
and outputs contain no screenshot bytes or credentials. A normal resume, including
on another worker, replays this decision and never asks Drex again. It compares
the current input digest and decision kind before consuming the person's answer;
if either changed, the call is blocked with a request for a new tool call.

Single-call messages keep the resume formats above. Whenever the latest AIMessage
has multiple tool calls, every approval answer requires `tool_call_id`, including
calls dispatched through `Send` or after middleware appends another message. For
example: `{"type": "approve", "tool_call_id": "call_b"}`. The interrupt sets
`requires_tool_call_id=True`. Missing or mismatched ids block the call. Interrupts
also include `nace_gate_id` and `nace_gate_count`; clients may echo the id for
explicit gate binding. Saved scope/count mismatches require a new tool call.

Each gate decides at most once per tool call id within a task. Repeated invocations
reuse that decision and the person's first approval/rejection/edit outcome, without
another Drex request or interrupt. Put the gate **innermost (closest to the tool)**,
after any middleware that rewrites tool calls, and **before retry middleware**:
`[argument_transformer, gate, ToolRetryMiddleware(...)]`. At actual tool invocation,
the name, id, and arguments must still match the approved call, including any edits
made by the person; a mismatch requires a new tool call. If a retry wraps the gate, it still
reuses the first decision or raised failure; it cannot obtain a different approval
for that same call. Separate gates have separate approval scopes, so outer Allow and
inner AskHuman work together. A final check at ToolNode's actual tool invocation
requires every saved approval scope to have been reached and its answer consumed.
Removing a gate therefore cannot reuse an earlier gate's Allow. Opaque custom
handlers fail closed when another approval scope cannot be verified.
Changing the middleware stack while approvals are paused is unsupported, including
adding, removing, or reordering gates or other interrupting middleware. Re-request
pending approvals after deploying such a change.
Task records remain checked against the call id;
a concurrent sibling's record blocks rather than authorizing another call.

The SHA-256 digest compares both the prepared Drex wire input (including screenshots,
questions and model) and the entire `request.state`, `runtime.context`, and reachable
runtime `configurable` values. It excludes configurable keys starting with `__`
and the bookkeeping keys `thread_id`, `checkpoint_id`, `checkpoint_ns`,
`checkpoint_map`, and `run_id`. These values are compared locally, not sent to Drex
or included in decision records. JSON encoding uses sorted keys and compact
separators. Mappings encode sorted key/value pairs with recursively type-tagged
keys and values, so integer and string keys cannot collide. Sequences preserve
their type and order; sets and frozensets preserve their type and sort their
recursively encoded members, making their comparison independent of iteration order. Pydantic objects include every declared field directly,
regardless of export/exclude settings, plus instance attributes, private attributes,
computed fields, and extra fields. Dataclasses include every field regardless of `repr`/`compare`
flags. `SecretStr`/`SecretBytes` use the underlying value
only in the local hash. Other objects use `repr`; if that representation
changes across workers, the old approval blocks. Changes through
`Command(resume=..., update=...)` or a different runtime context also block with
"request a new tool call".

The `experimental` extra requires **LangGraph >=1.2.16**. Earlier versions treat an
explicit checkpoint id as replay and can discard saved task results during an
approval resume, so every approval would be blocked.

Policies receive only the asked, typed answers and declared numeric usage fields;
raw responses, response extensions and server request ids are omitted. The model
name and score legends come from the configured request. Probability maps contain
only offered labels/levels, and future answer types retain only their asked type.

Editing state with `update_state` or forking a checkpoint re-checks as a new call.
An answer already pending for the previous decision cannot approve or reject the
new one, even if its new verdict would allow it. Request the call again to obtain
a new approval. Custom `GateDecision` subclasses need a stable, nonempty `id`
covering all configuration that changes their behavior to resume after a pause.
Policies must be deterministic for a given response. Screenshot callbacks run
again to compare inputs during replay and must not call `task()` or `interrupt()`.

`on_error="block"` saves a failed check as a block and replays it without asking
Drex or capturing screenshots again. `on_error="raise"` and cancellation escape
the decision task without saving a result; retrying the failed run checks again.
Python permits outer middleware to catch task errors, so a caught error alone does
not necessarily fail the run. Within one task, a repeated invocation re-raises that
same failure; restarting the failed graph run performs a new check. The gate never
searches later task positions for a saved decision after a failure.

In 0.1.5, score legends must match the length of the rubric asked, every asked
question must be answered with its own type, and choices must use offered labels.
Malformed answers block the gate through `on_error`. **Approvals paused by
0.1.4 or earlier must be re-requested after upgrading to 0.1.5.**

[`examples/support_agent.py`](https://github.com/nace-ai/langchain-nace/blob/main/examples/support_agent.py) runs four live scenarios: an allowed read, a blocked unrequested delete, a person approving a requested delete, and a screenshot-based block on a computer-use click.

## Backends

| Backend | `base_url` | Notes |
|---|---|---|
| Nace (default) | `https://console.nace.ai` | `NACE_BASE_URL` |
| OpenRouter | `https://openrouter.ai/api/v1` | models `nace-ai/drex-v2.0`, `nace-ai/drex-max`; images are sent the way OpenRouter expects |
| Vercel AI Gateway / TypeSafe-compatible hosts | the host's System One root | same request format as Nace |

Every request sends `User-Agent` and `X-Nace-Client: langchain-nace/<version>`. On OpenRouter it also sends `HTTP-Referer` and `X-OpenRouter-Title`.

`NACE_API_KEY` and `NACE_BASE_URL` configure the endpoint and its key, including
on-prem deployments and local HTTP servers. No code change or saved profile is required:

```bash
export NACE_API_KEY=nace_sk_...
export NACE_BASE_URL=https://nace.customer.example
```

```python
classifier = NaceClassifier()
```

Explicit `api_key` and `base_url` arguments take precedence. Legacy `DREX_API_KEY`
and `DREX_BASE_URL` work when the corresponding Nace variable is unset or blank.
The configured endpoint receives the selected key.

For the exact `openrouter.ai` host, the adapter uses `OPENROUTER_API_KEY` instead
of inheriting `NACE_API_KEY`. You can also pass that provider's key explicitly:

```python
classifier = NaceClassifier(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
```

The key follows the endpoint, even when `openrouter=` overrides the request format.
Serialized classifiers pin the endpoint and refer to `OPENROUTER_API_KEY` for
OpenRouter, or `NACE_API_KEY` for other endpoints; the key itself is not serialized.

Same-origin redirects retain credentials when the supplied HTTP client follows
redirects. A redirect to a different scheme, host or port strips authentication,
cookies and internal deployment headers. Internal deployment credentials are also
excluded from every endpoint except the production HTTPS hosts
`console.nace.ai` and its legacy alias `drex.nace.ai` on port 443.

Version 0.1.4 restores environment-based custom endpoint configuration, which
0.1.3 incorrectly restricted to Nace cloud hosts. It retains the provider-specific
OpenRouter credential policy and fixes cross-origin credential forwarding.
Versions 0.1.0–0.1.2 could inherit a Nace key for OpenRouter and forward internal
deployment credentials to other endpoints. If real credentials reached an
unintended destination, replace and revoke those credentials; upgrading cannot
revoke them or remove copies held elsewhere. Recreate old OpenRouter serialized
classifiers with the correct provider secret before reloading them.

## Errors

Errors also inherit LangChain's `Model*Error` classes. Displayed summaries are bounded
and redact configured API keys and bearer tokens. SDK response validation errors omit
response-supplied field paths; `.body` still holds the upstream payload for inspection:

| Error | When |
|---|---|
| `NaceAuthenticationError` | 401 |
| `NaceInsufficientCreditError` | 402, with the Billing link |
| `NaceUnprocessableEntityError` | 422; lists each issue's `path: message` |
| `NaceRateLimitError` | 429, after retries |
| `NaceOverloadedError` | 529, after retries |
| `NaceAPIConnectionError` / `NaceAPITimeoutError` | network problems |

429, 529 and 5xx responses are retried (`max_retries=2`), waiting as long as `retry-after-ms` asks. The timeout is 60 s, because Drex may take up to 55 s.

## Tracing

Calls show up in LangSmith as `llm` runs (`ls_provider="nace"`) with the questions, the full answers and probabilities, token usage and the Drex request id. The router and middlewares add their thresholds to the run metadata.

## Development

```bash
uv sync --all-extras --all-groups
make test lint typecheck                 # offline: fake Drex server
NACE_API_KEY_LIVE=nace_sk_... make integration_tests   # live contract tests
```
