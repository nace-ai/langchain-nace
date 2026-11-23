# Model selection replay benchmark

Compare `NaceModelSelector` with two candidate models against always-small, always-strong, and a length rule. Quality is exact-match accuracy. Cost and latency come from cached generations, so each arm is only a routing decision.

## Method

The public rows are MMLU (`cais/mmlu`, config `all`, split `test`) and GSM8K (`openai/gsm8k`, config `main`, split `test`). Both are MIT licensed. The script samples 150 rows from each with `random.Random(20261124)`, sorts them, then makes a seeded 30/120 dev/test split. `--limit N` keeps the first N rows of each split.

Task text is fetched from the public Hugging Face datasets-server rows API (no key, at most 100 rows per request, with backoff on 429 and 5xx) and stored only under `.cache/`, which is git-ignored. Committed outputs contain dataset names, row indexes, and numeric measures.

Each test row is prompted with one fixed system message and one human message. MMLU lists choices A-D. A row is correct when the last `ANSWER:` line matches the key: the MMLU letter, or the GSM8K number after `####`, compared as `Decimal` after stripping `$`, `,`, `%`, spaces, and one trailing `.0`. No `ANSWER:` line is wrong.

Generation goes to OpenRouter chat completions (`OPENROUTER_API_KEY`) at temperature 0 and `max_tokens` 512, asking for `usage.cost`. Each model is run `REPEATS = 2` times. Dev rows are not sent to models. They only set L, the median human-message length. The rule sends a row to the strong model when the human message is longer than L or contains a code fence.

Arms, per pool:

- Pool A: `meta-llama/llama-3.1-8b-instruct` and `anthropic/claude-sonnet-5.5`.
- Pool B: `openai/gpt-4o-mini` and `anthropic/claude-sonnet-5.5`.
- `fixed_small` and `fixed_strong`.
- `rule`, as above.
- `selector/<priority>/<context>` for priority in `balanced`, `quality`, `cheapest` and context in `conversation`, `latest_human`. Each one calls `NaceModelSelector.select` with `candidates={small: None, strong: None}`, `fallback=None`, and that priority and context. The key is the package default, `NACE_API_KEY`. A missing selection or a fallback uses the strong model (the agent's own model) and counts in the fallback rate.
- `decision`. This calls `NaceModelRouterMiddleware.before_agent`, the same method the middleware uses to ask its classifier which of two routes to take. The routes are `small` ("simple question a small model answers correctly") and `strong` ("hard question that needs a strong model"). The instructions are "Choose the model that should answer the question." Escalation uses the middleware default (the last route, strong, below confidence 0.6). The chat models passed in are not called.
- `random_at_same_share`: for each routed arm, the linear mix of fixed-small and fixed-strong at that arm's strong share. Quality gain is the arm's accuracy minus that mix.
- `oracle`: on each repeat, the strong model only when the small model is wrong and the strong model is right.

Reported routing cost is an estimate. The route call does not return a token count, so the estimate uses the small model's prompt-token count for that row (the character upper bound when that count is missing): $0.02 per 1M tokens for the selector and $0.06 per 1M for the decision. End-to-end latency is route latency plus the chosen model's generation latency. Bootstrap intervals are paired over tasks, 2,000 resamples, seed 20261124.

## How to run

From the repo root, with the project environment installed (this benchmark imports `langchain_nace` and the experimental middleware):

```bash
export OPENROUTER_API_KEY=...
export NACE_API_KEY=...
uv run python benchmarks/model_selection/run_benchmark.py --stage all
```

`--stage generate` only fills the generation cache. `--stage route` only fills routing decisions. `--stage report` rewrites `results/` from the cache and does not call the network. `--limit 1` is a smoke run: the first dev row and the first test row of each dataset.

`results/RESULTS.md` is rendered from `results/summary.json`. Do not edit the numbers by hand.

## Cost cap

`--budget-usd` defaults to 4.50 and rejects any value above 5.00. One `SpendLedger` owns the cap. It is thread-safe. The append-only file is `benchmarks/model_selection/.cache/spend.jsonl`. Its total counts across runs, so a rerun cannot spend more than the cap in total.

Before any paid call, the script adds the worst case of every call that is not already cached: prompt tokens at `len(prompt characters) / 2`, completion tokens at 512, priced from the table below, plus the same upper bound for routing. If the ledger total plus that worst case is over the budget, it prints the three amounts and exits non-zero without a paid call.

Before each paid call it reserves that worst case. If spent + reserved + this call would exceed the budget, it schedules nothing further, finishes calls already reserved, writes `results/` with `"complete": false`, and exits non-zero. After a call it replaces the reservation with the billed cost (`usage.cost`, otherwise the price table). The ledger always records the billed cost. If a billed cost does not fit under the budget, the run is marked `overrun` and incomplete. A failed call that returns no usage is charged its worst-case reservation. A paid generation call is retried at most twice on 429 and 5xx, and is not retried on a timeout.

The price table, USD per 1M tokens (input/output), is `meta-llama/llama-3.1-8b-instruct` 0.02/0.04, `openai/gpt-4o-mini` 0.15/0.60, `anthropic/claude-sonnet-5.5` 2.00/10.00.

Generation uses 6 worker threads. Routing uses 4. Cached generations are keyed by model, dataset, row, and repeat. Cached decisions are keyed by pool, arm, dataset, and row. Keys are read from the environment only and are not printed, logged, or written.

## Packaging

`pyproject.toml` builds the wheel from the `langchain_nace` package only, and the sdist includes `langchain_nace`, `README.md`, `LICENSE`, and `THIRD_PARTY_NOTICES.md`. `benchmarks/` is not in either. `make lint` checks `langchain_nace`, `tests`, `examples`, and `scripts`, and formats `langchain_nace` only. It does not list this folder. The unit tests under `tests/` are in that lint path.

## Limits

Of the method:

- Replay method instead of the full `create_agent` path.
- Only two task families (no code tasks with unit tests, no multi-turn tool tasks). Each task is one turn, so `conversation` and `latest_human` differ only by the system message.
- 2 repeats not 3.
- It is not known whether the Router was trained on these public sources.
- Small sample: 240 test rows. The intervals in `results/RESULTS.md` show the uncertainty.
- A completion that is cut off at 512 tokens, or stopped by a provider filter, before its `ANSWER:` line is scored as wrong.
- The reported routing cost is an estimate. It does not include the candidate list or the decision question that Drex also encodes.
- Each pool makes its own decision call, although the decision request is the same for both pools.

Of the cost cap:

- Run one process at a time. Two processes do not share reservations.
- The retries of one call share one reservation. A provider that bills a request and then returns 5xx is not counted.
- A ledger line is written after the call returns. If the process is killed between the provider's reply and that write, the spend is not in the ledger.
- The prompt-token bound (characters / 2) and the price table are assumptions. OpenRouter can route a model to a provider with a higher price than the table. The billed cost is what the ledger records.
- A selector fallback (for example a timeout) is cached as that arm's decision and counts in the fallback rate. Delete the file under `.cache/routes/` to ask again.
