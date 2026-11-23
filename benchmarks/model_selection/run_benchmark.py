"""Replay benchmark for NaceModelSelector against fixed models and a length rule.

Dev rows set the rule threshold only. Test rows are generated and routed.
The spend ledger is append-only and is counted across runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

import httpx
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_nace.experimental.middleware.model_router import (
    ModelChoice,
    NaceModelRouterMiddleware,
)
from langchain_nace.selection import NaceModelSelector

SEED = 20261124
REPEATS = 2
MAX_TOKENS = 512
TEMPERATURE = 0
GENERATE_WORKERS = 6
ROUTE_WORKERS = 4
DEFAULT_BUDGET_USD = Decimal("4.50")
MAX_BUDGET_USD = Decimal("5.00")
SAMPLE_SIZE = 150
DEV_SIZE = 30
TEST_SIZE = SAMPLE_SIZE - DEV_SIZE
REQUEST_TIMEOUT_S = 60.0
PAID_ATTEMPTS = 3
FETCH_ATTEMPTS = 8
FETCH_BACKOFF_S = 10.0
FETCH_PAUSE_S = 0.5
BOOTSTRAP_SAMPLES = 2000
SELECTOR_USD_PER_MILLION = Decimal("0.02")
DECISION_USD_PER_MILLION = Decimal("0.06")
MILLION = Decimal(1_000_000)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
MMLU_ROWS = 14_042
GSM8K_ROWS = 1_319
SYSTEM_MMLU = "Reason briefly, then end with a line ANSWER: <letter>."
SYSTEM_GSM8K = "Reason briefly, then end with a line ANSWER: <number>."
DECISION_INSTRUCTIONS = "Choose the model that should answer the question."
DECISION_FUNCTION = "NaceModelRouterMiddleware.before_agent"
SMALL_CRITERIA = "simple question a small model answers correctly"
STRONG_CRITERIA = "hard question that needs a strong model"
LOCAL_REASONS = frozenset({"no_input", "single_candidate", "no_eligible_candidate"})
PRIORITIES = ("balanced", "quality", "cheapest")
CONTEXTS = ("conversation", "latest_human")
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache"
RESULTS_DIR = HERE / "results"

# USD per 1M tokens, input then output. Worst-case bound and fallback price.
PRICES_USD_PER_MILLION: dict[str, tuple[Decimal, Decimal]] = {
    "meta-llama/llama-3.1-8b-instruct": (Decimal("0.02"), Decimal("0.04")),
    "openai/gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
    "anthropic/claude-sonnet-5.5": (Decimal("2.00"), Decimal("10.00")),
}
POOLS: dict[str, dict[str, str]] = {
    "A": {
        "small": "meta-llama/llama-3.1-8b-instruct",
        "strong": "anthropic/claude-sonnet-5.5",
    },
    "B": {
        "small": "openai/gpt-4o-mini",
        "strong": "anthropic/claude-sonnet-5.5",
    },
}
DATASETS: dict[str, dict[str, Any]] = {
    "mmlu": {
        "hf": "cais/mmlu",
        "config": "all",
        "split": "test",
        "rows": MMLU_ROWS,
    },
    "gsm8k": {
        "hf": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "rows": GSM8K_ROWS,
    },
}

CSV_FIELDS = (
    "record_type",
    "dataset",
    "row",
    "split",
    "repeat",
    "model",
    "correct",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "latency_s",
    "provider",
    "finish_reason",
    "pool",
    "arm",
    "chosen",
    "route_latency_s",
    "fallback",
    "reason",
    "chosen_probability",
    "routing_cost_usd",
    "routing_tokens",
)


class BudgetError(Exception):
    """Pre-flight worst case does not fit in the remaining budget."""

    def __init__(self, spent: Decimal, worst: Decimal, budget: Decimal) -> None:
        """Store the three amounts. The message is numbers only."""
        self.spent = spent
        self.worst = worst
        self.budget = budget
        super().__init__(f"spent_usd={spent} worst_case_usd={worst} budget_usd={budget}")


class CallSkippedError(Exception):
    """The paid request was not sent. The reservation is released."""


class LedgerError(Exception):
    """The spend ledger cannot be read safely."""


class TransportError(Exception):
    """A transport error that carries a type name and an optional HTTP status."""

    def __init__(self, kind: str, status: int | None = None) -> None:
        """Store the type name and status. Do not keep the original message."""
        self.kind = kind
        self.status = status
        super().__init__(kind)


@dataclass(frozen=True)
class HttpResult:
    """Status and a JSON object. The body is None when it is not an object."""

    status: int
    body: dict[str, Any] | None


@dataclass(frozen=True)
class Task:
    """One public question. Text stays in the git-ignored cache."""

    dataset: str
    row: int
    split: str
    question: str
    choices: tuple[str, ...]
    gold: str


@dataclass(frozen=True)
class Outcome:
    """One cached generation. Numbers and a short provider name only."""

    correct: bool
    prompt_tokens: int
    completion_tokens: int
    cost_usd: Decimal
    latency_s: Decimal
    provider: str
    finish_reason: str


@dataclass(frozen=True)
class RouteChoice:
    """One cached routing decision."""

    chosen: str
    fallback: bool
    reason: str
    probabilities: dict[str, float]
    latency_s: Decimal | None
    routing_cost_usd: Decimal
    routing_tokens: Decimal


@dataclass
class Job:
    """One paid call. `fn` returns the billed cost or raises CallSkippedError."""

    kind: str
    model: str
    worst: Decimal
    fn: Callable[[], Decimal]


class SpendLedger:
    """Append-only spend file. In memory, spent + reserved never exceeds the budget.

    `spent` is the sum of the ledger file. `reserved` is the worst case of calls
    that have been allowed and have not been committed yet. A commit replaces that
    call's reservation with its billed cost. The ledger always records the billed
    cost. If a billed cost is higher than its reservation and no longer fits, the
    `overrun` flag is set and the run stops as incomplete.
    """

    def __init__(self, path: Path, budget: Decimal) -> None:
        """Load `path` and take `budget` as a hard cap, including prior runs."""
        self.path = path
        self.budget = budget
        self._lock = threading.Lock()
        self._spent = Decimal(0)
        self._reserved = Decimal(0)
        self._peak = Decimal(0)
        self.overrun = False
        self._load()

    @property
    def spent(self) -> Decimal:
        """Sum of committed ledger lines."""
        with self._lock:
            return self._spent

    @property
    def peak(self) -> Decimal:
        """Highest spent + reserved observed under the lock."""
        with self._lock:
            return self._peak

    def preflight(self, worst: Decimal) -> None:
        """Refuse before any call when committed spend plus `worst` exceeds the cap."""
        with self._lock:
            if self._spent + worst > self.budget:
                raise BudgetError(self._spent, worst, self.budget)

    def try_reserve(self, worst: Decimal) -> bool:
        """Hold `worst` against the cap. False means this call must not start."""
        with self._lock:
            if worst < 0 or self._spent + self._reserved + worst > self.budget:
                return False
            self._reserved += worst
            self._note_peak()
            return True

    def release(self, reserved: Decimal) -> None:
        """Drop a reservation for a call that was not sent."""
        with self._lock:
            self._reserved -= reserved
            if self._reserved < 0:
                self._reserved = Decimal(0)

    def commit(
        self,
        *,
        reserved: Decimal,
        cost: Decimal,
        kind: str,
        model: str,
    ) -> Decimal:
        """Replace `reserved` with `cost` and append one ledger line."""
        with self._lock:
            self._reserved -= reserved
            if self._reserved < 0:
                self._reserved = Decimal(0)
            room = self.budget - self._spent - self._reserved
            if cost < 0:
                cost = Decimal(0)
            if cost > room:
                self.overrun = True
            charge = cost
            self._spent += charge
            self._note_peak()
            self._append(kind=kind, model=model, cost=charge)
            return charge

    def _note_peak(self) -> None:
        total = self._spent + self._reserved
        if total > self._peak:
            self._peak = total

    def _load(self) -> None:
        if not self.path.exists():
            return
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                cost = Decimal(str(item["cost"]))
            except (json.JSONDecodeError, KeyError, InvalidOperation) as exc:
                message = f"spend ledger line {number} is unreadable"
                raise LedgerError(message) from exc
            if cost < 0:
                message = f"spend ledger line {number} is unreadable"
                raise LedgerError(message)
            self._spent += cost
        self._note_peak()

    def _append(self, *, kind: str, model: str, cost: Decimal) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "time": datetime.now(UTC).isoformat(),
            "kind": kind,
            "model": model,
            "cost": format(cost, "f"),
        }
        line = json.dumps(payload, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _status_of(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def print_failure(exc: BaseException) -> None:
    """Print the exception type and HTTP status, never the message."""
    status = _status_of(exc)
    if isinstance(exc, TransportError):
        status = exc.status
        name = exc.kind
    else:
        name = type(exc).__name__
    if status is None:
        print(f"call failed: {name}", file=sys.stderr)
    else:
        print(f"call failed: {name} status={status}", file=sys.stderr)


def _money(value: Decimal | int | str) -> Decimal:
    return Decimal(value)


def _json_object(response: httpx.Response) -> dict[str, Any] | None:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(body, dict):
        return body
    return None


class HttpxTransport:
    """httpx transport. Exceptions are re-raised without their messages attached."""

    def __init__(self) -> None:
        """Open one client for the process."""
        self._client = httpx.Client()

    def get_json(self, url: str, timeout: float) -> HttpResult:
        """GET a JSON object."""
        return self._send("GET", url, None, None, timeout)

    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout: float,
    ) -> HttpResult:
        """POST a JSON object."""
        return self._send("POST", url, dict(headers), dict(body), timeout)

    def _send(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None,
        body: dict[str, Any] | None,
        timeout: float,
    ) -> HttpResult:
        try:
            response = self._client.request(
                method,
                url,
                headers=headers,
                json=body,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise TransportError(type(exc).__name__) from None
        except httpx.HTTPError as exc:
            raise TransportError(type(exc).__name__, _status_of(exc)) from None
        return HttpResult(response.status_code, _json_object(response))


def _retryable(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def with_retries(
    call: Callable[[], HttpResult],
    sleep: Callable[[float], None],
    attempts: int,
    *,
    retry_timeouts: bool = True,
    backoff: float = 1.0,
) -> HttpResult:
    """Retry 429, 5xx, and timeouts. `attempts` includes the first try.

    A paid call passes `retry_timeouts=False`: the provider can bill a request
    that timed out on this side, and one reservation covers one billed request.
    """
    last: HttpResult | None = None
    for attempt in range(attempts):
        try:
            result = call()
        except TransportError as exc:
            if retry_timeouts and "Timeout" in exc.kind and attempt + 1 < attempts:
                sleep(min(backoff * 2**attempt, 120.0))
                continue
            raise
        if _retryable(result.status) and attempt + 1 < attempts:
            last = result
            sleep(min(backoff * 2**attempt, 120.0))
            continue
        return result
    if last is None:
        raise TransportError("retry")
    return last


def sample_indices(population: int, count: int, rng: random.Random) -> list[int]:
    """Return `count` sorted indices from `range(population)`."""
    return sorted(rng.sample(range(population), count))


def assign_splits(limit: int | None = None) -> dict[str, dict[str, list[int]]]:
    """Sample both datasets and split 30 dev / 120 test, then apply `limit`."""
    rng = random.Random(SEED)  # noqa: S311
    chosen = {
        name: sample_indices(spec["rows"], SAMPLE_SIZE, rng) for name, spec in DATASETS.items()
    }
    assigned: dict[str, dict[str, list[int]]] = {}
    for name, indices in chosen.items():
        order = list(indices)
        rng.shuffle(order)
        dev = sorted(order[:DEV_SIZE])
        test = sorted(order[DEV_SIZE:])
        if limit is not None:
            dev = dev[:limit]
            test = test[:limit]
        assigned[name] = {"dev": dev, "test": test}
    return assigned


def index_batches(indices: Sequence[int], page: int = 100) -> list[tuple[int, int]]:
    """Cover `indices` with (offset, length) windows of at most `page` rows."""
    ordered = sorted(set(indices))
    if not ordered:
        return []
    batches: list[tuple[int, int]] = []
    start = ordered[0]
    previous = ordered[0]
    for index in ordered[1:]:
        if index - start + 1 > page:
            batches.append((start, previous - start + 1))
            start = index
        previous = index
    batches.append((start, previous - start + 1))
    return batches


def _rows_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{HF_ROWS_URL}?{query}"


def parse_row_page(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Map row index to the dataset row object."""
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TransportError("rows")
    found: dict[int, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        index = item.get("row_idx")
        row = item.get("row")
        if isinstance(index, int) and not isinstance(index, bool) and isinstance(row, dict):
            found[index] = row
    return found


def _gold_gsm(answer: str) -> str:
    marker = "####"
    if marker in answer:
        return answer.split(marker, 1)[1].strip()
    return answer.strip()


def task_from_row(dataset: str, row: int, split: str, raw: Mapping[str, Any]) -> Task:
    """Build a task from one dataset row."""
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise TransportError("question")
    if dataset == "mmlu":
        choices = raw.get("choices")
        answer = raw.get("answer")
        if (
            not isinstance(choices, list)
            or len(choices) != 4
            or not all(isinstance(item, str) for item in choices)
            or isinstance(answer, bool)
            or not isinstance(answer, int)
            or not 0 <= answer <= 3
        ):
            raise TransportError("mmlu")
        return Task(dataset, row, split, question, tuple(choices), "ABCD"[answer])
    answer_text = raw.get("answer")
    if not isinstance(answer_text, str):
        raise TransportError("gsm8k")
    return Task(dataset, row, split, question, (), _gold_gsm(answer_text))


def fetch_dataset_rows(
    dataset: str,
    indices: Sequence[int],
    transport: Any,
    sleep: Callable[[float], None],
) -> dict[int, dict[str, Any]]:
    """Fetch `indices` from the public datasets-server rows API."""
    spec = DATASETS[dataset]
    found: dict[int, dict[str, Any]] = {}
    for offset, length in index_batches(indices):
        url = _rows_url(spec["hf"], spec["config"], spec["split"], offset, length)

        def call(url: str = url) -> HttpResult:
            return transport.get_json(url, REQUEST_TIMEOUT_S)

        result = with_retries(call, sleep, FETCH_ATTEMPTS, backoff=FETCH_BACKOFF_S)
        if result.status != 200 or result.body is None:
            raise TransportError("fetch", result.status)
        found.update(parse_row_page(result.body))
        sleep(FETCH_PAUSE_S)
    return found


def _task_path(cache: Path) -> Path:
    return cache / "tasks.json"


def _write_tasks(path: Path, tasks: Sequence[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "dataset": task.dataset,
            "row": task.row,
            "split": task.split,
            "question": task.question,
            "choices": list(task.choices),
            "gold": task.gold,
        }
        for task in tasks
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_tasks(path: Path) -> list[Task]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[Task] = []
    if not isinstance(raw, list):
        return tasks
    for item in raw:
        if not isinstance(item, dict):
            continue
        choices = item.get("choices")
        if not isinstance(choices, list):
            continue
        tasks.append(
            Task(
                dataset=str(item["dataset"]),
                row=int(item["row"]),
                split=str(item["split"]),
                question=str(item["question"]),
                choices=tuple(str(choice) for choice in choices),
                gold=str(item["gold"]),
            )
        )
    return tasks


def ensure_tasks(
    cache: Path,
    splits: Mapping[str, Mapping[str, Sequence[int]]],
    transport: Any,
    sleep: Callable[[float], None],
    *,
    fetch: bool,
) -> list[Task]:
    """Return cached tasks, fetching any missing rows when `fetch` is set."""
    path = _task_path(cache)
    existing = _read_tasks(path) if path.exists() else []
    by_key = {(task.dataset, task.row): task for task in existing}
    missing: dict[str, list[tuple[int, str]]] = {}
    for dataset, groups in splits.items():
        for split, indices in groups.items():
            for index in indices:
                if (dataset, index) not in by_key:
                    missing.setdefault(dataset, []).append((index, split))
    if missing and not fetch:
        return [task for task in existing if _wanted(task, splits)]
    for dataset, pairs in missing.items():
        split_of = dict(pairs)
        found = fetch_dataset_rows(dataset, list(split_of), transport, sleep)
        for index, split in pairs:
            raw = found.get(index)
            if raw is None:
                print(f"row missing: dataset={dataset} row={index}", file=sys.stderr)
                raise TransportError("missing-row")
            by_key[(dataset, index)] = task_from_row(dataset, index, split, raw)
    if missing:
        _write_tasks(path, list(by_key.values()))
    return [task for task in by_key.values() if _wanted(task, splits)]


def _wanted(task: Task, splits: Mapping[str, Mapping[str, Sequence[int]]]) -> bool:
    group = splits.get(task.dataset)
    if group is None:
        return False
    return task.row in group.get(task.split, ())


def prompt_parts(task: Task) -> tuple[str, str]:
    """Return the system text and the human text for `task`."""
    if task.dataset == "mmlu":
        lines = [task.question, ""]
        for letter, choice in zip("ABCD", task.choices, strict=True):
            lines.append(f"{letter}. {choice}")
        return SYSTEM_MMLU, "\n".join(lines)
    return SYSTEM_GSM8K, task.question


def prompt_chars(task: Task) -> int:
    """Character count used as the prompt-token upper bound."""
    system, human = prompt_parts(task)
    return len(system) + len(human)


def messages_for(task: Task) -> list[BaseMessage]:
    """Build the system and human messages sent to a model and to Drex."""
    system, human = prompt_parts(task)
    return [SystemMessage(content=system), HumanMessage(content=human)]


def last_answer(text: str) -> str | None:
    """Return the text after the last `ANSWER:` line, if there is one."""
    found: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        prefix = "ANSWER:"
        if stripped.upper().startswith(prefix):
            found = stripped[len(prefix) :].strip()
    return found


def normalize_number(raw: str) -> Decimal | None:
    """Strip `$`, `,`, `%`, and spaces, then one trailing `.0`, and parse a Decimal."""
    cleaned = raw.replace("$", "").replace(",", "").replace("%", "").replace(" ", "")
    if cleaned.endswith(".0"):
        cleaned = cleaned[: -len(".0")]
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def score_text(dataset: str, text: str, gold: str) -> bool:
    """Score one completion. No `ANSWER:` line is wrong."""
    answer = last_answer(text)
    if answer is None:
        return False
    if dataset == "mmlu":
        letters = "".join(character for character in answer if character.isalpha()).upper()
        return len(letters) == 1 and letters == gold.upper()
    predicted = normalize_number(answer)
    expected = normalize_number(gold)
    return predicted is not None and expected is not None and predicted == expected


def generation_worst(task: Task, model: str) -> Decimal:
    """Worst-case generation cost: len(prompt)/2 input tokens and `MAX_TOKENS` output."""
    price_in, price_out = PRICES_USD_PER_MILLION[model]
    prompt_tokens = Decimal(prompt_chars(task)) / Decimal(2)
    return (prompt_tokens * price_in + Decimal(MAX_TOKENS) * price_out) / MILLION


def routing_worst(task: Task, kind: str) -> Decimal:
    """Worst-case routing cost from the same prompt-token upper bound."""
    rate = SELECTOR_USD_PER_MILLION if kind == "selector" else DECISION_USD_PER_MILLION
    tokens = Decimal(prompt_chars(task)) / Decimal(2)
    return tokens * rate / MILLION


def _safe_name(value: str) -> str:
    return value.replace("/", "__")


def generation_path(cache: Path, model: str, dataset: str, row: int, repeat: int) -> Path:
    """Path of one cached generation."""
    name = f"{_safe_name(model)}__{dataset}__{row}__{repeat}.json"
    return cache / "generations" / name


def route_path(cache: Path, pool: str, arm: str, dataset: str, row: int) -> Path:
    """Path of one cached routing decision."""
    name = f"{pool}__{_safe_name(arm)}__{dataset}__{row}.json"
    return cache / "routes" / name


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw
    return None


def load_outcome(path: Path) -> Outcome | None:
    """Load a generation cache record."""
    raw = _read_json(path)
    if raw is None or "correct" not in raw:
        return None
    try:
        return Outcome(
            correct=bool(raw["correct"]),
            prompt_tokens=int(raw["prompt_tokens"]),
            completion_tokens=int(raw["completion_tokens"]),
            cost_usd=Decimal(str(raw["cost_usd"])),
            latency_s=Decimal(str(raw["latency_s"])),
            provider=str(raw.get("provider", "")),
            finish_reason=str(raw.get("finish_reason", "")),
        )
    except (KeyError, InvalidOperation, TypeError, ValueError):
        return None


def load_route(path: Path) -> RouteChoice | None:
    """Load a routing cache record."""
    raw = _read_json(path)
    if raw is None or "chosen" not in raw:
        return None
    try:
        probabilities = raw.get("probabilities")
        if not isinstance(probabilities, dict):
            probabilities = {}
        latency = raw.get("latency_s")
        return RouteChoice(
            chosen=str(raw["chosen"]),
            fallback=bool(raw["fallback"]),
            reason=str(raw["reason"]),
            probabilities={str(key): float(value) for key, value in probabilities.items()},
            latency_s=None if latency is None else Decimal(str(latency)),
            routing_cost_usd=Decimal(str(raw["routing_cost_usd"])),
            routing_tokens=Decimal(str(raw["routing_tokens"])),
        )
    except (KeyError, InvalidOperation, TypeError, ValueError):
        return None


def billed_generation_cost(
    usage: Mapping[str, Any] | None,
    model: str,
    worst: Decimal,
) -> Decimal:
    """Use `usage.cost` when present, otherwise the price table, otherwise `worst`."""
    if usage is not None and usage.get("cost") is not None:
        return Decimal(str(usage["cost"]))
    if usage is None:
        return worst
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if isinstance(prompt, bool) or isinstance(completion, bool):
        return worst
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return worst
    price_in, price_out = PRICES_USD_PER_MILLION[model]
    return (Decimal(prompt) * price_in + Decimal(completion) * price_out) / MILLION


def _usage(body: Mapping[str, Any]) -> dict[str, Any] | None:
    usage = body.get("usage")
    if isinstance(usage, dict):
        return usage
    return None


def _completion_text(body: Mapping[str, Any]) -> tuple[str, str, str]:
    provider = body.get("provider")
    provider_name = provider if isinstance(provider, str) else ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "", provider_name, ""
    choice = choices[0]
    finish = choice.get("finish_reason")
    finish_reason = finish if isinstance(finish, str) else ""
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content, provider_name, finish_reason
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts), provider_name, finish_reason
    return "", provider_name, finish_reason


def openrouter_key() -> str:
    """Return the OpenRouter key, or an empty string when it is unset."""
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def generate_once(
    task: Task,
    model: str,
    transport: Any,
    sleep: Callable[[float], None],
    worst: Decimal,
) -> tuple[Outcome, str]:
    """Call OpenRouter, retrying 429 and 5xx at most twice, and score the completion."""
    key = openrouter_key()
    if not key:
        raise CallSkippedError
    system, human = prompt_parts(task)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": human},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    started = time.perf_counter()

    def call() -> HttpResult:
        return transport.post_json(OPENROUTER_URL, headers, body, REQUEST_TIMEOUT_S)

    result = with_retries(call, sleep, PAID_ATTEMPTS, retry_timeouts=False)
    latency = Decimal(str(time.perf_counter() - started))
    if result.status != 200 or result.body is None:
        raise TransportError("openrouter", result.status)
    usage = _usage(result.body)
    text, provider, finish_reason = _completion_text(result.body)
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    completion_tokens = usage.get("completion_tokens") if usage else None
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
        prompt_tokens = 0
    if not isinstance(completion_tokens, int) or isinstance(completion_tokens, bool):
        completion_tokens = 0
    outcome = Outcome(
        correct=score_text(task.dataset, text, task.gold),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=billed_generation_cost(usage, model, worst),
        latency_s=latency,
        provider=provider,
        finish_reason=finish_reason,
    )
    return outcome, text


def _outcome_payload(task: Task, model: str, repeat: int, outcome: Outcome, text: str) -> dict:
    return {
        "model": model,
        "dataset": task.dataset,
        "row": task.row,
        "repeat": repeat,
        "correct": outcome.correct,
        "prompt_tokens": outcome.prompt_tokens,
        "completion_tokens": outcome.completion_tokens,
        "cost_usd": format(outcome.cost_usd, "f"),
        "latency_s": format(outcome.latency_s, "f"),
        "provider": outcome.provider,
        "finish_reason": outcome.finish_reason,
        "completion": text,
    }


def model_ids() -> list[str]:
    """Unique model ids across both pools, in a stable order."""
    ordered: list[str] = []
    for pool in POOLS.values():
        for role in ("small", "strong"):
            model = pool[role]
            if model not in ordered:
                ordered.append(model)
    return ordered


def selector_arm_names() -> list[str]:
    """The six selector arms."""
    return [f"selector/{priority}/{context}" for priority in PRIORITIES for context in CONTEXTS]


def routed_arm_names() -> list[str]:
    """Arms that make a routing choice."""
    return ["rule", *selector_arm_names(), "decision"]


def arm_names() -> list[str]:
    """Every reported arm, including references."""
    return ["fixed_small", "fixed_strong", *routed_arm_names(), "oracle"]


_selectors: dict[tuple[str, str, str, str], tuple[NaceModelSelector, threading.Lock]] = {}
_selectors_guard = threading.Lock()
_router: NaceModelRouterMiddleware | None = None
_router_lock = threading.Lock()


def default_selector(
    messages: Sequence[BaseMessage],
    *,
    small: str,
    strong: str,
    priority: str,
    context: str,
) -> dict[str, Any]:
    """Call `NaceModelSelector.select` for one priority and context."""
    key = (small, strong, priority, context)
    with _selectors_guard:
        pair = _selectors.get(key)
        if pair is None:
            selector = NaceModelSelector(
                candidates={small: None, strong: None},
                priority=priority,
                context=context,
                fallback=None,
            )
            pair = (selector, threading.Lock())
            _selectors[key] = pair
    selector, lock = pair
    with lock:
        return dict(selector.select(list(messages)))


def build_router() -> NaceModelRouterMiddleware:
    """Build the decision router. The chat models are never called."""
    return NaceModelRouterMiddleware(
        choices={
            "small": ModelChoice(FakeListChatModel(responses=["unused"]), SMALL_CRITERIA),
            "strong": ModelChoice(FakeListChatModel(responses=["unused"]), STRONG_CRITERIA),
        },
        instructions=DECISION_INSTRUCTIONS,
    )


def default_decider(messages: Sequence[BaseMessage]) -> tuple[dict[str, Any] | None, float]:
    """Call `NaceModelRouterMiddleware.before_agent` and time that call."""
    global _router
    with _router_lock:
        if _router is None:
            _router = build_router()
        router_logger = logging.getLogger("langchain_nace.experimental.middleware.model_router")
        level = router_logger.level
        router_logger.setLevel(logging.CRITICAL)
        started = time.perf_counter()
        try:
            update = _router.before_agent({"messages": list(messages)}, cast("Any", None))
        finally:
            router_logger.setLevel(level)
        elapsed = time.perf_counter() - started
    route = update.get("nace_model_route")
    if isinstance(route, dict):
        return route, elapsed
    return None, elapsed


def _routing_tokens(task: Task, cache: Path, small: str) -> Decimal:
    totals: list[Decimal] = []
    for repeat in range(REPEATS):
        outcome = load_outcome(generation_path(cache, small, task.dataset, task.row, repeat))
        if outcome is not None and outcome.prompt_tokens > 0:
            totals.append(Decimal(outcome.prompt_tokens))
    if totals:
        return sum(totals, start=Decimal(0)) / Decimal(len(totals))
    return Decimal(prompt_chars(task)) / Decimal(2)


def routing_cost(tokens: Decimal, kind: str) -> Decimal:
    """Estimated routing cost. The route response does not include a token count."""
    rate = SELECTOR_USD_PER_MILLION if kind == "selector" else DECISION_USD_PER_MILLION
    return tokens * rate / MILLION


def _apply_selection(
    selection: Mapping[str, Any],
    *,
    small: str,
    strong: str,
    wall_s: Decimal,
) -> RouteChoice:
    model = selection.get("model")
    fallback = bool(selection.get("fallback")) or model is None or model not in {small, strong}
    chosen = strong if fallback or not isinstance(model, str) else model
    reason = selection.get("reason")
    probabilities = selection.get("probabilities")
    if not isinstance(probabilities, dict):
        probabilities = {}
    latency_ms = selection.get("latency_ms")
    if isinstance(latency_ms, int) and not isinstance(latency_ms, bool):
        latency = Decimal(latency_ms) / Decimal(1000)
    else:
        latency = wall_s
    return RouteChoice(
        chosen=chosen,
        fallback=fallback,
        reason=str(reason),
        probabilities={str(key): float(value) for key, value in probabilities.items()},
        latency_s=latency,
        routing_cost_usd=Decimal(0),
        routing_tokens=Decimal(0),
    )


def _route_payload(choice: RouteChoice) -> dict[str, Any]:
    return {
        "chosen": choice.chosen,
        "fallback": choice.fallback,
        "reason": choice.reason,
        "probabilities": choice.probabilities,
        "latency_s": None if choice.latency_s is None else format(choice.latency_s, "f"),
        "routing_cost_usd": format(choice.routing_cost_usd, "f"),
        "routing_tokens": format(choice.routing_tokens, "f"),
    }


def run_jobs(jobs: Sequence[Job], ledger: SpendLedger, workers: int) -> bool:
    """Run `jobs` on `workers` threads. Return True when a call was refused by the cap."""
    stopped = False
    stop_lock = threading.Lock()

    def run(job: Job) -> None:
        nonlocal stopped
        if not ledger.try_reserve(job.worst):
            with stop_lock:
                stopped = True
            return
        cost = job.worst
        skipped = False
        try:
            cost = job.fn()
        except CallSkippedError:
            skipped = True
        except Exception as exc:
            print_failure(exc)
            cost = job.worst
        if skipped:
            ledger.release(job.worst)
            return
        ledger.commit(reserved=job.worst, cost=cost, kind=job.kind, model=job.model)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in futures:
            future.result()
    return stopped


def generation_jobs(
    tasks: Sequence[Task],
    cache: Path,
    transport: Any,
    sleep: Callable[[float], None],
) -> list[Job]:
    """One job per missing (model, task, repeat) on the test split."""
    jobs: list[Job] = []
    test_tasks = [task for task in tasks if task.split == "test"]
    for model in model_ids():
        for task in test_tasks:
            for repeat in range(REPEATS):
                path = generation_path(cache, model, task.dataset, task.row, repeat)
                if load_outcome(path) is not None:
                    continue
                worst = generation_worst(task, model)

                def fn(
                    task: Task = task,
                    model: str = model,
                    repeat: int = repeat,
                    path: Path = path,
                    worst: Decimal = worst,
                ) -> Decimal:
                    outcome, text = generate_once(task, model, transport, sleep, worst)
                    _write_json(path, _outcome_payload(task, model, repeat, outcome, text))
                    return outcome.cost_usd

                jobs.append(Job(kind="generate", model=model, worst=worst, fn=fn))
    return jobs


def _selector_choice(
    task: Task,
    pool: str,
    arm: str,
    selector: Callable[..., dict[str, Any]],
    cache: Path,
) -> tuple[RouteChoice, bool]:
    priority, context = arm.split("/", 2)[1:]
    small = POOLS[pool]["small"]
    strong = POOLS[pool]["strong"]
    started = time.perf_counter()
    selection = selector(
        messages_for(task),
        small=small,
        strong=strong,
        priority=priority,
        context=context,
    )
    wall = Decimal(str(time.perf_counter() - started))
    choice = _apply_selection(selection, small=small, strong=strong, wall_s=wall)
    local = choice.reason in LOCAL_REASONS and selection.get("latency_ms") is None
    tokens = _routing_tokens(task, cache, small)
    cost = Decimal(0) if local else routing_cost(tokens, "selector")
    choice = RouteChoice(
        chosen=choice.chosen,
        fallback=choice.fallback,
        reason=choice.reason,
        probabilities=choice.probabilities,
        latency_s=choice.latency_s,
        routing_cost_usd=cost,
        routing_tokens=tokens,
    )
    return choice, local


def _decision_choice(
    task: Task,
    pool: str,
    decider: Callable[[Sequence[BaseMessage]], tuple[dict[str, Any] | None, float]],
    cache: Path,
) -> RouteChoice:
    small = POOLS[pool]["small"]
    strong = POOLS[pool]["strong"]
    route, elapsed = decider(messages_for(task))
    if not route:
        chosen = strong
        fallback = True
        reason = "error"
        probabilities: dict[str, float] = {}
    else:
        chosen = small if route.get("route") == "small" else strong
        fallback = bool(route.get("escalated")) or route.get("choice") is None
        reason = str(route.get("reason", "error"))
        raw_probabilities = route.get("probabilities")
        probabilities = (
            {str(key): float(value) for key, value in raw_probabilities.items()}
            if isinstance(raw_probabilities, dict)
            else {}
        )
    tokens = _routing_tokens(task, cache, small)
    return RouteChoice(
        chosen=chosen,
        fallback=fallback,
        reason=reason,
        probabilities=probabilities,
        latency_s=Decimal(str(elapsed)),
        routing_cost_usd=routing_cost(tokens, "decision"),
        routing_tokens=tokens,
    )


def route_jobs(
    tasks: Sequence[Task],
    cache: Path,
    selector: Callable[..., dict[str, Any]],
    decider: Callable[[Sequence[BaseMessage]], tuple[dict[str, Any] | None, float]],
) -> list[Job]:
    """One job per missing selector or decision call on the test split."""
    jobs: list[Job] = []
    test_tasks = [task for task in tasks if task.split == "test"]
    for pool in POOLS:
        for task in test_tasks:
            for arm in selector_arm_names():
                path = route_path(cache, pool, arm, task.dataset, task.row)
                if load_route(path) is not None:
                    continue
                worst = routing_worst(task, "selector")

                def fn(
                    task: Task = task,
                    pool: str = pool,
                    arm: str = arm,
                    path: Path = path,
                ) -> Decimal:
                    choice, local = _selector_choice(task, pool, arm, selector, cache)
                    _write_json(path, _route_payload(choice))
                    if local:
                        raise CallSkippedError
                    return choice.routing_cost_usd

                jobs.append(Job(kind="selector", model=f"{pool}/{arm}", worst=worst, fn=fn))
            path = route_path(cache, pool, "decision", task.dataset, task.row)
            if load_route(path) is not None:
                continue
            worst = routing_worst(task, "decision")

            def fn_decision(
                task: Task = task,
                pool: str = pool,
                path: Path = path,
            ) -> Decimal:
                choice = _decision_choice(task, pool, decider, cache)
                _write_json(path, _route_payload(choice))
                return choice.routing_cost_usd

            jobs.append(Job(kind="decision", model=f"{pool}/decision", worst=worst, fn=fn_decision))
    return jobs


def worst_of(jobs: Sequence[Job]) -> Decimal:
    """Sum of worst-case costs."""
    return sum((job.worst for job in jobs), start=Decimal(0))


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear percentile. `fraction` is between 0 and 1."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def bootstrap_mean(values: Sequence[float], seed: int = SEED) -> tuple[float, float, float]:
    """Return the mean and the seeded 95% bootstrap interval."""
    if not values:
        return 0.0, 0.0, 0.0
    point = sum(values) / len(values)
    rng = random.Random(seed)  # noqa: S311
    means: list[float] = []
    size = len(values)
    for _ in range(BOOTSTRAP_SAMPLES):
        total = 0.0
        for _ in range(size):
            total += values[rng.randrange(size)]
        means.append(total / size)
    return point, percentile(means, 0.025), percentile(means, 0.975)


def bootstrap_difference(
    left: Sequence[float],
    right: Sequence[float],
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Paired difference of means, with a seeded 95% interval."""
    point = (sum(left) - sum(right)) / len(left)
    rng = random.Random(seed)  # noqa: S311
    diffs: list[float] = []
    size = len(left)
    for _ in range(BOOTSTRAP_SAMPLES):
        total = 0.0
        for _ in range(size):
            index = rng.randrange(size)
            total += left[index] - right[index]
        diffs.append(total / size)
    return point, percentile(diffs, 0.025), percentile(diffs, 0.975)


def bootstrap_against_mix(
    arm: Sequence[float],
    small: Sequence[float],
    strong: Sequence[float],
    flags: Sequence[float],
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Difference between `arm` and the linear mix at `arm`'s strong share."""
    share = sum(flags) / len(flags)
    small_mean = sum(small) / len(small)
    strong_mean = sum(strong) / len(strong)
    point = sum(arm) / len(arm) - ((1 - share) * small_mean + share * strong_mean)
    rng = random.Random(seed)  # noqa: S311
    diffs: list[float] = []
    size = len(arm)
    for _ in range(BOOTSTRAP_SAMPLES):
        indexes = [rng.randrange(size) for _ in range(size)]
        arm_mean = sum(arm[index] for index in indexes) / size
        small_m = sum(small[index] for index in indexes) / size
        strong_m = sum(strong[index] for index in indexes) / size
        mix_share = sum(flags[index] for index in indexes) / size
        mix = (1 - mix_share) * small_m + mix_share * strong_m
        diffs.append(arm_mean - mix)
    return point, percentile(diffs, 0.025), percentile(diffs, 0.975)


def human_length(task: Task) -> int:
    """Length of the human message."""
    return len(prompt_parts(task)[1])


def dev_median(tasks: Sequence[Task]) -> float | None:
    """Median human-message length on the dev split."""
    lengths = sorted(human_length(task) for task in tasks if task.split == "dev")
    if not lengths:
        return None
    mid = len(lengths) // 2
    if len(lengths) % 2:
        return float(lengths[mid])
    return (lengths[mid - 1] + lengths[mid]) / 2


def rule_chooses_strong(task: Task, threshold: float) -> bool:
    """Strong when the human message is longer than `threshold` or fenced."""
    human = prompt_parts(task)[1]
    return len(human) > threshold or "```" in human


def _mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, start=Decimal(0)) / Decimal(len(values))


def _fmean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


@dataclass
class TaskMeasure:
    """Per-task numbers for one arm."""

    accuracy: float
    cost: float
    strong: float
    latencies: list[float]
    route_latency: float | None
    fallback: bool
    reason: str


def _repeat_outcomes(
    cache: Path,
    model: str,
    task: Task,
) -> list[Outcome] | None:
    found: list[Outcome] = []
    for repeat in range(REPEATS):
        outcome = load_outcome(generation_path(cache, model, task.dataset, task.row, repeat))
        if outcome is None:
            return None
        found.append(outcome)
    return found


def _fixed_measure(outcomes: Sequence[Outcome], strong: bool) -> TaskMeasure:
    return TaskMeasure(
        accuracy=_fmean([1.0 if item.correct else 0.0 for item in outcomes]),
        cost=float(_mean([item.cost_usd for item in outcomes])),
        strong=1.0 if strong else 0.0,
        latencies=[float(item.latency_s) for item in outcomes],
        route_latency=None,
        fallback=False,
        reason="",
    )


def _chosen_measure(outcomes: Sequence[Outcome], route: RouteChoice, strong: bool) -> TaskMeasure:
    end_to_end = []
    route_s = float(route.latency_s) if route.latency_s is not None else 0.0
    for item in outcomes:
        end_to_end.append(route_s + float(item.latency_s))
    return TaskMeasure(
        accuracy=_fmean([1.0 if item.correct else 0.0 for item in outcomes]),
        cost=float(_mean([item.cost_usd for item in outcomes]) + route.routing_cost_usd),
        strong=1.0 if strong else 0.0,
        latencies=end_to_end,
        route_latency=None if route.latency_s is None else float(route.latency_s),
        fallback=route.fallback,
        reason=route.reason,
    )


def _oracle_measure(small: Sequence[Outcome], strong: Sequence[Outcome]) -> TaskMeasure:
    correct: list[float] = []
    costs: list[Decimal] = []
    flags: list[float] = []
    latencies: list[float] = []
    for left, right in zip(small, strong, strict=True):
        use_strong = (not left.correct) and right.correct
        chosen = right if use_strong else left
        correct.append(1.0 if chosen.correct else 0.0)
        costs.append(chosen.cost_usd)
        flags.append(1.0 if use_strong else 0.0)
        latencies.append(float(chosen.latency_s))
    return TaskMeasure(
        accuracy=_fmean(correct),
        cost=float(_mean(costs)),
        strong=_fmean(flags),
        latencies=latencies,
        route_latency=None,
        fallback=False,
        reason="",
    )


def _arm_record(name: str, measures: Sequence[TaskMeasure]) -> dict[str, Any]:
    if not measures:
        return {
            "name": name,
            "n_tasks": 0,
            "accuracy": None,
            "accuracy_low": None,
            "accuracy_high": None,
            "cost_per_1000_usd": None,
            "cost_low": None,
            "cost_high": None,
            "latency_p50_s": None,
            "latency_p95_s": None,
            "route_p50_s": None,
            "route_p95_s": None,
            "route_p99_s": None,
            "strong_share": None,
            "fallback_rate": None,
            "fallback_reasons": {},
            "random_accuracy": None,
            "random_cost_per_1000_usd": None,
            "quality_gain": None,
        }
    accuracies = [item.accuracy for item in measures]
    costs = [item.cost * 1000 for item in measures]
    accuracy, accuracy_low, accuracy_high = bootstrap_mean(accuracies)
    _cost, cost_low, cost_high = bootstrap_mean(costs)
    point_cost = sum(item.cost for item in measures) / len(measures) * 1000
    latencies = [value for item in measures for value in item.latencies]
    routes = [item.route_latency for item in measures if item.route_latency is not None]
    reasons: dict[str, int] = {}
    fallbacks = 0
    for item in measures:
        if item.fallback:
            fallbacks += 1
            reasons[item.reason] = reasons.get(item.reason, 0) + 1
    share = sum(item.strong for item in measures) / len(measures)
    return {
        "name": name,
        "n_tasks": len(measures),
        "accuracy": accuracy,
        "accuracy_low": accuracy_low,
        "accuracy_high": accuracy_high,
        "cost_per_1000_usd": point_cost,
        "cost_low": cost_low,
        "cost_high": cost_high,
        "latency_p50_s": percentile(latencies, 0.50) if latencies else None,
        "latency_p95_s": percentile(latencies, 0.95) if latencies else None,
        "route_p50_s": percentile(routes, 0.50) if routes else None,
        "route_p95_s": percentile(routes, 0.95) if routes else None,
        "route_p99_s": percentile(routes, 0.99) if routes else None,
        "strong_share": share,
        "fallback_rate": fallbacks / len(measures),
        "fallback_reasons": reasons,
        "random_accuracy": None,
        "random_cost_per_1000_usd": None,
        "quality_gain": None,
    }


def _fill_random(
    record: dict[str, Any],
    measures: Sequence[TaskMeasure],
    small_measures: Sequence[TaskMeasure],
    strong_measures: Sequence[TaskMeasure],
) -> None:
    if not measures or not small_measures:
        return
    share = float(record["strong_share"])
    small_accuracy = sum(item.accuracy for item in small_measures) / len(small_measures)
    strong_accuracy = sum(item.accuracy for item in strong_measures) / len(strong_measures)
    small_cost = sum(item.cost for item in small_measures) / len(small_measures) * 1000
    strong_cost = sum(item.cost for item in strong_measures) / len(strong_measures) * 1000
    record["random_accuracy"] = (1 - share) * small_accuracy + share * strong_accuracy
    record["random_cost_per_1000_usd"] = (1 - share) * small_cost + share * strong_cost
    record["quality_gain"] = float(record["accuracy"]) - float(record["random_accuracy"])


def _comparison(
    name: str,
    left_accuracy: Sequence[float],
    right_accuracy: Sequence[float],
    left_cost: Sequence[float],
    right_cost: Sequence[float],
) -> dict[str, Any]:
    accuracy, accuracy_low, accuracy_high = bootstrap_difference(left_accuracy, right_accuracy)
    cost, cost_low, cost_high = bootstrap_difference(left_cost, right_cost)
    return {
        "name": name,
        "accuracy": accuracy,
        "accuracy_low": accuracy_low,
        "accuracy_high": accuracy_high,
        "cost_per_1000_usd": cost,
        "cost_low": cost_low,
        "cost_high": cost_high,
    }


def _mix_comparison(
    name: str,
    arm_accuracy: Sequence[float],
    arm_cost: Sequence[float],
    small_accuracy: Sequence[float],
    strong_accuracy: Sequence[float],
    small_cost: Sequence[float],
    strong_cost: Sequence[float],
    flags: Sequence[float],
) -> dict[str, Any]:
    accuracy, accuracy_low, accuracy_high = bootstrap_against_mix(
        arm_accuracy, small_accuracy, strong_accuracy, flags
    )
    cost, cost_low, cost_high = bootstrap_against_mix(arm_cost, small_cost, strong_cost, flags)
    return {
        "name": name,
        "accuracy": accuracy,
        "accuracy_low": accuracy_low,
        "accuracy_high": accuracy_high,
        "cost_per_1000_usd": cost,
        "cost_low": cost_low,
        "cost_high": cost_high,
    }


def pool_report(pool: str, tasks: Sequence[Task], cache: Path) -> dict[str, Any]:
    """Measures for one pool. Arms are aligned on the tasks they can score."""
    small_id = POOLS[pool]["small"]
    strong_id = POOLS[pool]["strong"]
    threshold = dev_median(tasks)
    per_arm: dict[str, dict[tuple[str, int], TaskMeasure]] = {name: {} for name in arm_names()}
    generated = 0
    for task in tasks:
        if task.split != "test":
            continue
        small = _repeat_outcomes(cache, small_id, task)
        strong = _repeat_outcomes(cache, strong_id, task)
        if small is None or strong is None:
            continue
        # Every arm is scored on the same tasks: a task with a missing route is left out.
        routes = {
            arm: load_route(route_path(cache, pool, arm, task.dataset, task.row))
            for arm in (*selector_arm_names(), "decision")
        }
        if any(choice is None for choice in routes.values()):
            continue
        generated += 1
        key = (task.dataset, task.row)
        per_arm["fixed_small"][key] = _fixed_measure(small, False)
        per_arm["fixed_strong"][key] = _fixed_measure(strong, True)
        per_arm["oracle"][key] = _oracle_measure(small, strong)
        if threshold is not None:
            use_strong = rule_chooses_strong(task, threshold)
            chosen = strong if use_strong else small
            route = RouteChoice(
                chosen=strong_id if use_strong else small_id,
                fallback=False,
                reason="rule",
                probabilities={},
                latency_s=None,
                routing_cost_usd=Decimal(0),
                routing_tokens=Decimal(0),
            )
            per_arm["rule"][key] = _chosen_measure(chosen, route, use_strong)
        for arm in (*selector_arm_names(), "decision"):
            choice = routes[arm]
            if choice is None:
                continue
            chosen_outcomes = strong if choice.chosen == strong_id else small
            per_arm[arm][key] = _chosen_measure(chosen_outcomes, choice, choice.chosen == strong_id)
    arms = []
    for name in arm_names():
        measures = list(per_arm[name].values())
        record = _arm_record(name, measures)
        if name in routed_arm_names():
            keys = list(per_arm[name])
            _fill_random(
                record,
                [per_arm[name][key] for key in keys],
                [per_arm["fixed_small"][key] for key in keys],
                [per_arm["fixed_strong"][key] for key in keys],
            )
        arms.append(record)
    return {
        "small": small_id,
        "strong": strong_id,
        "dev_median_chars": threshold,
        "arms": arms,
        "comparisons": _balanced_comparisons(per_arm),
        "n_tasks_with_generations": generated,
    }


def _aligned(
    per_arm: Mapping[str, Mapping[tuple[str, int], TaskMeasure]],
    left: str,
    right: str,
) -> tuple[list[TaskMeasure], list[TaskMeasure]] | None:
    shared = [key for key in per_arm[left] if key in per_arm[right]]
    if not shared:
        return None
    return [per_arm[left][key] for key in shared], [per_arm[right][key] for key in shared]


def _balanced_comparisons(
    per_arm: Mapping[str, Mapping[tuple[str, int], TaskMeasure]],
) -> list[dict[str, Any]]:
    name = "selector/balanced/conversation"
    if name not in per_arm or not per_arm[name]:
        return []
    rows: list[dict[str, Any]] = []
    for other in ("fixed_small", "fixed_strong", "rule"):
        pair = _aligned(per_arm, name, other)
        if pair is None:
            continue
        left, right = pair
        rows.append(
            _comparison(
                f"{name} - {other}",
                [item.accuracy for item in left],
                [item.accuracy for item in right],
                [item.cost * 1000 for item in left],
                [item.cost * 1000 for item in right],
            )
        )
    keys = [key for key in per_arm[name] if key in per_arm["fixed_small"]]
    if keys:
        arm = [per_arm[name][key] for key in keys]
        small = [per_arm["fixed_small"][key] for key in keys]
        strong = [per_arm["fixed_strong"][key] for key in keys]
        rows.append(
            _mix_comparison(
                f"{name} - random_at_same_share",
                [item.accuracy for item in arm],
                [item.cost * 1000 for item in arm],
                [item.accuracy for item in small],
                [item.accuracy for item in strong],
                [item.cost * 1000 for item in small],
                [item.cost * 1000 for item in strong],
                [item.strong for item in arm],
            )
        )
    return rows


def _round_numbers(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {str(key): _round_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_numbers(item) for item in value]
    return value


def build_summary(
    *,
    complete: bool,
    budget: Decimal,
    spent: Decimal,
    overrun: bool,
    splits: Mapping[str, Mapping[str, Sequence[int]]],
    tasks: Sequence[Task],
    cache: Path,
) -> dict[str, Any]:
    """Assemble the JSON summary from cached numbers."""
    summary = {
        "complete": complete,
        "seed": SEED,
        "repeats": REPEATS,
        "budget_usd": format(budget, "f"),
        "spent_usd": format(spent, "f"),
        "overrun": overrun,
        "decision_function": DECISION_FUNCTION,
        "routing_cost": (
            "estimate: small-model prompt tokens times 0.02 (selector) or 0.06 "
            "(decision) USD per 1M tokens; char/2 when the prompt-token count is missing"
        ),
        "samples": {
            name: {"dev": list(groups["dev"]), "test": list(groups["test"])}
            for name, groups in splits.items()
        },
        "pools": {pool: pool_report(pool, tasks, cache) for pool in POOLS},
    }
    return cast("dict[str, Any]", _round_numbers(summary))


def render_results(summary: Mapping[str, Any]) -> str:
    """Render markdown tables from a summary object. No numbers are added here."""
    lines = [
        "# Model selection benchmark",
        "",
        "Generated by `benchmarks/model_selection/run_benchmark.py` from `summary.json`.",
        "",
        f"- complete: {summary['complete']}",
        f"- seed: {summary['seed']}",
        f"- repeats: {summary['repeats']}",
        f"- budget_usd: {summary['budget_usd']}",
        f"- spent_usd: {summary['spent_usd']}",
        f"- overrun: {summary['overrun']}",
        f"- decision function: {summary['decision_function']}",
        f"- routing cost: {summary['routing_cost']}",
        "",
    ]
    for pool, body in summary["pools"].items():
        lines.append(f"## Pool {pool}")
        lines.append("")
        lines.append(f"- small: {body['small']}")
        lines.append(f"- strong: {body['strong']}")
        lines.append(f"- dev median human chars: {body['dev_median_chars']}")
        lines.append(f"- test tasks with both generations: {body['n_tasks_with_generations']}")
        lines.append("")
        lines.append(
            "| arm | n | accuracy | accuracy 95% | cost / 1000 USD | cost 95% | "
            "latency p50 | latency p95 | route p50 | route p95 | route p99 | "
            "strong share | fallback rate | quality gain vs random |"
        )
        lines.append(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        )
        for arm in body["arms"]:
            lines.append(
                "| {name} | {n_tasks} | {accuracy} | {accuracy_low} .. {accuracy_high} | "
                "{cost_per_1000_usd} | {cost_low} .. {cost_high} | {latency_p50_s} | "
                "{latency_p95_s} | {route_p50_s} | {route_p95_s} | {route_p99_s} | "
                "{strong_share} | {fallback_rate} | {quality_gain} |".format(**arm)
            )
        lines.append("")
        lines.append("Fallback reason counts are in `summary.json` under each arm.")
        lines.append("")
        if body["comparisons"]:
            lines.append(
                "| comparison | accuracy difference | accuracy 95% | "
                "cost / 1000 difference | cost 95% |"
            )
            lines.append("| --- | --- | --- | --- | --- |")
            for row in body["comparisons"]:
                lines.append(
                    "| {name} | {accuracy} | {accuracy_low} .. {accuracy_high} | "
                    "{cost_per_1000_usd} | {cost_low} .. {cost_high} |".format(**row)
                )
            lines.append("")
    lines.append(
        "Quality gain is the arm's accuracy minus the linear mix of fixed-small and "
        "fixed-strong at that arm's strong share. The mix is analytic."
    )
    lines.append("")
    return "\n".join(lines)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def write_per_task(
    path: Path,
    splits: Mapping[str, Mapping[str, Sequence[int]]],
    tasks: Sequence[Task],
    cache: Path,
) -> None:
    """Write ids and numbers only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    task_of = {(task.dataset, task.row): task for task in tasks}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for dataset, groups in splits.items():
            for split, indices in groups.items():
                for index in indices:
                    writer.writerow(
                        {
                            "record_type": "task",
                            "dataset": dataset,
                            "row": index,
                            "split": split,
                        }
                    )
        for dataset, groups in splits.items():
            for index in groups.get("test", ()):
                task = task_of.get((dataset, index))
                for model in model_ids():
                    for repeat in range(REPEATS):
                        outcome = load_outcome(
                            generation_path(cache, model, dataset, index, repeat)
                        )
                        if outcome is None or task is None:
                            continue
                        writer.writerow(
                            {
                                "record_type": "outcome",
                                "dataset": dataset,
                                "row": index,
                                "split": "test",
                                "repeat": repeat,
                                "model": model,
                                "correct": _csv_value(outcome.correct),
                                "prompt_tokens": outcome.prompt_tokens,
                                "completion_tokens": outcome.completion_tokens,
                                "cost_usd": _csv_value(outcome.cost_usd),
                                "latency_s": _csv_value(outcome.latency_s),
                                "provider": outcome.provider,
                                "finish_reason": outcome.finish_reason,
                            }
                        )
                for pool in POOLS:
                    for arm in routed_arm_names():
                        if arm == "rule":
                            threshold = dev_median(tasks)
                            if task is None or threshold is None:
                                continue
                            use_strong = rule_chooses_strong(task, threshold)
                            writer.writerow(
                                {
                                    "record_type": "choice",
                                    "dataset": dataset,
                                    "row": index,
                                    "split": "test",
                                    "pool": pool,
                                    "arm": arm,
                                    "chosen": POOLS[pool]["strong" if use_strong else "small"],
                                    "fallback": "0",
                                    "reason": "rule",
                                    "routing_cost_usd": "0",
                                }
                            )
                            continue
                        choice = load_route(route_path(cache, pool, arm, dataset, index))
                        if choice is None:
                            continue
                        probability = choice.probabilities.get(choice.chosen)
                        writer.writerow(
                            {
                                "record_type": "choice",
                                "dataset": dataset,
                                "row": index,
                                "split": "test",
                                "pool": pool,
                                "arm": arm,
                                "chosen": choice.chosen,
                                "route_latency_s": _csv_value(choice.latency_s),
                                "fallback": _csv_value(choice.fallback),
                                "reason": choice.reason,
                                "chosen_probability": "" if probability is None else probability,
                                "routing_cost_usd": _csv_value(choice.routing_cost_usd),
                                "routing_tokens": _csv_value(choice.routing_tokens),
                            }
                        )


def write_results(
    results: Path,
    summary: Mapping[str, Any],
    splits: Mapping[str, Mapping[str, Sequence[int]]],
    tasks: Sequence[Task],
    cache: Path,
) -> None:
    """Write summary.json, RESULTS.md, and per_task.csv."""
    results.mkdir(parents=True, exist_ok=True)
    summary_path = results / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    (results / "RESULTS.md").write_text(render_results(loaded), encoding="utf-8")
    write_per_task(results / "per_task.csv", splits, tasks, cache)


def _complete(
    tasks: Sequence[Task],
    cache: Path,
    splits: Mapping[str, Mapping[str, Sequence[int]]],
) -> bool:
    have = {(task.dataset, task.row) for task in tasks}
    for dataset, groups in splits.items():
        for indices in groups.values():
            if any((dataset, index) not in have for index in indices):
                return False
    test_tasks = [task for task in tasks if task.split == "test"]
    if not test_tasks:
        return False
    for task in test_tasks:
        for model in model_ids():
            for repeat in range(REPEATS):
                path = generation_path(cache, model, task.dataset, task.row, repeat)
                if load_outcome(path) is None:
                    return False
        for pool in POOLS:
            for arm in (*selector_arm_names(), "decision"):
                if load_route(route_path(cache, pool, arm, task.dataset, task.row)) is None:
                    return False
    return True


def run_benchmark(
    *,
    budget: Decimal,
    limit: int | None,
    stage: str,
    cache: Path,
    results: Path,
    transport: Any,
    sleep: Callable[[float], None],
    selector: Callable[..., dict[str, Any]] | None,
    decider: Callable[[Sequence[BaseMessage]], tuple[dict[str, Any] | None, float]] | None,
) -> int:
    """Run the requested stage. Return 0, or 1 when the cap stops the run."""
    splits = assign_splits(limit)
    fetch = stage != "report"
    try:
        tasks = ensure_tasks(cache, splits, transport, sleep, fetch=fetch)
    except (TransportError, httpx.HTTPError) as exc:
        print_failure(exc)
        return 1
    ledger = SpendLedger(cache / "spend.jsonl", budget)
    selector_fn = selector or default_selector
    decider_fn = decider or default_decider
    stopped = False
    paid = stage in {"generate", "route", "all"}
    if paid:
        if stage in {"generate", "all"}:
            gen_jobs = generation_jobs(tasks, cache, transport, sleep)
        else:
            gen_jobs = []
        route_plan = (
            route_jobs(tasks, cache, selector_fn, decider_fn) if stage in {"route", "all"} else []
        )
        planned = [*gen_jobs, *route_plan] if stage == "all" else (gen_jobs or route_plan)
        try:
            ledger.preflight(worst_of(planned))
        except BudgetError as exc:
            print(
                "preflight refused: "
                f"spent_usd={exc.spent} worst_case_usd={exc.worst} budget_usd={exc.budget}"
            )
            return 1
        if stage in {"generate", "all"} and gen_jobs:
            stopped = run_jobs(gen_jobs, ledger, GENERATE_WORKERS)
        if stage in {"route", "all"} and route_plan and not stopped:
            stopped = run_jobs(route_plan, ledger, ROUTE_WORKERS) or stopped
    complete = (not stopped) and (not ledger.overrun) and _complete(tasks, cache, splits)
    summary = build_summary(
        complete=complete,
        budget=budget,
        spent=ledger.spent,
        overrun=ledger.overrun,
        splits=splits,
        tasks=tasks,
        cache=cache,
    )
    write_results(results, summary, splits, tasks, cache)
    if stopped or ledger.overrun:
        return 1
    return 0


def _parse_budget(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        message = "budget must be a number"
        raise argparse.ArgumentTypeError(message) from exc


def main(argv: Sequence[str] | None = None, **kwargs: Any) -> int:
    """Parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description="Replay benchmark for NaceModelSelector.")
    parser.add_argument("--budget-usd", type=_parse_budget, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stage", choices=("generate", "route", "report", "all"), default="all")
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.budget_usd < 0 or args.budget_usd > MAX_BUDGET_USD:
        print("budget must be between 0 and 5.00", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 0:
        print("limit must be >= 0", file=sys.stderr)
        return 2
    try:
        return run_benchmark(
            budget=args.budget_usd,
            limit=args.limit,
            stage=args.stage,
            cache=args.cache,
            results=args.results,
            transport=kwargs.get("transport") or HttpxTransport(),
            sleep=kwargs.get("sleep") or time.sleep,
            selector=kwargs.get("selector"),
            decider=kwargs.get("decider"),
        )
    except LedgerError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
