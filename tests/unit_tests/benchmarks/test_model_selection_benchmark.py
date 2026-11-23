"""Offline tests for the model-selection replay benchmark."""

import json
import sys
import threading
from decimal import Decimal
from pathlib import Path

from langchain_core.messages import HumanMessage

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "model_selection"))

import run_benchmark as bench  # noqa: E402

SECRET = "super-secret-key-9f3a"


class Transport:
    def __init__(self, post=None, get=None):
        self.posts = []
        self.gets = []
        self._post = post
        self._get = get

    def post_json(self, url, headers, body, timeout):
        self.posts.append({"url": url, "headers": dict(headers), "body": body})
        if self._post is None:
            raise AssertionError(url)
        return self._post(url, headers, body, timeout)

    def get_json(self, url, timeout):
        self.gets.append(url)
        if self._get is None:
            raise AssertionError(url)
        return self._get(url, timeout)


def _no_sleep(_seconds):
    return None


def _ok_body(text="ANSWER: A"):
    return bench.HttpResult(
        200,
        {
            "provider": "test-provider",
            "choices": [
                {"finish_reason": "stop", "message": {"content": text}},
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "cost": 0.0002},
        },
    )


def _task(dataset, row, split, question, gold="A"):
    choices = ("a", "b", "c", "d") if dataset == "mmlu" else ()
    if dataset == "gsm8k":
        gold = "42"
    return bench.Task(dataset, row, split, question, choices, gold)


def test_scoring_and_normalisation():
    assert bench.score_text("mmlu", "because\nANSWER: A", "A")
    assert bench.score_text("mmlu", "answer: c", "C")
    assert bench.score_text("mmlu", "ANSWER: A\nANSWER: B", "B")
    assert not bench.score_text("mmlu", "ANSWER: A\nANSWER: B", "A")
    assert not bench.score_text("mmlu", "ANSWER: Apple", "A")
    assert not bench.score_text("mmlu", "I think it is A", "A")
    assert bench.score_text("gsm8k", "work\nANSWER: $1,234.50", "1234.50")
    assert bench.score_text("gsm8k", "ANSWER: 42.0", "42")
    assert bench.score_text("gsm8k", "ANSWER: 42%", "42")
    assert bench.score_text("gsm8k", "ANSWER: 1 000", "1000")
    assert bench.normalize_number("42.0") == Decimal(42)
    assert not bench.score_text("gsm8k", "the answer is 42", "42")
    assert bench._gold_gsm("calc\n#### 42") == "42"
    lengths = [_task("gsm8k", index, "dev", "x" * size) for index, size in enumerate((1, 2, 10))]
    assert bench.dev_median(lengths) == 2
    assert bench.rule_chooses_strong(_task("gsm8k", 1, "test", "hi"), 10) is False
    assert bench.rule_chooses_strong(_task("gsm8k", 1, "test", "hi"), 1) is True
    assert bench.rule_chooses_strong(_task("gsm8k", 1, "test", "```\ncode"), 1000) is True


def test_sample_is_stable_and_disjoint():
    first = bench.assign_splits()
    assert first == bench.assign_splits()
    for groups in first.values():
        assert len(groups["dev"]) == 30
        assert len(groups["test"]) == 120
        assert set(groups["dev"]).isdisjoint(groups["test"])
        assert groups["dev"] == sorted(groups["dev"])
        assert groups["test"] == sorted(groups["test"])
    limited = bench.assign_splits(1)
    assert len(limited["mmlu"]["dev"]) == 1
    assert limited["mmlu"]["dev"] == first["mmlu"]["dev"][:1]


def test_fetch_retries_429_and_parses_gsm8k():
    calls = []

    def get_json(url, timeout):
        del timeout
        calls.append(url)
        if len(calls) == 1:
            return bench.HttpResult(503, None)
        return bench.HttpResult(
            200,
            {
                "rows": [
                    {
                        "row_idx": 3,
                        "row": {"question": "What is 6*7?", "answer": "work\n#### 42"},
                    }
                ]
            },
        )

    found = bench.fetch_dataset_rows("gsm8k", [3], Transport(get=get_json), _no_sleep)
    task = bench.task_from_row("gsm8k", 3, "test", found[3])
    assert task.gold == "42"
    assert len(calls) == 2
    assert bench.index_batches([0, 50, 200]) == [(0, 51), (200, 1)]


def test_budget_above_five_is_refused():
    assert bench.main(["--budget-usd", "5.01", "--stage", "report"]) == 2


def test_preflight_refuses_before_any_paid_call(tmp_path):
    splits = bench.assign_splits(1)
    tasks = []
    for dataset, groups in splits.items():
        for split, indexes in groups.items():
            for index in indexes:
                tasks.append(_task(dataset, index, split, "Q" * 8000))
    cache = tmp_path / "cache"
    bench._write_tasks(cache / "tasks.json", tasks)
    transport = Transport()
    code = bench.main(
        [
            "--budget-usd",
            "0.001",
            "--limit",
            "1",
            "--stage",
            "all",
            "--cache",
            str(cache),
            "--results",
            str(tmp_path / "results"),
        ],
        transport=transport,
        sleep=_no_sleep,
    )
    assert code == 1
    assert transport.posts == []
    assert transport.gets == []
    assert not (tmp_path / "results").exists()


def test_runtime_stop_stays_at_the_cap(tmp_path):
    ledger = bench.SpendLedger(tmp_path / "spend.jsonl", Decimal("1.00"))
    calls = []
    lock = threading.Lock()

    def fn(index):
        def run():
            with lock:
                calls.append(index)
            return Decimal("0.25")

        return run

    jobs = [
        bench.Job(kind="generate", model="m", worst=Decimal("0.25"), fn=fn(index))
        for index in range(20)
    ]
    stopped = bench.run_jobs(jobs, ledger, workers=6)
    assert stopped
    assert len(calls) == 4
    assert ledger.spent == Decimal("1.00")
    assert ledger.peak <= Decimal("1.00")
    total = Decimal(0)
    for line in (tmp_path / "spend.jsonl").read_text().splitlines():
        total += Decimal(json.loads(line)["cost"])
    assert total == Decimal("1.00")


def test_ledger_counts_across_two_runs(tmp_path):
    path = tmp_path / "spend.jsonl"
    first = bench.SpendLedger(path, Decimal("0.50"))
    bench.run_jobs(
        [bench.Job("generate", "m", Decimal("0.30"), lambda: Decimal("0.30"))],
        first,
        workers=2,
    )
    second = bench.SpendLedger(path, Decimal("0.50"))
    assert second.spent == Decimal("0.30")
    stopped = bench.run_jobs(
        [bench.Job("generate", "m", Decimal("0.30"), lambda: Decimal("0.30"))],
        second,
        workers=2,
    )
    assert stopped
    assert second.spent == Decimal("0.30")
    assert len(path.read_text().splitlines()) == 1
    bench.run_jobs(
        [bench.Job("generate", "m", Decimal("0.20"), lambda: Decimal("0.20"))],
        second,
        workers=2,
    )
    assert second.spent == Decimal("0.50")
    assert second.peak <= Decimal("0.50")


def test_failed_call_is_charged_and_secret_is_not_written(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    monkeypatch.setenv("NACE_API_KEY", SECRET)
    cache = tmp_path / "cache"
    results = tmp_path / "results"
    splits = bench.assign_splits(1)
    tasks = []
    for dataset, groups in splits.items():
        for split, indexes in groups.items():
            for index in indexes:
                tasks.append(_task(dataset, index, split, "What is two plus two?"))
    bench._write_tasks(cache / "tasks.json", tasks)

    def post(url, headers, body, timeout):
        del url, body, timeout
        assert headers["Authorization"] == f"Bearer {SECRET}"
        raise RuntimeError(SECRET)

    code = bench.main(
        [
            "--budget-usd",
            "5",
            "--limit",
            "1",
            "--stage",
            "generate",
            "--cache",
            str(cache),
            "--results",
            str(results),
        ],
        transport=Transport(post=post),
        sleep=_no_sleep,
        selector=lambda *args, **kwargs: {},
        decider=lambda messages: (None, 0.0),
    )
    assert code == 0 or code == 1
    text = capsys.readouterr()
    blob = text.out + text.err
    for path in tmp_path.rglob("*"):
        if path.is_file():
            blob += path.read_text(encoding="utf-8", errors="ignore")
    assert SECRET not in blob
    lines = (cache / "spend.jsonl").read_text().splitlines()
    assert lines
    assert all(Decimal(json.loads(line)["cost"]) > 0 for line in lines)
    assert not list((cache / "generations").glob("*.json"))


def test_cache_prevents_a_second_paid_call(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-written")
    cache = tmp_path / "cache"
    task = _task("mmlu", 7, "test", "Capital of France?")
    bench._write_tasks(cache / "tasks.json", [task])
    calls = []

    def post(url, headers, body, timeout):
        del url, headers, body, timeout
        calls.append(1)
        return _ok_body()

    transport = Transport(post=post)
    jobs = bench.generation_jobs([task], cache, transport, _no_sleep)
    assert len(jobs) == len(bench.model_ids()) * bench.REPEATS
    ledger = bench.SpendLedger(cache / "spend.jsonl", Decimal("5"))
    assert bench.run_jobs(jobs, ledger, workers=4) is False
    assert len(calls) == len(jobs)
    again = bench.generation_jobs([task], cache, transport, _no_sleep)
    assert again == []
    bench.run_jobs(again, ledger, workers=4)
    assert len(calls) == len(jobs)


def test_bootstrap_is_deterministic():
    values = [0.0, 1.0, 0.5, 1.0]
    assert bench.bootstrap_mean(values) == bench.bootstrap_mean(values)
    assert bench.bootstrap_difference(values, [1.0, 0.0, 0.0, 0.0]) == bench.bootstrap_difference(
        values, [1.0, 0.0, 0.0, 0.0]
    )
    assert bench.percentile([1, 2, 3], 0.5) == 2


def test_selector_fallback_uses_the_documented_fields(tmp_path):
    task = _task("mmlu", 4, "test", "A short question")
    strong = bench.POOLS["A"]["strong"]

    def selector(messages, *, small, strong, priority, context):
        assert any(isinstance(message, HumanMessage) for message in messages)
        return {
            "model": None,
            "selected": None,
            "probabilities": {small: 0.2, strong: 0.8},
            "fallback": True,
            "reason": "timeout",
            "priority": priority,
            "candidates": [small, strong],
            "request_id": None,
            "latency_ms": 15,
            "context": context,
        }

    choice, local = bench._selector_choice(
        task, "A", "selector/balanced/conversation", selector, tmp_path
    )
    assert choice.chosen == strong
    assert choice.fallback is True
    assert choice.reason == "timeout"
    assert choice.probabilities[strong] == 0.8
    assert choice.latency_s == Decimal("0.015")
    assert local is False


def test_decision_router_uses_before_agent(monkeypatch, tmp_path):
    class _Classifier:
        def __init__(self, model=None):
            self.model = model or "drex-max"

    monkeypatch.setattr(
        "langchain_nace.experimental.middleware.model_router.NaceClassifier",
        _Classifier,
    )
    router = bench.build_router()
    assert router.question.criteria["small"] == bench.SMALL_CRITERIA
    assert router.question.criteria["strong"] == bench.STRONG_CRITERIA
    assert router.escalate_to == "strong"
    seen = {}

    def before_agent(self, state, runtime):
        del runtime
        seen["state"] = state
        return {
            "nace_model_route": {
                "route": "small",
                "choice": "small",
                "confidence": 0.9,
                "probabilities": {"small": 0.9, "strong": 0.1},
                "escalated": False,
                "reason": "confident",
            }
        }

    monkeypatch.setattr(bench.NaceModelRouterMiddleware, "before_agent", before_agent)
    bench._router = None
    route, elapsed = bench.default_decider([HumanMessage("How many days in a week?")])
    assert seen["state"]["messages"][0].content == "How many days in a week?"
    assert route["route"] == "small"
    assert elapsed >= 0
    choice = bench._decision_choice(
        _task("gsm8k", 1, "test", "How many days in a week?"),
        "A",
        lambda messages: (route, 0.01),
        tmp_path,
    )
    assert choice.chosen == bench.POOLS["A"]["small"]
    assert choice.fallback is False


def test_replay_oracle_and_gain(tmp_path):
    cache = tmp_path / "cache"
    small = bench.POOLS["A"]["small"]
    strong = bench.POOLS["A"]["strong"]
    dev = _task("gsm8k", 1, "dev", "x")
    test = _task("gsm8k", 2, "test", "yy")
    for model, correct, cost in ((small, False, "0.10"), (strong, True, "1.00")):
        for repeat in range(bench.REPEATS):
            bench._write_json(
                bench.generation_path(cache, model, "gsm8k", 2, repeat),
                {
                    "correct": correct,
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                    "cost_usd": cost,
                    "latency_s": "0.5",
                    "provider": "p",
                    "finish_reason": "stop",
                },
            )
    route = {
        "chosen": small,
        "fallback": False,
        "reason": "selected",
        "probabilities": {small: 0.6, strong: 0.4},
        "latency_s": "0.2",
        "routing_cost_usd": "0.001",
        "routing_tokens": "20",
    }
    arm = "selector/balanced/conversation"
    bench._write_json(bench.route_path(cache, "A", arm, "gsm8k", 2), route)
    # A task with a missing route is left out of every arm.
    partial = {a["name"]: a for a in bench.pool_report("A", [dev, test], cache)["arms"]}
    assert partial["fixed_small"]["n_tasks"] == 0
    assert partial[arm]["n_tasks"] == 0
    for other in (*bench.selector_arm_names(), "decision"):
        bench._write_json(bench.route_path(cache, "A", other, "gsm8k", 2), route)
    report = bench.pool_report("A", [dev, test], cache)
    arms = {arm["name"]: arm for arm in report["arms"]}
    assert arms["fixed_small"]["accuracy"] == 0
    assert arms["fixed_strong"]["accuracy"] == 1
    assert arms["oracle"]["accuracy"] == 1
    assert arms["oracle"]["strong_share"] == 1
    assert arms["selector/balanced/conversation"]["accuracy"] == 0
    assert arms["selector/balanced/conversation"]["quality_gain"] == 0
    assert arms["selector/balanced/conversation"]["strong_share"] == 0
    assert report["comparisons"]
