import pytest
from langchain_core.tracers.context import collect_runs, tracing_v2_enabled
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client

from langchain_nace import NaceClassifier, Noul
from tests.fakes import error
from tests.unit_tests.test_batch_stop import DISPATCH_REQUEST


def test_traced_as_llm_run_with_probabilities(classifier: NaceClassifier) -> None:
    with collect_runs() as cb:
        classifier.invoke(
            {"state": "x", "questions": {"q": Noul(instructions="?")}},
            config={"metadata": {"nace_threshold": 0.7}},
        )
    run = cb.traced_runs[0]
    assert run.run_type == "llm"
    metadata = run.extra["metadata"]
    assert metadata["ls_provider"] == "nace"
    assert metadata["ls_model_name"] == "drex-max"
    assert metadata["nace_threshold"] == 0.7
    assert run.outputs["output"].answers["q"].noul == 0.9
    assert run.inputs["questions"]["q"].instructions == "?"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("failed", [False, True])
async def test_env_style_tracing_has_one_run_per_batch_input(
    make_classifier, drex, monkeypatch, asynchronous, failed
):
    starts = []
    original = LangChainTracer.on_chain_start

    def start(self, serialized, inputs, **kwargs):
        starts.append(kwargs["run_id"])
        return original(self, serialized, inputs, **kwargs)

    monkeypatch.setattr(LangChainTracer, "on_chain_start", start)
    # The real env-style tracer participates; persistence is replaced to keep
    # this a local test rather than sending traces to a service.
    monkeypatch.setattr(Client, "create_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(Client, "update_run", lambda *args, **kwargs: None)
    client = Client(api_key="test", api_url="https://traces.test", auto_batch_tracing=False)
    classifier = make_classifier(max_retries=0)
    if failed:
        drex.replies.append(error(401, "invalid"))
    with tracing_v2_enabled(client=client):
        if asynchronous:
            await classifier.abatch(
                [DISPATCH_REQUEST] * 3, {"max_concurrency": 1}, return_exceptions=True
            )
        else:
            classifier.batch([DISPATCH_REQUEST] * 3, {"max_concurrency": 1}, return_exceptions=True)
    assert len(starts) == len(set(starts)) == 3
