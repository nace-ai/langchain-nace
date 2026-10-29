from langchain_core.tracers.context import collect_runs

from langchain_nace import NaceClassifier, Noul


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
