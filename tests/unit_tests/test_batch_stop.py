"""Batch dispatch regressions, including failures whose callbacks yield."""

import asyncio
import json
import threading
import time
from collections import Counter
from concurrent.futures import CancelledError

import httpx
import pytest
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler

from langchain_nace import (
    NaceAPIError,
    NaceAuthenticationError,
    NaceBatchStoppedError,
    NaceClassifier,
    Noul,
)
from tests.fakes import error, ok_body

CALL = {"state": "hello", "questions": {"q": Noul(instructions="Greeting?")}}


class Runs(BaseCallbackHandler):
    run_inline = True

    def __init__(self):
        self.starts = []
        self.ends = []
        self.errors = []
        self.stopped = threading.Event()

    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        self.starts.append(run_id)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self.ends.append(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self.errors.append((run_id, error))
        self.stopped.set()
        time.sleep(0.005)

    def assert_complete(self, count):
        assert len(self.starts) == len(set(self.starts)) == count
        assert Counter(self.starts) == Counter(self.ends + [r for r, _ in self.errors])


class AsyncRuns(AsyncCallbackHandler):
    run_inline = True

    def __init__(self):
        self.starts = []
        self.ends = []
        self.errors = []
        self.stopped = asyncio.Event()

    async def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        self.starts.append(run_id)

    async def on_chain_end(self, outputs, *, run_id, **kwargs):
        self.ends.append(run_id)

    async def on_chain_error(self, error, *, run_id, **kwargs):
        self.errors.append((run_id, error))
        self.stopped.set()
        await asyncio.sleep(0.005)

    def assert_complete(self, count):
        assert len(self.starts) == len(set(self.starts)) == count
        assert Counter(self.starts) == Counter(self.ends + [r for r, _ in self.errors])


@pytest.mark.parametrize("status", [401, 402, 403, 422, 500])
@pytest.mark.parametrize("return_exceptions", [False, True])
def test_sync_batch_after_failure(status, return_exceptions):
    runs = Runs()
    requests = []
    barrier = threading.Barrier(2)
    stops = not return_exceptions or status in (401, 402, 403)

    def handler(request):
        # The first two requests are sent together; request 1 completes while
        # request 0's error callback is still running.
        assert not stops or not runs.stopped.is_set()
        index = len(requests)
        requests.append(request)
        if index < 2:
            barrier.wait(timeout=5)
        body = json.loads(request.content)
        if index == 0:
            return httpx.Response(status, json=error(status, "failed")[2])
        if index == 1 and (not return_exceptions or status in (401, 402, 403)):
            assert runs.stopped.wait(timeout=5)
        return httpx.Response(200, json=ok_body(body))

    classifier = NaceClassifier(
        api_key="nace_sk_test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    config = {"max_concurrency": 2, "callbacks": [runs]}
    if return_exceptions:
        results = classifier.batch([CALL] * 20, config, return_exceptions=True)
        stops = status in (401, 402, 403)
        assert sum(isinstance(r, NaceBatchStoppedError) for r in results) == (18 if stops else 0)
    else:
        with pytest.raises(Exception, match=f"returned {status}"):
            classifier.batch([CALL] * 20, config)
        stops = True
    assert len(requests) == (2 if stops else 20)
    at_return = len(requests)
    time.sleep(0.02)
    assert len(requests) == at_return
    runs.assert_complete(20)
    stopped = [e for _, e in runs.errors if isinstance(e, NaceBatchStoppedError)]
    assert len({id(e) for e in stopped}) == len(stopped)
    assert all(e.__cause__ is runs.errors[0][1] for e in stopped)


@pytest.mark.parametrize("status", [401, 402, 403, 422, 500])
@pytest.mark.parametrize("return_exceptions", [False, True])
async def test_async_batch_after_failure(status, return_exceptions):
    runs = AsyncRuns()
    requests = []
    two_sent = asyncio.Event()
    stops = not return_exceptions or status in (401, 402, 403)

    async def handler(request):
        assert not stops or not runs.stopped.is_set()
        index = len(requests)
        requests.append(request)
        if len(requests) == 2:
            two_sent.set()
        await two_sent.wait()
        if index == 0:
            return httpx.Response(status, json=error(status, "failed")[2])
        if index == 1 and stops:
            await runs.stopped.wait()
        return httpx.Response(200, json=ok_body(json.loads(request.content)))

    classifier = NaceClassifier(
        api_key="nace_sk_test",
        max_retries=0,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    config = {"max_concurrency": 2, "callbacks": [runs]}
    if return_exceptions:
        results = await classifier.abatch([CALL] * 20, config, return_exceptions=True)
        assert sum(isinstance(r, NaceBatchStoppedError) for r in results) == (18 if stops else 0)
    else:
        with pytest.raises(Exception, match=f"returned {status}"):
            await classifier.abatch([CALL] * 20, config)
    assert len(requests) == (2 if stops else 20)
    at_return = len(requests)
    await asyncio.sleep(0.02)
    assert len(requests) == at_return
    runs.assert_complete(20)
    stopped = [e for _, e in runs.errors if isinstance(e, NaceBatchStoppedError)]
    assert len({id(e) for e in stopped}) == len(stopped)
    assert all(e.__cause__ is runs.errors[0][1] for e in stopped)


@pytest.mark.parametrize("return_exceptions", [False, True])
@pytest.mark.parametrize("self_cancel", [False, True])
async def test_input_cancellation_is_local(return_exceptions, self_cancel):
    requests = []
    runs = AsyncRuns()

    async def handler(request):
        requests.append(request)
        if len(requests) == 1:
            if self_cancel:
                asyncio.current_task().cancel()
                await asyncio.sleep(0)
            raise asyncio.CancelledError
        return httpx.Response(200, json=ok_body(json.loads(request.content)))

    classifier = NaceClassifier(
        api_key="nace_sk_test",
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    config = {"max_concurrency": 1, "callbacks": [runs]}
    if return_exceptions:
        results = await classifier.abatch([CALL] * 4, config, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert all(r.nouls["q"].noul == 0.9 for r in results[1:])
    else:
        with pytest.raises(asyncio.CancelledError):
            await classifier.abatch([CALL] * 4, config)
    assert len(requests) == 4
    runs.assert_complete(4)


async def test_abatch_cancellation_propagates():
    entered = asyncio.Event()
    finished = asyncio.Event()
    requests = []

    async def handler(request):
        requests.append(request)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    classifier = NaceClassifier(
        api_key="nace_sk_test",
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    task = asyncio.create_task(classifier.abatch([CALL] * 4, {"max_concurrency": 1}))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()
    assert len(requests) == 1


async def test_subclass_overrides_and_kwargs_are_honored(make_classifier):
    seen = []

    class Custom(NaceClassifier):
        def invoke(self, input, config=None, **kwargs):
            seen.append(("sync", kwargs["marker"], input["state"]))
            return super().invoke(input, config)

        async def ainvoke(self, input, config=None, **kwargs):
            seen.append(("async", kwargs["marker"], input["state"]))
            return await super().ainvoke(input, config)

    classifier = Custom(api_key="nace_sk_test", base_url="https://drex.test")
    classifier._client = make_classifier()._client
    classifier._aclient = make_classifier()._aclient
    inputs = [{**CALL, "state": str(i)} for i in range(3)]
    configs = [{"max_concurrency": 1, "tags": [str(i)]} for i in range(3)]
    assert len(classifier.batch(inputs, configs, marker="value")) == 3
    assert len(await classifier.abatch(inputs, configs, marker="value")) == 3
    assert seen == [(kind, "value", str(i)) for kind in ("sync", "async") for i in range(3)]


async def test_already_sent_request_finishes_sdk_retries(sleeps):
    runs = AsyncRuns()
    sent = asyncio.Event()
    requests = []
    attempts = 0

    async def handler(request):
        nonlocal attempts
        body = json.loads(request.content)
        requests.append(body["state"])
        if body["state"] == "fail":
            await sent.wait()
            return httpx.Response(401, json=error(401, "bad key")[2])
        attempts += 1
        if attempts == 1:
            sent.set()
            await runs.stopped.wait()
            return httpx.Response(529, json=error(529, "busy")[2])
        return httpx.Response(200, json=ok_body(body))

    classifier = NaceClassifier(
        api_key="nace_sk_test",
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(NaceAuthenticationError):
        await classifier.abatch(
            [{**CALL, "state": "fail"}, {**CALL, "state": "retry"}] + [CALL] * 3,
            {"max_concurrency": 2, "callbacks": [runs]},
        )
    assert requests == ["fail", "retry", "retry"]
    runs.assert_complete(5)


async def test_batch_validation_and_empty_inputs(classifier):
    assert classifier.batch([]) == []
    assert await classifier.abatch([]) == []
    with pytest.raises(ValueError, match="same length"):
        classifier.batch([CALL] * 2, [{}])
    with pytest.raises(ValueError, match="same length"):
        await classifier.abatch([CALL] * 2, [{}])


@pytest.mark.parametrize("return_exceptions", [False, True])
async def test_local_validation_failure_obeys_stop_policy(classifier, drex, return_exceptions):
    invalid = {"state": "bad", "questions": {}}
    runs = AsyncRuns()
    config = {"max_concurrency": 1, "callbacks": [runs]}
    if return_exceptions:
        results = await classifier.abatch([invalid, CALL], config, return_exceptions=True)
        assert isinstance(results[0], ValueError)
        assert results[1].nouls["q"].noul == 0.9
        assert len(drex.requests) == 1
    else:
        with pytest.raises(ValueError, match="At least one question"):
            await classifier.abatch([invalid, CALL], config)
        assert not drex.requests
        assert isinstance(runs.errors[1][1], NaceBatchStoppedError)
    runs.assert_complete(2)


async def test_concurrent_batches_and_later_invokes_are_independent(classifier, drex):
    drex.replies.append(error(401, "bad key"))
    failed, successful = await asyncio.gather(
        classifier.abatch([CALL] * 4, {"max_concurrency": 1}, return_exceptions=True),
        classifier.abatch([CALL] * 4, {"max_concurrency": 1}, return_exceptions=True),
    )
    assert isinstance(failed[0], NaceAuthenticationError)
    assert all(isinstance(r, NaceBatchStoppedError) for r in failed[1:])
    assert all(r.nouls["q"].noul == 0.9 for r in successful)
    assert (await classifier.ainvoke(CALL)).nouls["q"].noul == 0.9


async def test_negative_concurrency_is_rejected(classifier):
    with pytest.raises(ValueError, match="max_concurrency must be positive"):
        classifier.batch([CALL], {"max_concurrency": -1})
    with pytest.raises(ValueError, match="max_concurrency must be positive"):
        await classifier.abatch([CALL], {"max_concurrency": -1})


DISPATCH_REQUEST = {"state": "ok", "questions": {"q": Noul(instructions="Safe?")}}


def test_stop_and_sdk_dispatch_have_one_atomic_boundary(make_classifier, drex, monkeypatch):
    import langchain_nace.classifier as module

    late_checked = threading.Event()
    recording = threading.Event()
    recorded = threading.Event()
    local = threading.local()
    check = module._BatchState.check
    record = module._BatchState.record
    prepare = NaceClassifier._prepare

    def prepared(self, request):
        local.late = request["state"] == "late"
        return prepare(self, request)

    def checked(self):
        check(self)
        if getattr(local, "late", False):
            late_checked.set()
            assert recording.wait(5)
            # An atomic implementation may keep record() waiting until the
            # SDK call is handed off. A non-atomic implementation records now.
            recorded.wait(0.05)

    def recording_error(self, failure):
        recording.set()
        record(self, failure)
        recorded.set()

    classifier = make_classifier(max_retries=0)
    sdk_call = classifier._client.system_one
    sent = []

    def response(body):
        from tests.fakes import ok_body

        if body["state"] == "fail":
            assert late_checked.wait(5)
            return error(401, "invalid")
        return 200, {}, ok_body(body)

    drex.replies.extend([response, response])

    def send(**kwargs):
        if kwargs["state"] == "late":
            sent.append(recorded.is_set())
        return sdk_call(**kwargs)

    monkeypatch.setattr(NaceClassifier, "_prepare", prepared)
    monkeypatch.setattr(module._BatchState, "check", checked)
    monkeypatch.setattr(module._BatchState, "record", recording_error)
    monkeypatch.setattr(classifier._client, "system_one", send)
    classifier.batch(
        [{**DISPATCH_REQUEST, "state": "fail"}, {**DISPATCH_REQUEST, "state": "late"}],
        {"max_concurrency": 2},
        return_exceptions=True,
    )
    assert sent == [False]


def test_sync_input_cancellation_does_not_stop_batch(make_classifier):
    seen = []

    class Cancelling(NaceClassifier):
        def invoke(self, input, config=None, **kwargs):
            seen.append(input["state"])
            if input["state"] == "cancel":
                raise CancelledError()
            return super().invoke(input, config, **kwargs)

    base = make_classifier()
    classifier = Cancelling(api_key="nace_sk_test")
    classifier._client = base._client
    inputs = [{**DISPATCH_REQUEST, "state": value} for value in ("cancel", "one", "two")]
    with pytest.raises(CancelledError):
        classifier.batch(inputs, {"max_concurrency": 1})
    assert seen == ["cancel", "one", "two"]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("status", [401, 429])
async def test_recovered_override_does_not_record_batch_stop(
    make_classifier, drex, asynchronous, status
):
    class Recovering(NaceClassifier):
        def invoke(self, input, config=None, **kwargs):
            try:
                return super().invoke(input, config, **kwargs)
            except NaceAPIError:
                return super().invoke(input, config, **kwargs)

        async def ainvoke(self, input, config=None, **kwargs):
            try:
                return await super().ainvoke(input, config, **kwargs)
            except NaceAPIError:
                return await super().ainvoke(input, config, **kwargs)

    base = make_classifier(max_retries=0)
    classifier = Recovering(api_key="nace_sk_test", max_retries=0)
    classifier._client, classifier._aclient = base._client, base._aclient
    drex.replies.append(error(status, "try fallback"))
    results = (
        await classifier.abatch([DISPATCH_REQUEST] * 3, {"max_concurrency": 1})
        if asynchronous
        else classifier.batch([DISPATCH_REQUEST] * 3, {"max_concurrency": 1})
    )
    assert len(results) == 3
    assert all(response.nouls["q"].noul == 0.9 for response in results)
    assert len(drex.requests) == 4


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("where", ["start", "end"])
async def test_callback_failure_stops_dispatch(make_classifier, drex, asynchronous, where):
    class Failing(BaseCallbackHandler):
        raise_error = True
        run_inline = True
        failed = False

        def on_chain_start(self, serialized, inputs, **kwargs):
            if where == "start" and not self.failed:
                self.failed = True
                raise ValueError("callback failed")

        def on_chain_end(self, outputs, **kwargs):
            if where == "end" and not self.failed:
                self.failed = True
                raise ValueError("callback failed")

    classifier = make_classifier()
    config = {"callbacks": [Failing()], "max_concurrency": 1}

    async def batch():
        return (
            await classifier.abatch([DISPATCH_REQUEST] * 3, config)
            if asynchronous
            else classifier.batch([DISPATCH_REQUEST] * 3, config)
        )

    with pytest.raises(ValueError, match="callback failed"):
        await batch()
    assert len(drex.requests) == (0 if where == "start" else 1)
