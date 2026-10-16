import asyncio
import json

import httpx
import pytest
from fakeredis.aioredis import FakeRedis
from fastapi.testclient import TestClient

from serving.api import create_app
from serving.config import Settings
from serving.engine import BackendUnavailable, Busy, Engine
from serving.models import Generate
from serving.queue import IdempotencyConflict, JobQueue, QueueFull
from serving.worker import process_one


def request(text="hello"):
    return Generate(messages=[{"role": "user", "content": text}])


@pytest.fixture
async def queue():
    redis = FakeRedis(decode_responses=True)
    yield JobQueue(redis, Settings(queue_capacity=2))
    await redis.aclose()


async def test_idempotency_and_capacity(queue):
    ids = await asyncio.gather(*(queue.submit(request(), "same") for _ in range(20)))
    assert len(set(ids)) == 1
    with pytest.raises(IdempotencyConflict):
        await queue.submit(request("different"), "same")
    await queue.submit(request())
    with pytest.raises(QueueFull):
        await queue.submit(request())


async def test_crash_recovery_fences_stale_worker(queue):
    jid = await queue.submit(request(), "key")
    _, old_token, _ = await queue.claim(now=0)
    await queue.recover(now=100)
    _, token, _ = await queue.claim(now=100)
    assert not await queue.finish(jid, old_token, {"wrong": True})
    assert await queue.finish(jid, token, {"ok": True})
    assert not await queue.finish(jid, token, {"duplicate": True})
    assert (await queue.get(jid))["result"] == {"ok": True}
    assert int(await queue.redis.get(queue.active)) == 0


async def test_poison_job_retry_budget(queue):
    jid = await queue.submit(request())
    for i in range(3):
        await queue.claim(now=i * 100)
        await queue.recover(now=(i + 1) * 100)
    assert (await queue.get(jid))["status"] == "failed"
    assert await queue.claim() is None
    assert int(await queue.redis.get(queue.active)) == 0


async def test_worker_end_to_end(queue):
    jid = await queue.submit(request())
    engine = Engine(Settings())
    try:
        assert await process_one(queue, engine)
        result = await queue.get(jid)
        assert result["status"] == "succeeded"
        assert result["result"]["mock"] is True
    finally:
        await engine.close()


async def test_admission_and_cleanup():
    engine = Engine(Settings(max_inflight=1))
    first = asyncio.create_task(engine.generate(request()))
    await asyncio.sleep(0)
    with pytest.raises(Busy):
        await engine.generate(request())
    await first
    assert engine.active == 0
    await engine.close()


async def test_replica_failure_then_routes_to_other():
    seen = []

    def handler(req):
        seen.append(req.url.host)
        if req.url.host == "bad":
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 1}},
        )

    engine = Engine(
        Settings(backend="vllm", urls=["http://bad", "http://good"]), httpx.MockTransport(handler)
    )
    with pytest.raises(BackendUnavailable):
        await engine.generate(request())
    await engine.generate(request())
    assert seen == ["bad", "good"]
    assert engine.loads == [0, 0]
    await engine.close()


def test_api_auth_validation_and_jobs():
    redis = FakeRedis(decode_responses=True)
    with TestClient(create_app(Settings(api_key="test-secret"), redis=redis)) as client:
        assert client.post("/v1/chat/completions", json={}).status_code == 401
        headers = {"Authorization": "Bearer test-secret", "Idempotency-Key": "a"}
        assert client.post("/v1/chat/completions", json={}, headers=headers).status_code == 422
        data = request().model_dump()
        assert client.post("/v1/chat/completions", json=data, headers=headers).json()["mock"]
        response = client.post("/v1/jobs", json=data, headers=headers)
        assert response.status_code == 202
        assert (
            client.get(response.json()["status_url"], headers=headers).json()["status"] == "queued"
        )
        assert client.get("/health/ready").status_code == 200
        assert client.get("/metrics").status_code == 200


async def test_timeout_releases_admission():
    async def slow(req):
        await asyncio.sleep(1)
        return httpx.Response(200)

    engine = Engine(Settings(backend="vllm", timeout=0.01), httpx.MockTransport(slow))
    with pytest.raises(TimeoutError):
        await engine.generate(request())
    assert engine.active == 0
    assert engine.loads == [0, 0]
    await engine.close()


async def test_worker_failure_retains_lease_for_retry(queue):
    engine = Engine(Settings(backend="vllm"), httpx.MockTransport(lambda _: httpx.Response(503)))
    jid = await queue.submit(request())
    assert await process_one(queue, engine)
    assert (await queue.get(jid))["status"] == "running"
    assert int(await queue.redis.get(queue.active)) == 1
    await engine.close()


async def test_busy_worker_requeues_without_spending_attempt(queue):
    engine = Engine(Settings(max_inflight=1))
    engine.active = 1  # simulate a saturated process
    jid = await queue.submit(request())
    assert await process_one(queue, engine)
    job = await queue.get(jid)
    assert job["status"] == "queued" and job["attempts"] == 0
    assert await queue.redis.zcard(queue.leases) == 0
    engine.active = 0
    assert await process_one(queue, engine)
    job = await queue.get(jid)
    assert job["status"] == "succeeded" and job["attempts"] == 1
    await engine.close()


async def test_release_is_fenced(queue):
    jid = await queue.submit(request())
    _, token, _ = await queue.claim(now=0)
    assert not await queue.release(jid, "stale-token")
    assert await queue.release(jid, token)
    assert not await queue.release(jid, token)


def parse_sse(body):
    events = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    return [json.loads(e) for e in events[:-1]]


def test_api_streams_mock_and_rejects_streaming_jobs():
    with TestClient(create_app(Settings(), redis=FakeRedis(decode_responses=True))) as client:
        data = {**request("one two three").model_dump(), "stream": True}
        r = client.post("/v1/chat/completions", json=data)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        chunks = parse_sse(r.text)
        text = "".join(c["choices"][0]["delta"]["content"] for c in chunks if c["choices"])
        assert text == "Mock response: one two three"
        assert chunks[-1]["usage"]["completion_tokens"] == 5
        assert client.post("/v1/jobs", json=data).status_code == 422
        assert "serving_ttft_seconds_count" in client.get("/metrics").text


async def test_vllm_stream_relay_and_cleanup():
    sent = []
    upstream = (
        'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(req):
        sent.append(json.loads(req.content))
        return httpx.Response(200, text=upstream, headers={"content-type": "text/event-stream"})

    engine = Engine(Settings(backend="vllm", urls=["http://a"]), httpx.MockTransport(handler))
    chunks = [c async for c in await engine.stream(request())]
    assert chunks[-1] == "data: [DONE]\n\n"
    assert sent[0]["stream"] is True and sent[0]["stream_options"] == {"include_usage": True}
    assert engine.active == 0 and engine.loads == [0]
    await engine.close()


async def test_vllm_stream_connect_failure_is_503_not_broken_stream():
    engine = Engine(
        Settings(backend="vllm", urls=["http://a"]),
        httpx.MockTransport(lambda _: httpx.Response(500)),
    )
    with pytest.raises(BackendUnavailable):
        await engine.stream(request())
    assert engine.active == 0 and engine.loads == [0]
    assert engine.cooldown[0] > 0
    await engine.close()
