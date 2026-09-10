import asyncio
import logging
import os

from prometheus_client import start_http_server
from redis.asyncio import Redis

from .config import Settings
from .engine import BackendUnavailable, Busy, Engine
from .models import Generate
from .queue import JobQueue
from .telemetry import JOBS, configure, tracer

log = logging.getLogger(__name__)


async def process_one(queue, engine):
    claim = await queue.claim()
    if not claim:
        return False
    jid, token, payload = claim
    with tracer.start_as_current_span("job.process"):
        try:
            result = await engine.generate(Generate.model_validate(payload))
        except (TimeoutError, BackendUnavailable, Busy):
            # Leave the lease for bounded retry by recovery.
            JOBS.labels("retry").inc()
            return True
        except Exception:
            log.exception("Permanent job error: %s", jid)
            await queue.finish(jid, token, {"error": "Job processing failed"}, "failed")
            JOBS.labels("failed").inc()
            return True
        committed = await queue.finish(jid, token, result)
        JOBS.labels("succeeded" if committed else "stale").inc()
    return True


async def main():
    settings = Settings()
    configure("serving-worker")
    start_http_server(int(os.getenv("WORKER_METRICS_PORT", "9101")))
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    queue, engine = JobQueue(redis, settings), Engine(settings)
    await engine.warmup()

    async def consume():
        while True:
            try:
                if not await process_one(queue, engine):
                    await asyncio.sleep(0.2)
            except Exception:
                log.exception("Worker loop failed; retrying")
                await asyncio.sleep(1)

    async def recover():
        while True:
            try:
                await queue.recover()
            except Exception:
                log.exception("Lease recovery failed")
            await asyncio.sleep(2)

    try:
        async with asyncio.TaskGroup() as group:
            group.create_task(recover())
            for _ in range(settings.worker_concurrency):
                group.create_task(consume())
    finally:
        await engine.close()
        await redis.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
