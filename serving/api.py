import hmac
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .config import Settings
from .engine import BackendUnavailable, Busy, Engine
from .models import Generate
from .queue import IdempotencyConflict, JobQueue, QueueFull
from .telemetry import QUEUE, configure


def create_app(settings=None, redis=None, engine=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        configure("serving-api")
        app.state.redis = (
            redis
            if redis is not None
            else Redis.from_url(settings.redis_url, decode_responses=True)
        )
        app.state.engine = engine if engine is not None else Engine(settings)
        app.state.queue = JobQueue(app.state.redis, settings)
        await app.state.engine.warmup()
        yield
        await app.state.engine.close()
        if redis is None:
            await app.state.redis.aclose()

    app = FastAPI(title="Distributed Model Serving", version="0.1.0", lifespan=lifespan)

    async def auth(authorization: str = Header(default="")):
        if settings.api_key and not hmac.compare_digest(
            authorization, "Bearer " + settings.api_key
        ):
            raise HTTPException(401, "Invalid API key")

    @app.get("/health/live")
    async def live():
        return {"status": "alive", "backend": settings.backend}

    @app.get("/health/ready")
    async def ready():
        try:
            await app.state.redis.ping()
        except RedisError:
            raise HTTPException(503, "Redis unavailable")
        if not await app.state.engine.health():
            raise HTTPException(503, "No healthy inference replica")
        return {"status": "ready", "backend": settings.backend}

    @app.get("/metrics")
    async def metrics():
        try:
            QUEUE.set(int(await app.state.redis.get(JobQueue.active) or 0))
        except RedisError:
            QUEUE.set(float("nan"))
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/chat/completions", dependencies=[Depends(auth)])
    async def generate(request: Generate):
        try:
            return await app.state.engine.generate(request)
        except Busy:
            raise HTTPException(429, "Inference capacity reached", headers={"Retry-After": "1"})
        except (TimeoutError, BackendUnavailable):
            raise HTTPException(503, "Inference backend unavailable or timed out")

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(auth)])
    async def submit(
        request: Generate, idempotency_key: str | None = Header(default=None, max_length=128)
    ):
        try:
            jid = await app.state.queue.submit(request, idempotency_key)
            return {"id": jid, "status_url": "/v1/jobs/" + jid}
        except QueueFull:
            raise HTTPException(429, "Job capacity reached", headers={"Retry-After": "5"})
        except IdempotencyConflict:
            raise HTTPException(409, "Idempotency key reused with a different payload")
        except RedisError:
            raise HTTPException(503, "Job queue unavailable")

    @app.get("/v1/jobs/{jid}", dependencies=[Depends(auth)])
    async def job(jid: str):
        if len(jid) != 32 or any(c not in "0123456789abcdef" for c in jid):
            raise HTTPException(404, "Job not found")
        try:
            value = await app.state.queue.get(jid)
        except RedisError:
            raise HTTPException(503, "Job queue unavailable")
        if value is None:
            raise HTTPException(404, "Job not found or expired")
        return value

    return app


app = create_app()
