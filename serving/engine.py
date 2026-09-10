import asyncio
import time
import uuid

import httpx

from .telemetry import INFLIGHT, LATENCY, REQUESTS, TOKENS, tracer


class Busy(Exception):
    pass


class BackendUnavailable(Exception):
    pass


class Engine:
    """Per-process admission control; least-inflight routing to independent replicas."""

    def __init__(self, settings, transport=None):
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=settings.timeout, transport=transport)
        self.active = 0
        self.loads = [0] * len(settings.urls)
        self.cooldown = [0.0] * len(settings.urls)
        self.cursor = 0

    async def close(self):
        await self.client.aclose()

    async def health(self):
        if self.settings.backend == "mock":
            return True

        async def check(url):
            try:
                return (await self.client.get(url + "/health", timeout=2)).is_success
            except httpx.HTTPError:
                return False

        return any(await asyncio.gather(*(check(u) for u in self.settings.urls)))

    async def warmup(self):
        if self.settings.backend == "mock":
            return

        async def warm(url):
            try:
                r = await self.client.post(
                    url + "/v1/chat/completions",
                    json={
                        "model": self.settings.model,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 1,
                    },
                )
                r.raise_for_status()
            except httpx.HTTPError:
                return False
            return True

        await asyncio.gather(*(warm(u) for u in self.settings.urls))

    async def generate(self, request):
        if self.active >= self.settings.max_inflight:
            raise Busy("Inference capacity reached")
        self.active += 1  # No await between check and increment on this event loop.
        INFLIGHT.inc()
        started = time.perf_counter()
        outcome = "error"
        try:
            with tracer.start_as_current_span("inference"):
                async with asyncio.timeout(self.settings.timeout):
                    result = await self._generate(request)
            TOKENS.labels(self.settings.backend).inc(result["usage"]["completion_tokens"])
            outcome = "success"
            return result
        finally:
            REQUESTS.labels(outcome, self.settings.backend).inc()
            LATENCY.observe(time.perf_counter() - started)
            self.active -= 1
            INFLIGHT.dec()

    async def _generate(self, request):
        if self.settings.backend == "mock":
            await asyncio.sleep(0.01)
            text = " ".join(
                ("Mock response: " + request.messages[-1].content).split()[: request.max_tokens]
            )
            return {
                "id": "mock-" + uuid.uuid4().hex,
                "model": "mock",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": len(text.split())},
                "mock": True,
            }
        eligible = [i for i in range(len(self.loads)) if self.cooldown[i] <= time.monotonic()]
        if not eligible:
            raise BackendUnavailable("All replicas are in cooldown")
        n = len(self.loads)
        idx = min(eligible, key=lambda i: (self.loads[i], (i - self.cursor) % n))
        self.cursor = (idx + 1) % n
        self.loads[idx] += 1
        try:
            response = await self.client.post(
                self.settings.urls[idx] + "/v1/chat/completions",
                json={"model": self.settings.model, **request.model_dump()},
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data.get("usage", {}).get("completion_tokens"), int) or not data.get(
                "choices"
            ):
                raise BackendUnavailable("Malformed backend response")
            return data
        except (httpx.HTTPError, ValueError) as exc:
            self.cooldown[idx] = time.monotonic() + 5
            raise BackendUnavailable("Replica request failed") from exc
        finally:
            self.loads[idx] -= 1
