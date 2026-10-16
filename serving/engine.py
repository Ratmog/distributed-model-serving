import asyncio
import json
import time
import uuid

import httpx

from .telemetry import INFLIGHT, LATENCY, REQUESTS, TOKENS, TTFT, tracer


class Busy(Exception):
    pass


class BackendUnavailable(Exception):
    pass


def sse(data):
    return "data: " + (data if isinstance(data, str) else json.dumps(data)) + "\n\n"


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

    # Admission -----------------------------------------------------------------

    def _admit(self):
        if self.active >= self.settings.max_inflight:
            raise Busy("Inference capacity reached")
        self.active += 1  # No await between check and increment on this event loop.
        INFLIGHT.inc()
        return time.perf_counter()

    def _finish(self, started, outcome):
        REQUESTS.labels(outcome, self.settings.backend).inc()
        LATENCY.observe(time.perf_counter() - started)
        self.active -= 1
        INFLIGHT.dec()

    # Routing -------------------------------------------------------------------

    def _pick(self):
        eligible = [i for i in range(len(self.loads)) if self.cooldown[i] <= time.monotonic()]
        if not eligible:
            raise BackendUnavailable("All replicas are in cooldown")
        n = len(self.loads)
        idx = min(eligible, key=lambda i: (self.loads[i], (i - self.cursor) % n))
        self.cursor = (idx + 1) % n
        self.loads[idx] += 1
        return idx

    def _fail(self, idx):
        self.cooldown[idx] = time.monotonic() + 5

    def _body(self, request, stream):
        body = {"model": self.settings.model, **request.model_dump(exclude={"stream"})}
        if stream:
            body |= {"stream": True, "stream_options": {"include_usage": True}}
        return body

    # Non-streaming -------------------------------------------------------------

    async def generate(self, request):
        started = self._admit()
        outcome = "error"
        try:
            with tracer.start_as_current_span("inference"):
                async with asyncio.timeout(self.settings.timeout):
                    result = await self._generate(request)
            TOKENS.labels(self.settings.backend).inc(result["usage"]["completion_tokens"])
            outcome = "success"
            return result
        finally:
            self._finish(started, outcome)

    async def _generate(self, request):
        if self.settings.backend == "mock":
            await asyncio.sleep(0.01)
            text = " ".join(self._mock_words(request))
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
        idx = self._pick()
        try:
            response = await self.client.post(
                self.settings.urls[idx] + "/v1/chat/completions",
                json=self._body(request, stream=False),
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data.get("usage", {}).get("completion_tokens"), int) or not data.get(
                "choices"
            ):
                raise BackendUnavailable("Malformed backend response")
            return data
        except (httpx.HTTPError, ValueError) as exc:
            self._fail(idx)
            raise BackendUnavailable("Replica request failed") from exc
        finally:
            self.loads[idx] -= 1

    # Streaming -----------------------------------------------------------------

    async def stream(self, request):
        """Admit and connect before returning, so capacity and connect failures map to
        429/503 instead of a broken 200 stream. Returns an async iterator of SSE strings."""
        started = self._admit()
        deadline = time.monotonic() + self.settings.timeout
        if self.settings.backend == "mock":
            return self._relay_mock(request, started)
        idx = None
        try:
            idx = self._pick()
            async with asyncio.timeout(self.settings.timeout):
                response = await self.client.send(
                    self.client.build_request(
                        "POST",
                        self.settings.urls[idx] + "/v1/chat/completions",
                        json=self._body(request, stream=True),
                    ),
                    stream=True,
                )
            if not response.is_success:
                await response.aclose()
                raise BackendUnavailable(f"Replica returned {response.status_code}")
        except BaseException as exc:
            if idx is not None:
                if isinstance(exc, (httpx.HTTPError, BackendUnavailable)):
                    self._fail(idx)
                self.loads[idx] -= 1
            self._finish(started, "error")
            if isinstance(exc, httpx.HTTPError):
                raise BackendUnavailable("Replica connection failed") from exc
            raise
        return self._relay(response, idx, started, deadline)

    async def _relay(self, response, idx, started, deadline):
        outcome, first = "error", True
        try:
            with tracer.start_as_current_span("inference.stream"):
                async for line in response.aiter_lines():
                    if time.monotonic() > deadline:
                        yield sse({"error": {"message": "Request timed out"}})
                        return
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        outcome = "success"
                        yield sse("[DONE]")
                        return
                    chunk = json.loads(data)
                    if first and any(
                        c.get("delta", {}).get("content") for c in chunk.get("choices", [])
                    ):
                        TTFT.observe(time.perf_counter() - started)
                        first = False
                    usage = chunk.get("usage") or {}
                    if isinstance(usage.get("completion_tokens"), int):
                        TOKENS.labels(self.settings.backend).inc(usage["completion_tokens"])
                    yield sse(chunk)
        except (httpx.HTTPError, ValueError):
            self._fail(idx)
            yield sse({"error": {"message": "Replica stream failed"}})
        finally:
            await response.aclose()
            self.loads[idx] -= 1
            self._finish(started, outcome)

    async def _relay_mock(self, request, started):
        outcome, cid = "error", "mock-" + uuid.uuid4().hex
        try:
            words = self._mock_words(request)
            for i, word in enumerate(words):
                await asyncio.sleep(0.002)
                if i == 0:
                    TTFT.observe(time.perf_counter() - started)
                content = word if i == 0 else " " + word
                yield sse(
                    {
                        "id": cid,
                        "model": "mock",
                        "choices": [{"index": 0, "delta": {"content": content}}],
                        "mock": True,
                    }
                )
            TOKENS.labels("mock").inc(len(words))
            yield sse(
                {
                    "id": cid,
                    "model": "mock",
                    "choices": [],
                    "usage": {"completion_tokens": len(words)},
                    "mock": True,
                }
            )
            outcome = "success"
            yield sse("[DONE]")
        finally:
            self._finish(started, outcome)

    @staticmethod
    def _mock_words(request):
        return ("Mock response: " + request.messages[-1].content).split()[: request.max_tokens]
