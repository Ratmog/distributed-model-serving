"""Closed-loop load test; reports failures separately, never invents GPU metrics."""

import argparse
import asyncio
import json
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path

import httpx


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


async def run(args):
    if min(args.requests, args.concurrency) <= 0:
        raise ValueError("requests and concurrency must be positive")
    headers = {"Authorization": "Bearer " + os.getenv("API_KEY", "")}
    latencies, tokens, errors = [], [], Counter()
    models = set()
    pending = iter(range(args.requests))
    async with httpx.AsyncClient(
        base_url=args.url,
        headers=headers,
        timeout=120,
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as client:

        async def worker():
            for i in pending:
                start = time.perf_counter()
                try:
                    r = await client.post(
                        "/v1/chat/completions",
                        json={
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"Explain database indexing. Request {i}.",
                                }
                            ],
                            "max_tokens": args.max_tokens,
                        },
                    )
                    if r.status_code != 200:
                        errors[str(r.status_code)] += 1
                        continue
                    data = r.json()
                    tokens.append(data["usage"]["completion_tokens"])
                    models.add(data["model"])
                    latencies.append(time.perf_counter() - start)
                except (httpx.HTTPError, ValueError, KeyError):
                    errors["transport_or_schema"] += 1

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed = time.perf_counter() - started
    result = {
        "kind": "closed_loop_load",
        "models": sorted(models),
        "client_platform": platform.platform(),
        "configuration": vars(args),
        "elapsed_seconds": elapsed,
        "successful": len(latencies),
        "errors": dict(errors),
        "error_rate": (args.requests - len(latencies)) / args.requests,
        "requests_per_second": len(latencies) / elapsed,
        "output_tokens_per_second": sum(tokens) / elapsed,
        "latency_seconds": {f"p{p}": percentile(latencies, p / 100) for p in (50, 95, 99)},
        "note": "Success-only latency. Mock output uses whitespace counts, not model tokens. No TTFT measurement.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result["error_rate"] > args.max_error_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--output", default="results/load.json")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
