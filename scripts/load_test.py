"""Closed-loop load test; reports failures separately, never invents GPU metrics.

With --stream, also measures time to first token (TTFT) and time per output token (TPOT)
from the client side.
"""

import argparse
import asyncio
import json
import math
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from _auth import auth_headers

FILLER = (
    "Context: indexes trade write cost and storage for faster reads; B-trees keep keys "
    "sorted, hash indexes support equality lookups, and covering indexes avoid table reads. "
)


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else None


def prompt(i, words):
    padding = (FILLER * (words // len(FILLER.split()) + 1)).split()[:words] if words else []
    return " ".join(padding + [f"Explain database indexing. Request {i}."])


async def one_plain(client, body):
    r = await client.post("/v1/chat/completions", json=body)
    if r.status_code != 200:
        return str(r.status_code), None
    data = r.json()
    return None, {"tokens": data["usage"]["completion_tokens"], "model": data["model"]}


async def one_stream(client, body, start):
    ttft, tokens, model = None, None, None
    async with client.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as r:
        if r.status_code != 200:
            await r.aread()
            return str(r.status_code), None
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                return "stream_error", None
            model = chunk.get("model", model)
            if ttft is None and any(
                c.get("delta", {}).get("content") for c in chunk.get("choices", [])
            ):
                ttft = time.perf_counter() - start
            if (chunk.get("usage") or {}).get("completion_tokens") is not None:
                tokens = chunk["usage"]["completion_tokens"]
        else:
            return "stream_truncated", None
    if tokens is None or ttft is None:
        return "stream_schema", None
    return None, {"tokens": tokens, "model": model, "ttft": ttft}


async def run(args):
    stream = getattr(args, "stream", False)
    prompt_words = getattr(args, "prompt_words", 0)
    if min(args.requests, args.concurrency) <= 0:
        raise ValueError("requests and concurrency must be positive")
    headers = auth_headers()
    latencies, tokens, ttfts, tpots, errors = [], [], [], [], Counter()
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
                body = {
                    "messages": [{"role": "user", "content": prompt(i, prompt_words)}],
                    "max_tokens": args.max_tokens,
                }
                start = time.perf_counter()
                try:
                    if stream:
                        error, ok = await one_stream(client, body, start)
                    else:
                        error, ok = await one_plain(client, body)
                except (httpx.HTTPError, ValueError, KeyError):
                    error, ok = "transport_or_schema", None
                if error:
                    errors[error] += 1
                    continue
                latency = time.perf_counter() - start
                latencies.append(latency)
                tokens.append(ok["tokens"])
                models.add(ok["model"])
                if stream:
                    ttfts.append(ok["ttft"])
                    if ok["tokens"] > 1:
                        tpots.append((latency - ok["ttft"]) / (ok["tokens"] - 1))

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
        "mean_output_tokens": sum(tokens) / len(tokens) if tokens else None,
        "latency_seconds": {f"p{p}": percentile(latencies, p / 100) for p in (50, 95, 99)},
        "note": "Success-only latency. Mock output uses whitespace counts, not model tokens.",
    }
    if stream:
        result["ttft_seconds"] = {f"p{p}": percentile(ttfts, p / 100) for p in (50, 95, 99)}
        result["tpot_seconds"] = {f"p{p}": percentile(tpots, p / 100) for p in (50, 95, 99)}
    else:
        result["note"] += " No TTFT measurement; use --stream."
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    if not getattr(args, "quiet", False):
        print(json.dumps(result, indent=2))
    return result["error_rate"] > args.max_error_rate


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--requests", type=int, default=500)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompt-words", type=int, default=0, help="Filler words to lengthen input")
    p.add_argument("--stream", action="store_true", help="Measure TTFT and TPOT via streaming")
    p.add_argument("--max-error-rate", type=float, default=0.01)
    p.add_argument("--output", default="results/load.json")
    return p


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parser().parse_args())))
