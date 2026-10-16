"""Concurrency sweep with a reproducibility manifest, GPU sampling, and a summary table.

Run on the GPU host (so nvidia-smi sees the replicas). Each (concurrency, repeat) cell is
saved as raw JSON; summary.json/summary.md report the median across repeats.

    python scripts/benchmark.py --label qwen7b-seqs32 --gpu-hourly-cost 0.80
"""

import argparse
import asyncio
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import load_test
from gpu_monitor import GpuSampler, describe

TRACKED_ENV = (
    "MODEL",
    "VLLM_IMAGE",
    "VLLM_VERSION",
    "MAX_NUM_SEQS",
    "MAX_INFLIGHT",
    "WORKER_CONCURRENCY",
)


def settings(env_file=".env"):
    """Tracked settings from .env (what Compose used), overridden by the shell environment."""
    values = {}
    path = Path(env_file)
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() in TRACKED_ENV and value.strip():
                values[key.strip()] = value.strip()
    values |= {k: os.environ[k] for k in TRACKED_ENV if os.environ.get(k)}
    return values


def git(*cmd):
    try:
        return subprocess.run(
            ["git", *cmd], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def median(cells, *path):
    values = []
    for cell in cells:
        value = cell
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            values.append(value)
    return statistics.median(values) if values else None


def summarize(concurrency, cells, gpu_hourly_cost):
    gpus = [c["gpu"] for c in cells if c["gpu"].get("available")]
    row = {
        "concurrency": concurrency,
        "repeats": len(cells),
        "output_tokens_per_second": median(cells, "output_tokens_per_second"),
        "requests_per_second": median(cells, "requests_per_second"),
        "error_rate": max(c["error_rate"] for c in cells),
        "latency_p50": median(cells, "latency_seconds", "p50"),
        "latency_p95": median(cells, "latency_seconds", "p95"),
        "latency_p99": median(cells, "latency_seconds", "p99"),
        "ttft_p50": median(cells, "ttft_seconds", "p50"),
        "ttft_p95": median(cells, "ttft_seconds", "p95"),
        "tpot_p50": median(cells, "tpot_seconds", "p50"),
        "gpu_mean_util_pct": None,
        "gpu_peak_memory_mib": None,
        "usd_per_million_output_tokens": None,
    }
    if gpus:
        row["gpu_mean_util_pct"] = round(
            statistics.mean(g["mean_util_pct"] for s in gpus for g in s["gpus"]), 1
        )
        row["gpu_peak_memory_mib"] = max(g["peak_memory_mib"] for s in gpus for g in s["gpus"])
        if gpu_hourly_cost and row["output_tokens_per_second"]:
            gpu_count = len(gpus[0]["gpus"])
            hourly = gpu_hourly_cost * gpu_count
            row["usd_per_million_output_tokens"] = round(
                hourly / (row["output_tokens_per_second"] * 3600) * 1e6, 4
            )
    return row


def fmt(value, scale=1, digits=2):
    return "–" if value is None else f"{value * scale:.{digits}f}"


def markdown(manifest, rows):
    lines = [
        f"# Benchmark: {manifest['label']}",
        "",
        f"- Commit: `{manifest['git_commit']}`{' (dirty)' if manifest['git_dirty'] else ''}",
        f"- Backend: {manifest['backend']} · Models: {', '.join(manifest['models']) or '–'}",
        f"- GPUs: {'; '.join(manifest['host_gpus'].get('gpus', [])) or 'not detected'}",
        f"- Settings: {json.dumps(manifest['env'])}",
        (
            f"- Load: max({manifest['min_requests']}, {manifest['rounds']}×concurrency) "
            f"requests/cell, {manifest['max_tokens']} max output "
            f"tokens, {manifest['prompt_words']} filler prompt words, "
            f"streaming={manifest['stream']}, median of {manifest['repeats']} repeats"
        ),
        "",
        (
            "| Conc. | Out tok/s | Req/s | Err % | p50 / p95 / p99 latency (s) "
            "| TTFT p50 / p95 (ms) | TPOT p50 (ms) | GPU util % | Peak mem (MiB) | $/1M out tok |"
        ),
        "|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['concurrency']} | {fmt(r['output_tokens_per_second'], digits=0)} "
            f"| {fmt(r['requests_per_second'])} | {fmt(r['error_rate'], 100, 1)} "
            f"| {fmt(r['latency_p50'])} / {fmt(r['latency_p95'])} / {fmt(r['latency_p99'])} "
            f"| {fmt(r['ttft_p50'], 1000, 0)} / {fmt(r['ttft_p95'], 1000, 0)} "
            f"| {fmt(r['tpot_p50'], 1000, 1)} | {fmt(r['gpu_mean_util_pct'], digits=1)} "
            f"| {fmt(r['gpu_peak_memory_mib'], digits=0)} "
            f"| {fmt(r['usd_per_million_output_tokens'], digits=3)} |"
        )
    if manifest["backend"] == "mock":
        lines += ["", "**MOCK BACKEND — these numbers say nothing about GPU performance.**"]
    return "\n".join(lines) + "\n"


async def run(args):
    out = Path(args.out_dir or f"benchmarks/{args.label}")
    async with httpx.AsyncClient(base_url=args.url, timeout=10) as client:
        backend = (await client.get("/health/live")).json().get("backend", "unknown")
    if backend == "mock" and not args.allow_mock:
        raise SystemExit("Gateway reports BACKEND=mock. Use the GPU stack, or pass --allow-mock.")
    concurrencies = [int(c) for c in args.concurrency.split(",")]
    out.mkdir(parents=True, exist_ok=True)

    def load_args(concurrency, requests, path):
        return argparse.Namespace(
            url=args.url,
            requests=requests,
            concurrency=concurrency,
            max_tokens=args.max_tokens,
            prompt_words=args.prompt_words,
            stream=args.stream,
            max_error_rate=1.0,
            output=str(path),
            quiet=True,
        )

    if args.warmup_requests:
        print(f"warmup: {args.warmup_requests} unrecorded requests", flush=True)
        await load_test.run(
            load_args(max(concurrencies), args.warmup_requests, out / "warmup.json")
        )
        (out / "warmup.json").unlink(missing_ok=True)

    rows, models = [], set()
    for concurrency in concurrencies:
        cells = []
        for repeat in range(1, args.repeats + 1):
            path = out / "raw" / f"c{concurrency:03d}-r{repeat}.json"
            requests = max(args.min_requests, concurrency * args.rounds)
            with GpuSampler(interval=args.gpu_interval) as sampler:
                await load_test.run(load_args(concurrency, requests, path))
            cell = json.loads(path.read_text())
            cell["gpu"] = sampler.summary()
            path.write_text(json.dumps(cell, indent=2))
            models.update(cell["models"])
            cells.append(cell)
            print(
                f"c={concurrency} r={repeat}: {cell['output_tokens_per_second']:.0f} tok/s, "
                f"err {cell['error_rate']:.1%}",
                flush=True,
            )
        rows.append(summarize(concurrency, cells, args.gpu_hourly_cost))

    manifest = {
        "label": args.label,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "backend": backend,
        "models": sorted(models),
        "env": settings(),
        "host_gpus": describe(),
        "client_platform": platform.platform(),
        "rounds": args.rounds,
        "min_requests": args.min_requests,
        "repeats": args.repeats,
        "max_tokens": args.max_tokens,
        "prompt_words": args.prompt_words,
        "stream": args.stream,
        "gpu_hourly_cost_per_gpu": args.gpu_hourly_cost,
        "notes": args.notes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    (out / "summary.md").write_text(markdown(manifest, rows))
    print((out / "summary.md").read_text())
    return any(r["error_rate"] > args.max_error_rate for r in rows)


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--label", required=True, help="e.g. qwen7b-l4x2-seqs32")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--concurrency", default="1,8,16,32,64,100")
    p.add_argument("--rounds", type=int, default=5, help="Requests per cell = rounds×concurrency")
    p.add_argument("--min-requests", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--prompt-words", type=int, default=200)
    p.add_argument("--no-stream", dest="stream", action="store_false")
    p.add_argument("--warmup-requests", type=int, default=50)
    p.add_argument("--gpu-interval", type=float, default=0.5)
    p.add_argument("--gpu-hourly-cost", type=float, help="USD per GPU-hour, for $/1M tokens")
    p.add_argument("--max-error-rate", type=float, default=0.01)
    p.add_argument("--out-dir")
    p.add_argument("--notes", default="")
    p.add_argument("--allow-mock", action="store_true")
    return p


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parser().parse_args())))
