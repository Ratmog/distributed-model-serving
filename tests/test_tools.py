import argparse
import importlib.util
import json
from pathlib import Path

import httpx
import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_evaluation_gates_and_dataset_guard(tmp_path, monkeypatch):
    module = load("evaluate")
    original = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={
                "model": "test",
                "choices": [{"message": {"content": "2"}}],
                "usage": {"completion_tokens": 1},
            },
        )
    )
    monkeypatch.setattr(
        module.httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs)
    )
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({"id": "1", "prompt": "1+1?", "expected": "2"}) + "\n")
    args = argparse.Namespace(
        dataset=str(dataset),
        url="http://test",
        output=str(tmp_path / "result.json"),
        baseline=None,
        min_accuracy=0.9,
        max_accuracy_drop=0.02,
        max_latency_ratio=100,
        mlflow=False,
    )
    assert not await module.run(args)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(Path(args.output).read_text())
    args.baseline = str(baseline)
    dataset.write_text(json.dumps({"id": "1", "prompt": "1+1?", "expected": "3"}) + "\n")
    with pytest.raises(ValueError, match="hash"):
        await module.run(args)
    args.baseline = None
    assert await module.run(args)


async def test_load_reports_overload(tmp_path, monkeypatch):
    module = load("load_test")
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda _: httpx.Response(429))
    monkeypatch.setattr(
        module.httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs)
    )
    args = argparse.Namespace(
        requests=10,
        concurrency=3,
        url="http://test",
        max_tokens=10,
        output=str(tmp_path / "load.json"),
        max_error_rate=0.01,
    )
    assert await module.run(args)
    report = json.loads(Path(args.output).read_text())
    assert report["successful"] == 0
    assert report["errors"] == {"429": 10}
    assert report["latency_seconds"]["p99"] is None


def test_gpu_sampler_summarizes_fake_source():
    import time

    gpu_monitor = load("gpu_monitor")
    calls = iter(range(1000))

    def source():
        n = next(calls)
        return [
            {
                "index": i,
                "name": "FakeGPU",
                "util_pct": 50.0 + n % 2 * 50,
                "memory_used_mib": 1000.0 + n,
                "memory_total_mib": 24000.0,
            }
            for i in (0, 1)
        ]

    with gpu_monitor.GpuSampler(interval=0.01, source=source) as sampler:
        time.sleep(0.1)
    summary = sampler.summary()
    assert summary["available"] and len(summary["gpus"]) == 2
    assert 50 <= summary["gpus"][0]["mean_util_pct"] <= 100
    assert summary["gpus"][0]["peak_memory_mib"] >= 1001
    assert gpu_monitor.GpuSampler(source=None).summary() == {"available": False}


async def test_load_streaming_reports_ttft(tmp_path, monkeypatch):
    module = load("load_test")
    original = httpx.AsyncClient
    body = (
        'data: {"model":"m","choices":[{"delta":{"content":"a"}}]}\n\n'
        'data: {"model":"m","choices":[{"delta":{"content":" b"}}]}\n\n'
        'data: {"model":"m","choices":[],"usage":{"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=body))
    monkeypatch.setattr(
        module.httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs)
    )
    args = module.parser().parse_args(
        [
            "--requests",
            "6",
            "--concurrency",
            "2",
            "--stream",
            "--prompt-words",
            "30",
            "--output",
            str(tmp_path / "s.json"),
        ]
    )
    assert not await module.run(args)
    report = json.loads((tmp_path / "s.json").read_text())
    assert report["successful"] == 6 and report["models"] == ["m"]
    assert report["ttft_seconds"]["p50"] is not None
    assert report["tpot_seconds"]["p50"] is not None
    assert len(module.prompt(0, 30).split()) == 30 + 5


def fake_run(label, tok_s, env):
    return {
        "manifest": {
            "label": label,
            "git_commit": "abc",
            "git_dirty": False,
            "backend": "vllm",
            "models": ["Qwen"],
            "env": env,
            "host_gpus": {"available": True, "gpus": ["L4"]},
            "rounds": 5,
            "min_requests": 32,
            "repeats": 2,
            "max_tokens": 64,
            "prompt_words": 0,
            "stream": True,
        },
        "cells": [
            {
                "output_tokens_per_second": tok_s + d,
                "requests_per_second": 1.0,
                "error_rate": 0.0,
                "latency_seconds": {"p50": 1, "p95": 2, "p99": 3},
                "ttft_seconds": {"p50": 0.1, "p95": 0.2},
                "tpot_seconds": {"p50": 0.02},
                "gpu": {
                    "available": True,
                    "gpus": [
                        {"mean_util_pct": 40.0, "peak_memory_mib": 20000},
                        {"mean_util_pct": 60.0, "peak_memory_mib": 21000},
                    ],
                },
            }
            for d in (-10, 10)
        ],
    }


def test_benchmark_summary_cost_and_compare(tmp_path):
    bench, compare = load("benchmark"), load("compare")
    dirs = []
    for label, tok_s, seqs in (("base", 400, "16"), ("cand", 700, "64")):
        run = fake_run(label, tok_s, {"MAX_NUM_SEQS": seqs})
        row = bench.summarize(8, run["cells"], gpu_hourly_cost=1.0)
        assert row["output_tokens_per_second"] == tok_s
        assert row["gpu_mean_util_pct"] == 50.0 and row["gpu_peak_memory_mib"] == 21000
        assert row["usd_per_million_output_tokens"] == round(2.0 / (tok_s * 3600) * 1e6, 4)
        assert "| 8 |" in bench.markdown(run["manifest"], [row])
        d = tmp_path / label
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps(run["manifest"]))
        (d / "summary.json").write_text(json.dumps([row]))
        dirs.append(d)
    table = compare.compare(*dirs)
    assert "400 → 700 (+75%)" in table and '"MAX_NUM_SEQS": ["16", "64"]' in table
    manifest = json.loads((dirs[1] / "manifest.json").read_text())
    manifest["max_tokens"] = 128
    (dirs[1] / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shape"):
        compare.compare(*dirs)


def test_auth_header_omitted_without_key(monkeypatch):
    auth = load("_auth")
    monkeypatch.delenv("API_KEY", raising=False)
    assert auth.auth_headers() == {}
    httpx.Request("GET", "http://x", headers=auth.auth_headers())  # must be a legal request
    monkeypatch.setenv("API_KEY", "k")
    assert auth.auth_headers() == {"Authorization": "Bearer k"}
