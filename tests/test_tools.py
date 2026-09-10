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
