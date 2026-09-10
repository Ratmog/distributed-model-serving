"""Versioned exact-answer regression suite. Optional MLflow logging and baseline gates."""

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import httpx


async def run(args):
    raw = Path(args.dataset).read_bytes()
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Dataset must be nonempty with unique IDs")
    rows = []
    async with httpx.AsyncClient(
        base_url=args.url,
        timeout=120,
        headers={"Authorization": "Bearer " + os.getenv("API_KEY", "")},
    ) as client:
        for case in cases:
            start = time.perf_counter()
            try:
                r = await client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": case["prompt"]}],
                        "temperature": 0,
                        "max_tokens": 32,
                    },
                )
                r.raise_for_status()
                data = r.json()
                answer = data["choices"][0]["message"]["content"].strip()
                rows.append(
                    {
                        "id": case["id"],
                        "answer": answer,
                        "correct": answer.casefold() == str(case["expected"]).strip().casefold(),
                        "model": data["model"],
                        "seconds": time.perf_counter() - start,
                        "output_tokens": data["usage"]["completion_tokens"],
                    }
                )
            except (httpx.HTTPError, KeyError, ValueError):
                rows.append(
                    {
                        "id": case["id"],
                        "correct": False,
                        "error": True,
                        "seconds": time.perf_counter() - start,
                    }
                )
    result = {
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "cases": len(cases),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "mean_latency_seconds": sum(row["seconds"] for row in rows) / len(rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "rows": rows,
    }
    failed = result["errors"] > 0 or result["accuracy"] < args.min_accuracy
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        if baseline["dataset_sha256"] != result["dataset_sha256"]:
            raise ValueError("Baseline dataset hash does not match")
        failed |= result["accuracy"] < baseline["accuracy"] - args.max_accuracy_drop
        failed |= (
            result["mean_latency_seconds"]
            > baseline["mean_latency_seconds"] * args.max_latency_ratio
        )
    result["gate_passed"] = not failed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    if args.mlflow:
        import mlflow

        mlflow.set_experiment("model-serving-regression")
        with mlflow.start_run():
            mlflow.log_param("dataset_sha256", result["dataset_sha256"])
            mlflow.log_param("models", sorted({r.get("model", "error") for r in rows}))
            mlflow.log_metrics(
                {k: result[k] for k in ("accuracy", "mean_latency_seconds", "errors")}
            )
            mlflow.log_artifact(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="data/regression.jsonl")
    parser.add_argument("--output", default="results/eval.json")
    parser.add_argument("--baseline")
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--max-latency-ratio", type=float, default=1.25)
    parser.add_argument("--mlflow", action="store_true")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
