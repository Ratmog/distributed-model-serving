"""Submit a JSONL batch with stable per-row idempotency keys and poll results."""

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import httpx


async def run(args):
    raw = Path(args.input).read_bytes()
    requests = [json.loads(line) for line in raw.splitlines() if line.strip()]
    digest = hashlib.sha256(raw).hexdigest()
    deadline = time.monotonic() + args.timeout
    async with httpx.AsyncClient(
        base_url=args.url,
        timeout=120,
        headers={"Authorization": "Bearer " + os.getenv("API_KEY", "")},
    ) as client:
        ids = []
        for i, payload in enumerate(requests):
            response = await client.post(
                "/v1/jobs", json=payload, headers={"Idempotency-Key": f"{digest}:{i}"}
            )
            response.raise_for_status()
            ids.append(response.json()["id"])
        pending, results = set(ids), {}
        while pending:
            if time.monotonic() > deadline:
                raise TimeoutError("Polling timed out; rerun same input to resume retained jobs")
            for jid in list(pending):
                response = await client.get("/v1/jobs/" + jid)
                response.raise_for_status()
                data = response.json()
                if data["status"] in {"succeeded", "failed"}:
                    results[jid] = data
                    pending.remove(jid)
            if pending:
                await asyncio.sleep(1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("".join(json.dumps(results[jid]) + "\n" for jid in ids))
    return any(value["status"] == "failed" for value in results.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", default="results/batch.jsonl")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
