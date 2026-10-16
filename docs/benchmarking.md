# GPU benchmark runbook

Goal: produce real, reproducible numbers on two GPUs and commit them under `benchmarks/`.
The headline experiment is a batching-limit change (vLLM `--max-num-seqs`) measured
before and after under identical load. Budget roughly 2–3 GPU-hours.

There are two ways to run the stack: on a GPU **pod** without Docker (section A, simplest
and usually cheapest), or on a full **VM** with Docker Compose (section B). The benchmark
steps (3–5) are the same for both.

## A. RunPod (or any GPU container) without Docker

Pods are containers, so Compose cannot run inside them. `scripts/run_without_docker.sh`
starts the same topology as `compose.gpu.yaml` as plain processes: Redis, two vLLM
replicas pinned to GPU 0 and 1 with identical flags, the gateway, and a worker.

1. **Deploy a pod** with **2 GPUs of ≥24 GB each** (RTX 4090, RTX A5000, L4, A40, or larger).
   Use the official RunPod PyTorch template. Set container disk to ~50 GB and a
   **volume of ~100 GB** (mounted at `/workspace`; model weights are cached there).
   Note the hourly price shown for the pod and divide by 2 for `--gpu-hourly-cost`.
2. **Open a terminal** on the pod (web terminal or SSH) and work inside `/workspace`:

   ```bash
   cd /workspace
   git clone https://github.com/Ratmog/distributed-model-serving.git
   cd distributed-model-serving
   nvidia-smi                                  # both GPUs listed
   scripts/run_without_docker.sh setup         # Redis, Python 3.12 env, vLLM (~5–10 min)
   cp .env.example .env
   ```

3. **Edit `.env`:** set `MAX_INFLIGHT=128` and `MAX_NUM_SEQS=8` (the baseline). `API_KEY`
   can stay empty because nothing is exposed outside the pod.
4. **Start everything** (the first start downloads ~15 GB of weights):

   ```bash
   scripts/run_without_docker.sh up
   source .venv/bin/activate
   ```

5. Run steps 3–5 below. Wherever they say to recreate the vLLM containers, instead run
   `scripts/run_without_docker.sh restart-vllm` after editing `MAX_NUM_SEQS` in `.env`.
6. `scripts/run_without_docker.sh status`, `logs vllm-0`, and `down` are there when needed.
   Push your committed results to GitHub, then **terminate the pod**. A stopped pod still
   bills for its volume.

## B. A VM with Docker Compose

- **A VM, not a container "pod".** Compose needs its own Docker daemon. Pick an instance
  where you get a full Linux VM with Docker and the NVIDIA Container Toolkit (or install them).
- **2 GPUs with ≥24 GB each** (L4, A10G, RTX A5000/4090, or larger). A 7B model in BF16
  is about 15 GB of weights per replica; the rest goes to KV cache.
- **~100 GB disk** for the vLLM image and model weights.
- Write down the instance type and the per-GPU hourly price. You will pass the price in so
  the summary can report $/1M output tokens.

Sanity check:

```bash
nvidia-smi                       # both GPUs listed
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Start the stack with Compose

```bash
git clone https://github.com/Ratmog/distributed-model-serving.git
cd distributed-model-serving
cp .env.example .env
```

Edit `.env`:

```bash
BACKEND=vllm
API_KEY=some-long-secret
MAX_INFLIGHT=128        # above the highest sweep concurrency, so the gateway doesn't 429
MAX_NUM_SEQS=8          # baseline: deliberately small batch limit
```

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
docker compose -f compose.yaml -f compose.gpu.yaml logs -f vllm-0 vllm-1   # wait for "Application startup complete"
curl localhost:8000/health/ready
```

If vLLM fails to start, check the image/driver match (`VLLM_IMAGE` in `.env`) and VRAM.

Client tools run **on the GPU host** so `nvidia-smi` sampling works:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,eval]'
export API_KEY=some-long-secret
```

Sanity-check one streamed request (either path; drop the header if `API_KEY` is empty):

```bash
curl -N localhost:8000/v1/chat/completions -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi"}],"stream":true}'
```

## 3. Baseline sweep

```bash
python scripts/benchmark.py --label qwen7b-<gpu>x2-seqs8 --gpu-hourly-cost <USD per GPU-hour>
```

Defaults: concurrency 1, 8, 16, 32, 64, 100; `max(32, 5×concurrency)` requests per cell;
256 max output tokens; ~200-word prompts; streaming on (so TTFT/TPOT are measured);
3 repeats per cell with the median reported; 50 unrecorded warmup requests first.

Output in `benchmarks/<label>/`: `manifest.json` (commit, GPUs, driver, settings, load
shape), `raw/` (every cell), `summary.json`, and `summary.md` (the table).

Also run the quality eval on the baseline:

```bash
python scripts/evaluate.py --output benchmarks/qwen7b-<gpu>x2-seqs8/eval.json --mlflow
```

## 4. Change one variable and rerun

Set `MAX_NUM_SEQS=64` in `.env`, then recreate only the vLLM replicas:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d --force-recreate vllm-0 vllm-1
# wait for both replicas to be healthy again
curl localhost:8000/health/ready
python scripts/benchmark.py --label qwen7b-<gpu>x2-seqs64 --gpu-hourly-cost <price>
python scripts/evaluate.py --baseline benchmarks/qwen7b-<gpu>x2-seqs8/eval.json \
  --output benchmarks/qwen7b-<gpu>x2-seqs64/eval.json
```

## 5. Compare and commit

```bash
python scripts/compare.py benchmarks/qwen7b-<gpu>x2-seqs8 benchmarks/qwen7b-<gpu>x2-seqs64 \
  > benchmarks/seqs8-vs-seqs64.md
git add benchmarks && git commit -m "Add measured 2x<gpu> benchmark: max-num-seqs 8 vs 64"
```

`compare.py` refuses runs with different load shapes or mock backends, so the table is
like-for-like by construction.

Then **tear down the VM**.

## Reading the results honestly

- Throughput gains from a larger batch limit usually cost per-request latency (TPOT rises).
  Report both; the comparison table shows them side by side.
- `GPU util %` is nvidia-smi's "kernel running" fraction, not SM occupancy. Treat it as a
  coarse signal.
- vLLM preallocates KV cache (`--gpu-memory-utilization 0.90`), so peak memory mostly
  reflects that setting, not live load. For cache pressure, use vLLM's own KV-cache usage
  metric in Prometheus (search `cache_usage` in the vLLM target; the name varies by version).
- `$/1M output tokens` assumes the GPUs are fully dedicated for the measured throughput.
- The load generator is closed-loop. It measures capacity under N concurrent users, not
  behavior under a fixed arrival rate.
- Put only numbers from `summary.md` / `compare` output on a resume, with the hardware named.

## Failure drills

### Crash recovery

Submit `data/batch.example.jsonl` (use longer generation requests for an easier kill
window). Stop the worker while a job is running, then start it again. Recovery checks every
two seconds; by default a 90-second lease must expire before a retry. A terminal result
should appear or the three-attempt budget should exhaust. Inference may execute twice, but
stale completion tokens must not overwrite the current result.

### Replica failure

On the GPU stack, stop `vllm-0`. Requests already routed to it may return 503. Subsequent
requests skip it for a five-second cooldown; the other replica continues handling requests.
The failed replica is retried after cooldown, so this is simple failure isolation rather
than a full circuit breaker with proactive probing. Restore the service and watch recovery.

### Queue saturation

Set `QUEUE_CAPACITY` through the service environment, pause the worker, and submit that
many unique jobs. Further submissions must return 429. A repeated idempotency key for an
existing matching request still returns its ID. The bound includes running work, not just
the waiting list. Completed results are TTL-limited rather than count-limited; size Redis
accordingly.

### Worker back-pressure

Set `WORKER_CONCURRENCY` above `MAX_INFLIGHT` and submit a batch. Jobs that hit the
worker's local admission limit are requeued immediately (`serving_jobs_total{outcome="requeued"}`)
without spending a retry attempt or waiting for lease expiry.

## Quality gates

The supplied 200-prompt dataset is deliberately simple and synthetic. Build a separate
representative, reviewed dataset before claiming model-selection quality. Compare
baseline/candidate under the same hardware and load. The script gates exact accuracy and
mean latency; it also records peak GPU memory when run on the GPU host. It does not measure
factuality.
