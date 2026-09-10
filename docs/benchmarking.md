# Experiments and failure drills

## Establish a real baseline

1. Record the git commit, GPU model/count/VRAM, driver, CUDA compatibility, model revision, image tag, prompts, output limit, and settings in a result-side manifest.
2. Wait for both replicas to finish loading. Warm each with a representative prompt, then run an unrecorded load-test pass. Startup warmup is only one token.
3. Run concurrency 1, 8, 16, 32, 64, and 100 with fixed request count and output limit. Save a separate JSON per setting. A high concurrency value can intentionally trigger 429s with the default admission limit.
4. Change one variable (e.g. `MAX_NUM_SEQS` from 16 to 32), recreate GPU services, repeat warmup, and repeat at least three runs. Compare error rate alongside throughput and tail latency.
5. Inspect Prometheus `rate(serving_output_tokens_total{backend="vllm"}[1m])`, `serving_inflight`, `serving_queue_active`, and vLLM's own scheduler/cache metrics. Prometheus histogram quantiles are estimates; client JSON uses nearest-rank samples.
6. Keep raw outputs and hardware metadata. Report measured results only; a mock test says nothing about GPU utilization.

## Crash recovery

Submit `data/batch.example.jsonl` (use longer generation requests for an easier kill window). Stop the worker while a job is running, then start it again. Recovery checks every two seconds; by default a 90-second lease must expire before a retry. A terminal result should appear or the three-attempt budget should exhaust. Inference may execute twice, but stale completion tokens must not overwrite the current result.

## Replica failure

On the GPU stack, stop `vllm-0`. Requests already routed to it may return 503. Subsequent requests skip it for a five-second cooldown; the other replica continues handling requests. The failed replica is retried after cooldown, so this is simple failure isolation rather than a full circuit breaker with proactive probing. Restore the service and watch recovery.

## Queue saturation

Set `QUEUE_CAPACITY` through the service environment, pause the worker, and submit that many unique jobs. Further submissions must return 429. A repeated idempotency key for an existing matching request still returns its ID. The bound includes running work, not just the waiting list. Completed results are TTL-limited rather than count-limited; size Redis accordingly.

## Quality gates

The supplied 200-prompt dataset is deliberately simple and synthetic. Build a separate representative, reviewed dataset before claiming model-selection quality. Compare baseline/candidate under the same hardware and load. The script gates exact accuracy and mean latency, not factuality, cost, or memory. Collect those additional dimensions separately if needed.
