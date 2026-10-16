#!/usr/bin/env bash
# Run the GPU stack without Docker, e.g. on a RunPod/Vast.ai pod with 2 GPUs.
# Same topology and vLLM flags as compose.gpu.yaml: Redis, two vLLM replicas pinned to
# GPU 0 and GPU 1, the API gateway, and one worker. Prometheus/Jaeger are not started.
#
#   scripts/run_without_docker.sh setup          # once: redis, uv, Python env, vLLM
#   scripts/run_without_docker.sh up             # start everything, wait until ready
#   scripts/run_without_docker.sh restart-vllm   # after changing MAX_NUM_SEQS in .env
#   scripts/run_without_docker.sh status | logs [name] | down
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
RUN="$ROOT/.run"
VENV="$ROOT/.venv"
mkdir -p "$RUN/logs" "$RUN/pids" "$RUN/redis"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
VLLM_VERSION=${VLLM_VERSION:-0.10.2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
export BACKEND=vllm MODEL
export REDIS_URL=redis://127.0.0.1:6379/0
export VLLM_URLS=http://127.0.0.1:8001,http://127.0.0.1:8002
# Keep model weights on the persistent volume when there is one.
if [[ -z "${HF_HOME:-}" && -d /workspace ]]; then export HF_HOME=/workspace/hf; fi
VLLM_BIN=${VLLM_BIN:-$VENV/bin/vllm}
PY=$VENV/bin/python

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

start() { # name, command...
  local name=$1
  shift
  if is_running "$name"; then
    log "$name already running"
    return
  fi
  setsid "$@" >"$RUN/logs/$name.log" 2>&1 < /dev/null &
  echo $! >"$RUN/pids/$name"
  log "started $name (pid $!, log .run/logs/$name.log)"
}

is_running() {
  [[ -f "$RUN/pids/$1" ]] && kill -0 "$(cat "$RUN/pids/$1")" 2>/dev/null
}

stop() {
  local name=$1 pid
  [[ -f "$RUN/pids/$name" ]] || return 0
  pid=$(cat "$RUN/pids/$name")
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$pid" 2>/dev/null || true
    log "stopped $name"
  fi
  rm -f "$RUN/pids/$name"
}

wait_http() { # name, url, timeout seconds
  local name=$1 url=$2 limit=$3 waited=0
  until curl -sf -o /dev/null "$url"; do
    if ! is_running "$name"; then
      log "$name exited; last log lines:"
      tail -n 30 "$RUN/logs/$name.log" >&2
      exit 1
    fi
    if ((waited >= limit)); then
      log "$name not ready after ${limit}s; see .run/logs/$name.log"
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  log "$name ready"
}

start_vllm() {
  for i in 0 1; do
    start "vllm-$i" env CUDA_VISIBLE_DEVICES=$i "$VLLM_BIN" serve "$MODEL" \
      --host 127.0.0.1 --port $((8001 + i)) \
      --enable-prefix-caching --max-model-len 4096 \
      --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  done
  # First start downloads ~15 GB of weights, so allow a long wait.
  wait_http vllm-0 http://127.0.0.1:8001/health 1800
  wait_http vllm-1 http://127.0.0.1:8002/health 1800
}

case "${1:-}" in
  setup)
    command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found; is this a GPU machine?" >&2; exit 1; }
    gpus=$(nvidia-smi -L | wc -l)
    ((gpus >= 2)) || { echo "Need 2 GPUs, found $gpus" >&2; exit 1; }
    if ! command -v redis-server >/dev/null; then
      log "installing redis-server"
      apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq redis-server curl
    fi
    command -v uv >/dev/null || { python3 -m pip install -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh; }
    export PATH="$HOME/.local/bin:$PATH"
    log "creating Python 3.12 env and installing vLLM $VLLM_VERSION (several minutes)"
    uv venv --python 3.12 "$VENV"
    uv pip install --python "$PY" -e '.[dev,eval]' "vllm==$VLLM_VERSION"
    "$VLLM_BIN" --version
    log "setup done. Next: edit .env, then $0 up"
    ;;
  up)
    [[ -x "$PY" ]] || { echo "Run '$0 setup' first" >&2; exit 1; }
    start redis redis-server --bind 127.0.0.1 --port 6379 --dir "$RUN/redis" \
      --appendonly yes --maxmemory-policy noeviction --save ""
    start_vllm
    # Gateway and worker start after the replicas so their warmup reaches both.
    start api "$PY" -m uvicorn serving.api:app --host 127.0.0.1 --port 8000
    start worker env WORKER_METRICS_PORT=9101 "$PY" -m serving.worker
    wait_http api http://127.0.0.1:8000/health/ready 120
    log "stack up: BACKEND=vllm MODEL=$MODEL MAX_NUM_SEQS=$MAX_NUM_SEQS vllm=$VLLM_VERSION"
    ;;
  restart-vllm)
    stop vllm-0
    stop vllm-1
    start_vllm
    log "replicas restarted with MAX_NUM_SEQS=$MAX_NUM_SEQS"
    ;;
  down)
    for name in worker api vllm-1 vllm-0 redis; do stop "$name"; done
    ;;
  status)
    for name in redis vllm-0 vllm-1 api worker; do
      if is_running "$name"; then echo "$name: running"; else echo "$name: stopped"; fi
    done
    curl -s http://127.0.0.1:8000/health/ready && echo
    ;;
  logs)
    tail -n 50 -f "$RUN/logs/${2:-api}.log"
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 1
    ;;
esac
