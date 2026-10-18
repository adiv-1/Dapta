#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DAPTA_ROOT_DIR:-/workspace/dapta}"
ENV_FILE="${DAPTA_RUNTIME_ENV:-$ROOT_DIR/runtime.env}"
PYTHON_BIN="${DAPTA_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
LOG_DIR="$ROOT_DIR/artifacts/logs"
PID_DIR="$ROOT_DIR/artifacts/pids"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR" "$PID_DIR"
set -a
source "$ENV_FILE"
set +a
export PYTHONPATH="$ROOT_DIR/backend"
QWEN_BACKEND="${DAPTA_QWEN_BACKEND:-vllm}"
VLLM_BIN="${VLLM_BIN:-$ROOT_DIR/.venv-vllm/bin/vllm}"
VLLM_PORT="${VLLM_PORT:-8003}"

stop_pid() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(<"$file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      for _ in {1..30}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
    fi
  fi
}

stop_matching() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "$pattern" || true)"
  if [[ -n "$pids" ]]; then
    # These patterns are scoped to Dapta's Uvicorn entry points and its loopback vLLM server.
    kill $pids
    for _ in {1..30}; do
      pgrep -f "$pattern" >/dev/null || break
      sleep 1
    done
  fi
}

stop_pid "$PID_DIR/controller.pid"
stop_pid "$PID_DIR/model-worker.pid"
stop_pid "$PID_DIR/vllm.pid"
stop_matching "[u]vicorn dapta.api:app"
stop_matching "[u]vicorn dapta.model_server:app"
stop_matching "[v]llm serve .* --port $VLLM_PORT"

if [[ "$QWEN_BACKEND" == "vllm" ]]; then
  if [[ ! -x "$VLLM_BIN" ]]; then
    echo "vLLM backend selected but $VLLM_BIN is missing: run deploy/runpod-install-vllm.sh" >&2
    exit 1
  fi
  vllm_args=(
    serve "${QWEN_BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
    --host 127.0.0.1 --port "$VLLM_PORT"
    --enable-lora
    --max-lora-rank "${VLLM_MAX_LORA_RANK:-16}"
    --max-loras "${VLLM_MAX_LORAS:-4}"
    --max-cpu-loras "${VLLM_MAX_CPU_LORAS:-8}"
    --max-model-len "${VLLM_MAX_MODEL_LEN:-16384}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
    --enable-sleep-mode
  )
  [[ -n "${VLLM_API_KEY:-}" ]] && vllm_args+=(--api-key "$VLLM_API_KEY")
  # shellcheck disable=SC2206 -- operator-supplied extra flags
  [[ -n "${VLLM_EXTRA_ARGS:-}" ]] && vllm_args+=(${VLLM_EXTRA_ARGS})
  # Sleep/wake and runtime LoRA registration are dev endpoints: vLLM binds to loopback only.
  # The venv's bin comes first on PATH: FlashInfer JIT-builds kernels with the venv's `ninja`.
  PATH="$(dirname "$VLLM_BIN"):$PATH" VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
    nohup "$VLLM_BIN" "${vllm_args[@]}" >"$LOG_DIR/vllm.log" 2>&1 &
  echo $! >"$PID_DIR/vllm.pid"
  # First start downloads/compiles; allow up to 15 minutes.
  for _ in {1..180}; do
    if curl --fail --silent "http://127.0.0.1:$VLLM_PORT/health" >/dev/null; then
      break
    fi
    kill -0 "$(<"$PID_DIR/vllm.pid")" 2>/dev/null || { echo "vLLM exited; see $LOG_DIR/vllm.log" >&2; exit 1; }
    sleep 5
  done
  curl --fail --silent "http://127.0.0.1:$VLLM_PORT/health" >/dev/null
  export VLLM_URL="${VLLM_URL:-http://127.0.0.1:$VLLM_PORT}"
fi

DAPTA_QWEN_BACKEND="$QWEN_BACKEND" nohup "$PYTHON_BIN" -m uvicorn dapta.model_server:app \
  --host 127.0.0.1 --port 8002 \
  >"$LOG_DIR/model-worker.log" 2>&1 &
echo $! >"$PID_DIR/model-worker.pid"

for _ in {1..120}; do
  if curl --fail --silent http://127.0.0.1:8002/health >/dev/null; then
    break
  fi
  sleep 5
done
curl --fail --silent http://127.0.0.1:8002/health >/dev/null

nohup "$PYTHON_BIN" -m uvicorn dapta.api:app \
  --host 0.0.0.0 --port 8000 \
  >"$LOG_DIR/controller.log" 2>&1 &
echo $! >"$PID_DIR/controller.pid"

for _ in {1..30}; do
  if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent http://127.0.0.1:8000/health
