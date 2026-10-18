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
    # These patterns are scoped to Dapta's two Uvicorn module entry points.
    kill $pids
    for _ in {1..30}; do
      pgrep -f "$pattern" >/dev/null || break
      sleep 1
    done
  fi
}

stop_pid "$PID_DIR/controller.pid"
stop_pid "$PID_DIR/model-worker.pid"
stop_matching "[u]vicorn dapta.api:app"
stop_matching "[u]vicorn dapta.model_server:app"

nohup "$PYTHON_BIN" -m uvicorn dapta.model_server:app \
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
