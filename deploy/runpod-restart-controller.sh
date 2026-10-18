#!/usr/bin/env bash
# Apply controller-only changes while leaving the loaded Qwen worker in place.
set -euo pipefail

ROOT_DIR="${DAPTA_ROOT_DIR:-/workspace/dapta}"
PYTHON_BIN="${DAPTA_PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
cd "$ROOT_DIR"
set -a
source "${DAPTA_RUNTIME_ENV:-$ROOT_DIR/runtime.env}"
set +a
export PYTHONPATH="$ROOT_DIR/backend"

curl --fail --silent http://127.0.0.1:8002/health >/dev/null
# Refuse a restart while the controller owns an active job or comparison.
"$PYTHON_BIN" - <<'PY'
import json
import os
import urllib.request

for path, terminal in (("jobs", {"ready", "failed"}), ("runs", {"completed", "failed"})):
    request = urllib.request.Request(
        f"http://127.0.0.1:8000/v1/{path}",
        headers={"X-Dapta-Token": os.environ["DAPTA_CONTROL_TOKEN"]},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        items = json.load(response)
    if any(item["status"] not in terminal for item in items):
        raise SystemExit(f"Controller restart refused: active {path}")
PY

PID_FILE="$ROOT_DIR/artifacts/pids/controller.pid"
controller_pid="$(<"$PID_FILE")"
controller_args="$(ps -p "$controller_pid" -o args=)"
if [[ "$controller_args" != *"uvicorn dapta.api:app"* ]]; then
  echo "Controller PID does not match Dapta; refusing to stop it." >&2
  exit 1
fi
kill "$controller_pid"
for _ in {1..30}; do
  kill -0 "$controller_pid" 2>/dev/null || break
  sleep 1
done
if kill -0 "$controller_pid" 2>/dev/null; then
  echo "Controller has not stopped; refusing to start a second process." >&2
  exit 1
fi

nohup "$PYTHON_BIN" -m uvicorn dapta.api:app --host 0.0.0.0 --port 8000 \
  >"$ROOT_DIR/artifacts/logs/controller.log" 2>&1 &
echo $! >"$PID_FILE"
for _ in {1..30}; do
  if curl --fail --silent http://127.0.0.1:8000/health; then
    echo
    exit 0
  fi
  sleep 1
done
echo "Controller did not become healthy; inspect its log." >&2
exit 1
