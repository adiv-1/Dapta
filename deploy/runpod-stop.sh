#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DAPTA_ROOT_DIR:-/workspace/dapta}"
PID_DIR="$ROOT_DIR/artifacts/pids"

for name in controller model-worker vllm; do
  file="$PID_DIR/$name.pid"
  [[ -f "$file" ]] || continue
  pid="$(<"$file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
  fi
done
