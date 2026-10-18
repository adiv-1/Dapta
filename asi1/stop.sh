#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DAPTA_ROOT_DIR:-/workspace/dapta}"
PID_FILE="$ROOT_DIR/artifacts/pids/asi1-agent.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "ASI:One companion is not running"
  exit 0
fi

pid="$(<"$PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  for _ in {1..20}; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
fi
rm -f "$PID_FILE"
echo "Stopped ASI:One companion"
