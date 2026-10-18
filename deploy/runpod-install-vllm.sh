#!/usr/bin/env bash
# Create the separate vLLM serving environment on the RunPod pod. Safe to rerun.
# It does not touch the training/controller virtualenv or restart any service.
set -euo pipefail

ROOT_DIR="${DAPTA_ROOT_DIR:-/workspace/dapta}"
VLLM_VENV="${VLLM_VENV:-$ROOT_DIR/.venv-vllm}"
BASE_PYTHON="${VLLM_BASE_PYTHON:-python3}"

cd "$ROOT_DIR"
echo "GPU and driver:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true

[[ -x "$VLLM_VENV/bin/python" ]] || "$BASE_PYTHON" -m venv "$VLLM_VENV"
"$VLLM_VENV/bin/pip" install --upgrade pip
"$VLLM_VENV/bin/pip" install -r deploy/requirements-vllm.txt

# vLLM's default wheels target a recent CUDA; if CUDA is not available here, the pod's driver is
# older than the wheel needs: install the CUDA 12.x build listed in the vLLM release notes instead.
"$VLLM_VENV/bin/python" - <<'PY'
import torch, vllm
print(f"vllm {vllm.__version__}  torch {torch.__version__}  cuda build {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not usable from the vLLM environment: driver/wheel CUDA mismatch")
print("CUDA OK:", torch.cuda.get_device_name(0))
PY
