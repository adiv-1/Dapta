# vLLM inference backend

The Qwen worker (`backend/dapta/model_server.py`) serves Qwen3-VL-8B + LoRA through a local
vLLM server. The worker's API (`/generate`, `/admin/unload`, `/admin/load`, `/health`), its auth,
prompts and Sentry instrumentation are unchanged, so the controller needs no changes.

```text
controller :8000 ──/generate──▶ Qwen worker :8002 ──OpenAI API──▶ vLLM :8003 (loopback only)
                                 prompts, auth, Sentry             continuous batching, LoRA
```

- **Backend switch:** `DAPTA_QWEN_BACKEND=vllm` (default) or `transformers` (rollback; also used by
  the Hostinger Docker worker). `/health` reports the active backend.
- **Separate venv:** vLLM 0.30.0 pins torch 2.13 / transformers 5.x; training pins torch 2.7.1 /
  transformers 4.57.3. vLLM lives in `/workspace/dapta/.venv-vllm` (`deploy/requirements-vllm.txt`).
- **Adapters:** on first use, the worker downloads the adapter subfolder from `QWEN_ADAPTER_REPO`
  into `VLLM_ADAPTER_DIR` and registers it with `/v1/load_lora_adapter`. New factory adapters work
  without a restart. vLLM applies LoRA to the language model; our adapters (rank 16, attention and
  MLP projections) are language-model-only. `VLLM_MAX_LORA_RANK` must be at least the adapter rank.
- **Training on the same GPU:** the factory's existing unload/load calls put vLLM to sleep
  (`VLLM_SLEEP_LEVEL=1`: weights to CPU RAM, KV cache freed) and wake it after training.
- **Metrics:** token counts come from vLLM's reported usage only; TTFT is measured from the first
  streamed token (the transformers path reported none). The ledger and Sentry record
  `inference_backend`.

## First deployment on the RunPod pod

Run only when no factory job is active (see the guard in `TRAINING_RUNBOOK.md`).

```bash
cd /workspace/dapta
nvidia-smi                                   # driver / CUDA version
free -g && df -h /workspace                  # ~17 GB free RAM for sleep level 1; ~10 GB disk for the venv
./deploy/runpod-install-vllm.sh              # creates .venv-vllm; fails clearly on a CUDA mismatch
./deploy/runpod-start.sh                     # starts vLLM, then the worker, then the controller
curl -s http://127.0.0.1:8002/health         # expect "backend": "vllm", "model_loaded": true
tail -n 50 artifacts/logs/vllm.log
```

If the install reports a CUDA mismatch, the pod driver is older than the wheel's CUDA build:
install the CUDA 12.x vLLM build from the vLLM release notes into `.venv-vllm` instead.

## Verification still required on the GPU

These need the A40 and are not covered by the local tests (which use a fake vLLM server):

1. **Startup:** vLLM loads Qwen3-VL-8B with `--enable-lora --enable-sleep-mode` within the
   15-minute start window, with `VLLM_MAX_MODEL_LEN` and `VLLM_GPU_MEMORY_UTILIZATION` as set.
2. **Adapter:** `nda-intake/qwen3-vl-8b-v1` registers and a comparison completes.
3. **Training handoff:** after `/admin/unload`, `nvidia-smi` shows enough free memory for QLoRA
   training (CUDA graph memory stays allocated during sleep); `/admin/load` wakes vLLM.
4. **Quality parity:** the frozen NDA benchmark scores (field F1, schema pass rate) match the
   transformers backend closely. vLLM output is not byte-identical (different kernels; vLLM's Qwen-VL
   preprocessing differs from the official examples).

## Rollback

Set `DAPTA_QWEN_BACKEND=transformers` in `runtime.env` and rerun `./deploy/runpod-start.sh`.
