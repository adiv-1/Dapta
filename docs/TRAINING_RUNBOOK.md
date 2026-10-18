# Dapta training and recovery runbook

Use this runbook for every paid adapter build. It is designed to prevent the
failure loops encountered during the first A16, Colab, and RunPod runs.

## 1. Preflight

Before creating a job:

- Confirm the workflow is repeated, stable, narrow, and deterministically
  evaluable.
- Confirm `GET /health` reports factory, Qwen, publishing, and Supabase ready.
- Confirm the A40 has about 46 GiB visible VRAM with `nvidia-smi`.
- Confirm persistent storage has at least 35 GiB free for Qwen cache, artifacts,
  and temporary trainer output.
- Confirm Hugging Face write access and the target registry repo.
- Confirm `DAPTA_FACTORY_MAX_JOB_COST_USD` and
  `DAPTA_FACTORY_MAX_TRAIN_HOURS` fit the remaining RunPod balance.
- Confirm no other training or evaluation process owns the GPU.
- Back up the SQLite ledger before risky maintenance. Never copy secrets with it
  into Git.

Do not use Colab as the live controller. Colab is fallback experimentation only:
its kernel identity can expire, sessions disconnect, and its filesystem is not
the product's durable state.

## 2. Safe A40 settings

The working Qwen3-VL-8B configuration is:

```text
4-bit NF4 + double quantization
BF16 compute, TF32 enabled
LoRA r=16, alpha=32, dropout=0.05
targets: q/k/v/o + gate/up/down projections
micro-batch=1, gradient accumulation=8
gradient checkpointing=true, use_reentrant=false
paged_adamw_8bit
max sequence length=4096
SDPA attention
```

Do not restore the failed `batch_size=2`, `max_length=8192` configuration on a
48 GB card. It reached `No executable batch size found, reached zero`. If a
larger context is essential, reduce dataset/token length deliberately or move
to an 80 GB GPU after a smoke test; do not rely on `auto_find_batch_size` to
rescue an oversized multimodal batch.

`prepare_model_for_kbit_training` promotes frozen non-quantized modules to
float32. `restore_frozen_compute_dtype()` converts only frozen Qwen compute
layers back to BF16 while leaving norms stable. Removing this causes:

```text
RuntimeError: expected scalar type Float but found BFloat16
```

Generation after training must remain inside BF16 autocast because PEFT LoRA
projections may remain float32. Do not remove the autocast in
`evaluate_condition()`.

## 3. Dataset and evaluation gates

Training cannot begin until the validator passes schema, label shape,
duplicates, leakage, split isolation, and frozen TEST hashing. TRAIN/DEV/TEST/OOD
must remain separate. Only DEV failures may guide data repair.

Every benchmark uses the same frozen cases and decoding settings for:

1. Qwen base with a short task prompt;
2. Qwen base with the full Skill;
3. the saved LoRA with a short prompt.

`DAPTA_FACTORY_MAX_NEW_TOKENS=4096` is the current structured-output ceiling.
The worker records `generation_max_new_tokens` and `truncated_cases` for every
condition. Any truncated case invalidates scoring. It must not appear as 0%.

## 4. Monitoring a live job

Use the Base44 Skills & Adapters view first. It shows the universal timer/cost,
per-stage time/cost/result, GPU progress, ETA, loss, and agent artifacts.

On RunPod:

```bash
tail -f /workspace/dapta/artifacts/logs/controller.log
tail -f /workspace/dapta/artifacts/logs/model-worker.log
nvidia-smi
```

The factory artifacts for job `JOB_ID` live under:

```text
/workspace/dapta/artifacts/factory/JOB_ID/round-N/
```

Do not launch a second worker manually while the factory runner is active.

## 5. Failure classification and exact response

### CUDA out of memory / no executable batch size

Stop the stage. Verify no stale process owns VRAM. Keep micro-batch 1,
accumulation 8, checkpointing enabled, and sequence length 4096. Retry only the
failed training stage after correcting configuration. Do not regenerate data.

### Float/BFloat16 mismatch

Confirm the deployed commit includes `restore_frozen_compute_dtype`,
non-reentrant gradient checkpointing, and generation autocast. Restart the
controller/model services through `runpod-start.sh`, then retry the failed
training stage. Do not add blind casts throughout the model.

### OpenRouter content is not text

Inspect the sanitized provider envelope and finish reason. The client supports
structured/non-text assistant content; do not treat a large reasoning budget as
permission to accept an empty final answer. A failed paid call is recorded in
the ledger. Retry the exact claimed agent stage only after the response parser
or provider issue is corrected.

### Controller restart during a paid stage

Start with `./deploy/runpod-start.sh`. `recover_interrupted_stages()` converts
orphaned running claims into resumable state. Never create a second job to
work around the claim, and never erase `job_stage_attempts`.

### Evaluation output truncated

Do not retrain and do not invoke Kimi data repair. The UI repair action calls:

```text
POST /v1/jobs/{job_id}/repair-evaluation
```

This moves only to `evaluating`, unloads live inference, loads the saved adapter,
regenerates the frozen base/base+Skill/LoRA outputs with the larger cap, records
the incremental compute cost, and reloads inference. If truncation remains at
4096, inspect the output contract and consider a deliberately higher cap; do
not loop automatically.

### Model-quality gate fails with complete outputs

Show the measured failures and proposed DEV-only repairs to the user. Continue
only after explicit selection. Kimi may build targeted DEV/TRAIN candidates;
the frozen TEST inputs and labels remain untouched. A new training round must
have its own job-round allocation and costs.

### Hugging Face publication fails

Do not retrain. Confirm the adapter directory contains
`adapter_model.safetensors`, adapter config, tokenizer/processor files, and the
skill bundle. Retry only publication. Existing adapter IDs may be updated only
when skill ID, version, and base model match.

### Inference model fails to reload

The training result may still be valid. Check model-worker logs, GPU memory, and
the `/health` endpoint. Restart both services with `runpod-start.sh`; it is safer
than manually spawning another Uvicorn process and risking port conflicts.

## 6. Publication contract

Hugging Face receives:

```text
adapter_model.safetensors
adapter_config.json
processor/tokenizer files
manifest.json
metrics.json
skill/SKILL.md
skill/skill_spec.json
skill/output_schema.json
skill/evaluation_plan.json
skill/integration_evidence.json
```

Private input documents and secrets never belong in the bundle. An adapter with
an incomplete benchmark may be published for testing only when metadata says
`benchmark_status=needs_repair` and `evaluation_verified=false`.

## 7. Acceptance checklist

- Training artifact exists and weights are non-empty.
- Inference model is healthy after the GPU workload.
- No benchmark outputs are truncated.
- Deterministic schema and accuracy metrics are saved.
- Base, base+Skill, and LoRA use the same frozen cases.
- Ledger contains model, compute, timing, and token totals.
- Hugging Face contains weights, Skill, manifest, and metrics.
- Registry points to the exact version.
- Base44 shows every stage result and the universal totals.
- Supabase contains only sanitized agent artifacts.
- Backend tests and frontend production build pass.
- No secret or private document appears in Git, logs, Supabase, or Hugging Face.
