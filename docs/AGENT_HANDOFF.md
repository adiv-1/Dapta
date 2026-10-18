# Dapta agent handoff

Last verified: October 18, 2026 (America/Los_Angeles)

This is the authoritative takeover document. Read it completely before making
changes or starting a paid factory run.

## Repository and deployed surfaces

- GitHub: `adiv-1/Dapta`; shared branch: `main`.
- Base44 app: <https://dapta-bf7a516c.base44.app> (private/invite-only).
- Base44 project ID: `6ad441d282a59638bf7a516c`.
- Hugging Face registry: `adityave/dapta-lora-adapters`.
- Production controller and Qwen worker: one persistent RunPod A40 (48 GB).
- Production database: `/workspace/dapta/dapta.sqlite3` on the RunPod volume.
- Factory artifacts: `/workspace/dapta/artifacts/factory`.
- Controller/model logs: `/workspace/dapta/artifacts/logs`.
- Runtime secrets: `/workspace/dapta/runtime.env`; never copy it into Git.

The RunPod address and pod ID are operational state and may change. Resolve
them from the RunPod console or the ignored local `.env`; do not hard-code them.
The ignored `.runpod_dapta_ed25519` private key must never be committed.

## What is live

The full path is implemented:

```text
Base44 UI -> Base44 authenticated secret proxy -> FastAPI controller
  -> JEV registry routing
  -> Kimi K3 Skill Architect (OpenRouter)
  -> Devin skill integration evidence (no GitHub writes or PRs)
  -> Kimi K3 Data Builder (OpenRouter)
  -> deterministic dataset validator and frozen TEST hash
  -> compute/budget gate
  -> in-place RunPod Qwen3-VL-8B QLoRA worker
  -> base / base+Skill / LoRA frozen benchmark
  -> Hugging Face weights + skill bundle + manifest
  -> Base44 registry and Supabase sanitized artifact mirror
```

The controller and worker share the one GPU. Training or benchmark repair first
unloads the inference model and always reloads it in a `finally` block. Live
inference must not run concurrently with training.

Job states are persisted in SQLite:

```text
queued -> architecting -> integrating -> building_dataset -> validating
       -> awaiting_compute -> provisioning_gpu -> training -> evaluating
       -> publishing -> ready | failed
```

Stage claims and idempotency keys prevent paid work from being silently
duplicated after a controller restart. `FactoryRunner` recovers interrupted
claims on startup. Do not delete the SQLite file to "unstick" a job.

## Current adapters and evidence

### NDA intake

- Registry path: `nda-intake/qwen3-vl-8b-v1`.
- Frozen TEST: 83 Kleister NDA documents.
- Field F1: base 57.58%, base+Skill 61.28%, LoRA 70.75%.
- Exact-form rate: base 6.02%, LoRA 33.73%.

### CSV cleaner and dashboard

- Factory job: `31d6e12e-3da0-4c85-8bf5-e080a809db98`.
- Registry path:
  `generated/csv-clean-dashboard-v1/v20261018093318`.
- Training completed; adapter weights are approximately 175 MB.
- Four frozen cases were generated under all three benchmark conditions.
- Active adapter-training cost recorded in its manifest: `$0.124198`.
- End-to-end job ledger at publication time: `$1.326729496` total, comprising
  `$1.123181496` model calls and `$0.203548` compute. Devin usage is recorded
  separately in ACUs because a truthful USD conversion was unavailable.
- The first benchmark was **not scored** because the old 768-token generation
  ceiling truncated every JSON output. The weights are valid; the benchmark is
  marked `needs_repair`, never as a fabricated 0% result.
- The UI's repair action now performs an evaluation-only pass with the saved
  adapter and frozen data. It does not regenerate data or retrain weights.

## Production configuration (non-secret)

Verified on the live RunPod:

```dotenv
DAPTA_ENV=production
DAPTA_LEDGER_PATH=/workspace/dapta/dapta.sqlite3
DAPTA_FACTORY_AUTOSTART=true
DAPTA_FACTORY_TRAINING_ENABLED=true
DAPTA_FACTORY_MAX_JOB_COST_USD=5
DAPTA_FACTORY_MAX_TRAIN_HOURS=2
DAPTA_FACTORY_EPOCHS=2
DAPTA_FACTORY_MAX_LENGTH=4096
DAPTA_FACTORY_MAX_TRAIN_EXAMPLES=64
DAPTA_FACTORY_MAX_TEST_EXAMPLES=12
DAPTA_FACTORY_MAX_NEW_TOKENS=4096
QWEN_BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct
QWEN_ATTN_IMPLEMENTATION=sdpa
QWEN_MAX_NEW_TOKENS=1200
RUNPOD_GPU_TYPE=RTX-A40
RUNPOD_GPU_PLAN=RTX-A40-48GB
RUNPOD_GPU_HOURLY_COST_USD=0.59
HF_ADAPTER_REPO=adityave/dapta-lora-adapters
```

`DAPTA_FACTORY_MAX_NEW_TOKENS` uses a safe code default of 4096 even when not
present in an older `runtime.env`. Add it explicitly when recreating a runtime.

Required secret families are documented in `.env.example`. Browser code may
receive only public configuration and `VITE_SENTRY_DSN`. Provider credentials,
the controller token, Hugging Face token, Supabase secret, Devin credentials,
RunPod credentials, and Base44 management access remain server-side.

## Code map

- `backend/dapta/api_v2.py`: API assembly, auth, runner wiring, repair endpoint.
- `backend/dapta/jobs.py`: state machine, claims, artifacts, registry.
- `backend/dapta/factory_runtime.py`: Devin, dataset, training lifecycle,
  benchmark-only regeneration, Hugging Face publication.
- `backend/dapta/factory_gpu.py`: QLoRA and evaluation-only GPU entry points.
- `backend/dapta/openrouter.py`: Kimi/critic JSON calls and cost capture.
- `backend/dapta/ledger.py`: exact token/model/compute accounting.
- `backend/dapta/model_server.py`: base Qwen and dynamic PEFT loading.
- `backend/dapta/supabase_store.py`: sanitized external mirror.
- `frontend/src/App.tsx`: three-tab UI, factory timeline, repair and telemetry.
- `frontend/base44/functions/dapta-proxy/entry.ts`: authenticated secret proxy.
- `deploy/runpod-start.sh`: supported way to restart both services.

## Invariants a new agent must preserve

1. Never tune on frozen TEST or send TEST labels to a repair model.
2. Dataset failures may request targeted DEV data; evaluator configuration
   failures must never trigger more training data or retraining.
3. A truncated generation is unscored, not 0% accuracy.
4. Generated `SKILL.md`, schema, validator evidence, metrics, and weights go to
   Hugging Face beside the adapter. Devin never writes to the GitHub repo.
5. Private judge documents are not uploaded to Hugging Face or Supabase.
6. Costs come from the ledger. Do not label an unavailable cost `$0`.
7. One GPU workload at a time. Always restore the inference model.
8. Mutations use idempotency keys. Never bypass the stage-claim system for a
   normal factory job.
9. Preserve user-owned untracked notebooks and unrelated worktree changes.
10. Do not claim a benchmark passed until deterministic scoring says so.

## Verification before handoff or deployment

```bash
cd /path/to/Dapta
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=backend \
  .venv/bin/python -m unittest discover -s backend/tests -v
cd frontend && npm run build && npm test
```

Expected at this handoff: 99 backend tests, 16 frontend tests, and a successful
TypeScript/Vite production build.

For a live smoke check:

```bash
curl --fail https://CONTROLLER_HOST/health
```

The response must report `qwen_configured`, `factory_autostart`,
`training_configured`, `publishing_configured`, and `supabase_configured` as
true. Authenticate `/v1/*` checks with `X-Dapta-Token`; never paste that token
into logs or documentation.

## Deployment

Backend: pull `main` on the persistent RunPod, install changed dependencies if
requirements changed, and run `./deploy/runpod-start.sh`. Do not start a second
Uvicorn process on port 8000; the script stops the recorded services first.

Frontend:

```bash
cd frontend
npx base44 deploy --build --yes
```

Keep the app private. The Base44 function requires only
`DAPTA_CONTROLLER_URL` and `DAPTA_CONTROL_TOKEN` secrets. After deployment,
open Skills & Adapters and verify jobs, ledger totals, stage results, Hugging
Face links, and the recoverable benchmark repair panel render.

## Immediate next actions

1. Deploy the benchmark-only repair code before pressing the CSV repair button.
2. Press **Apply repair & rerun benchmark** once. Watch that the stage is
   `evaluating`, not `training`, and that the inference model reloads afterward.
3. If it passes, verify the existing Hugging Face version and registry metadata
   are updated to `benchmark_status=verified`.
4. Run one CSV through the adapter from the Test adapter action.
5. Register the stable `/v1/agent/tools` surface with ASI:One/Agentverse only
   after the above acceptance check.

Do not start another paid training run merely to repair the current CSV score.
