# Dapta

Dapta turns stable, repetitive AI workflows into evaluated LoRA adapters for
`Qwen/Qwen3-VL-8B-Instruct`. The first end-to-end workflow converts receipt
images into a strict expense-report JSON object, then compares the specialized
Qwen adapter with frontier baselines on accuracy, time, and cost.

This README is the teammate handoff and current execution checklist. The deeper
product rationale lives in `docs/HACKATHON_MASTER_PLAN.md`; implementation
details live in `docs/BUILD_PLAN.md`.

## Current status

As of October 17, 2026:

- Cloud accounts and credentials needed for the first adapter are configured.
- The cost ledger, job state machine, and metered OpenRouter agent boundary are
  implemented and tested.
- Kimi K3 generated the first receipt skill contract through the real pipeline.
- The pinned CORD v2 and SROIE remote dataset adapter is implemented and tested
  without downloading either dataset to a teammate's computer.
- All 22 backend tests pass.
- No Vultr GPU has been created, Qwen has not been downloaded, and no LoRA has
  been trained yet.
- Sentry, Devin, and ASI:One runtime integration remain for the hackathon.

## Demo promise

The finished demo has three tabs:

1. **Chat** runs the same receipt task through Qwen + LoRA, Claude, and GPT with
   live output and separate clocks.
2. **Performance** shows saved benchmark runs, performance over time, the
   deterministic winner, and separate Time, Cost, and Accuracy charts.
3. **Skills & Adapters** lists available adapters and shows the live build stage
   when a new skill is being created.

The comparison models read the complete skill instructions. Qwen performs the
same task with the LoRA adapter and the short runtime prompt. Every condition is
measured independently.

## End-to-end architecture

```text
User describes workflow
        |
        +-- Sentry Session Replay records the progress UI
        v
ROOT TRACE: dapta.adapter_build
        |
        v
JEV Router
        |-- existing adapter --> run Qwen + LoRA
        |-- unstable workflow --> fallback or reject
        v
Skill Architect -- Kimi K3 via OpenRouter
        |  skill_spec.json
        |  output_schema.json
        |  evaluation_plan.json
        v
Skill Integration Engineer -- Devin
        |  deterministic plugin, tests, PR, human approval
        v
Data Builder -- Kimi K3 via OpenRouter
        |  TRAIN / DEV / TEST / OOD candidates
        v
Deterministic Dataset Validator
        |  schema, arithmetic, duplicates, leakage, frozen TEST hash
        v
Training Worker -- Vultr
        |  Qwen3-VL-8B QLoRA, telemetry, automatic destruction
        v
Independent Evaluator -- code + gpt-oss-120b
        |  accuracy, cost, time, and failure report
        v
PASS --> Adapter Registry / Hugging Face
FAIL --> targeted DEV feedback or scoped Devin repair
        |
        v
ASI:One Dapta Gateway Agent exposes completed backend actions
```

Only Qwen is trained and served on Vultr. Kimi K3 and gpt-oss-120b are logical
agents called through OpenRouter. Claude and GPT are comparison baselines, not
teachers and not part of adapter training.

## What the project owner has completed

- Created the Vultr account and applied the `$250` hackathon credit.
- Created a Vultr API key and restricted API access to the current public IPv4
  address.
- Registered the local public SSH key with Vultr.
- Selected the first worker configuration: one NVIDIA A40 with 48 GB VRAM in
  `ewr`, plan `vcg-a40-24c-120g-48vram`, at `$1.712/hour` when last verified.
- Created the private Hugging Face adapter repository and a token with write
  access. Hugging Face is the durable adapter registry; full Qwen weights are
  not uploaded again.
- Created and funded the OpenRouter account and configured its account limit.
- Added the JEV API key for adapter routing.
- Created the Sentry organization and 14-day trial. The Dapta Sentry project
  values are not yet connected to the application.
- Created the private Dapta GitHub repository.

Secrets live only in `.env`, which is ignored by Git. `.env.example` documents
the required variable names without credential values.

## Ordered build plan and completion state

### 1. Verify cloud access — complete

The read-only preflight verifies Vultr account access, A40 availability, region,
SSH key registration, configured hourly price, Hugging Face repository write
access, OpenRouter model availability and account limit, and JEV access.

No infrastructure is created by the preflight.

### 2. Build the cost and usage ledger — complete

`backend/dapta/ledger.py` stores each workflow run, model call, provider
generation, benchmark condition, and Vultr allocation in SQLite. It keeps the
following separate for each request:

- input, output, reasoning, cached, image, and audio usage;
- provider-reported cost and pricing source;
- latency and time to first token;
- model, provider, role, stage, and benchmark condition; and
- dataset, code, adapter, and hardware versions.

Money uses decimal/nanodollar arithmetic. Provider generation IDs cannot be
billed twice. Compute cost covers the complete instance lifetime from creation
through deletion, not only the training process. The UI will compute the winner
from stored metrics without another LLM call.

### 3. Build the adapter-job orchestrator — complete

`backend/dapta/jobs.py` persists jobs, immutable transitions, stage attempts,
artifacts, idempotency keys, and adapter registry entries:

```text
queued -> architecting -> integrating -> building_dataset -> validating
       -> provisioning_gpu -> training -> evaluating -> publishing
       -> ready | failed
```

Paid stages can be claimed only once. The current FastAPI surface is:

```text
POST /jobs
GET  /jobs
GET  /jobs/{job_id}
GET  /adapters
GET  /benchmarks
POST /compare     # intentionally returns 501 until Qwen serving exists
```

### 4. Connect the OpenRouter agents — complete

`backend/dapta/openrouter.py` provides one metered client and three role-bound
agents:

- Kimi K3 Skill Architect;
- Kimi K3 Data Builder; and
- gpt-oss-120b Independent Evaluator.

Every call requests usage metadata, reconciles the provider generation, and
writes one attributable ledger record. Prompts and receipt contents are not
stored in usage metadata.

### 5. Generate and review the receipt skill contract — implementation complete

The real Kimi K3 Skill Architect generated:

- `research/adapter-assets/skills/receipt-to-expense/skill_spec.json`
- `research/adapter-assets/skills/receipt-to-expense/output_schema.json`
- `research/adapter-assets/skills/receipt-to-expense/evaluation_plan.json`

`review.json` records the architect run, provider-reported usage/cost, canonical
artifact hashes, and engineering review checks. The contract requires visible
evidence only, nulls for absent fields, a closed JSON schema, and deterministic
CORD/SROIE metrics. It explicitly forbids inferred currency, expense category,
policy approval, or reimbursement eligibility.

The contract is approved for implementation. A project-owner signoff remains a
hard gate before paid GPU training.

The first attempt reached its original response-token cap and was recorded as a
failed run. The retry used the corrected response budget and succeeded:

| Architect attempt | Input | Output | Reasoning | Cost |
| --- | ---: | ---: | ---: | ---: |
| Truncated, preserved as failed | 320 | 6,000 | 4,961 | `$0.081268800` |
| Successful contract | 336 | 3,933 | 2,367 | `$0.032026644` |
| **Total OpenRouter spend** | **656** | **9,933** | **7,328** | **`$0.113295444`** |

No GPU cost was incurred by either attempt.

### 6. Build the remote CORD/SROIE adapter — code complete

`backend/dapta/receipts.py` reads the immutable revisions in
`research/adapter-assets/sources.lock.json` and:

1. refuses to materialize unless `DAPTA_REMOTE_WORKER=1`;
2. streams the pinned CORD v2 and SROIE revisions into an ephemeral job folder;
3. maps CORD menu, subtotal, tax, service, and total labels into the contract;
4. maps SROIE company, address, date, and total labels into the same contract;
5. uses CORD's official train/validation/test split;
6. creates a stable SROIE DEV subset from train and never changes SROIE TEST;
7. writes Qwen multimodal TRAIN/DEV/TEST conversations and receipt images;
8. freezes TEST with SHA-256 and writes a reproducibility manifest; and
9. deletes the Hugging Face source cache after materialization.

Dataset rows, receipt images, archives, and model weights are not committed and
have not been downloaded locally. Tests exercise the complete adapter with
in-memory fixtures. The real remote materialization waits for the Vultr worker
lifecycle in Step 8.

### 7. Finish deterministic validation — partially complete

The validator already checks required splits, row shape, safe image paths,
duplicate IDs, duplicate image/prompt cases, split leakage, and the frozen TEST
hash. Remaining receipt-specific gates are:

- JSON Schema validation of every completion;
- decoded-image readability and minimum dimensions;
- amount and line-item arithmetic;
- source/preprocessor manifest verification; and
- a validation report that advances or fails the persisted job.

This is the next coding step and must finish before a GPU can be provisioned.

### 8. Build the Vultr worker lifecycle — not started

The worker will check plan availability, create exactly one A40, install the
pinned environment, check out an exact commit, materialize the remote dataset,
run training, upload durable artifacts, record final cost, and delete the
instance in a `finally` path.

Stopping Linux does not stop Vultr billing; the instance must be deleted. Block
storage is intentionally not used for the first adapter.

### 9. Run the cloud smoke test — not started

After Steps 7 and 8:

1. verify `nvidia-smi`;
2. download Qwen3-VL-8B on the worker;
3. run base inference on one receipt;
4. train QLoRA on roughly 16–32 examples;
5. reload the adapter and run inference;
6. upload the smoke adapter and manifest to Hugging Face; and
7. delete the worker.

The smoke test must pass before a full training run.

### 10. Train the first receipt adapter — not started

Provision one A40, remotely materialize the validated full dataset, train the
Qwen QLoRA, save loss/hyperparameters/GPU telemetry, upload the adapter and
reproducibility manifest, and delete the worker. The durable artifact is the
small LoRA adapter, not a second copy of Qwen.

### 11. Evaluate all conditions — not started

The frozen examples and identical decoding settings will compare:

```text
Base Qwen + short prompt
Base Qwen + full Skill
Qwen + Dapta LoRA
Claude + full Skill
GPT + full Skill
```

Code scores schema validity, SROIE field accuracy, CORD field/line-item
accuracy, arithmetic, latency, TTFT, tokens, and cost. gpt-oss-120b reviews only
qualities deterministic code cannot score.

### 12. Serve Qwen with adapters — not started

The inference service will load base Qwen once, select the requested LoRA,
accept one or more receipt images, stream strict JSON, and return measurement
metadata. One GPU is sufficient for the hackathon inference demo.

### 13. Build the frontend — wireframe complete, application not started

The agreed three-tab interaction and performance charts are preserved in
`docs/wireframes/dapta-demo-wireframe.html` and `docs/UX_DEMO_SPEC.md`. The
production frontend should begin while the first full training job is running.

### 14. Add sponsor integrations — accounts partially ready

- **Sentry:** organization/trial created; SDK, Tracing, Logs, Session Replay,
  and AI monitoring are not connected yet.
- **Devin:** will implement skill plugins and scoped Sentry-driven repair PRs;
  hackathon access still needs to be connected.
- **JEV:** key and model are configured; runtime routing is not connected yet.
- **ASI:One:** intentionally scheduled after stable backend actions exist. The
  Dapta Gateway Agent will call the backend rather than host or train Qwen.

ASI:One does not block dataset validation, Vultr smoke training, or frontend
work.

## Run the repository locally

Create a virtual environment and install the backend requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
```

Add secrets only to `.env`, then run the read-only cloud check:

```bash
PYTHONPATH=backend python -m dapta.preflight
```

Run the test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=backend \
  python -m unittest discover -s backend/tests -v
```

Run the API:

```bash
PYTHONPATH=backend uvicorn dapta.api:app --reload
```

The real dataset adapter is deliberately remote-gated. On a disposable Vultr
worker it will be invoked as:

```bash
export DAPTA_REMOTE_WORKER=1
PYTHONPATH=backend python -m dapta.receipts \
  --job-dir "/mnt/dapta-job/$JOB_ID"
```

Do not run that command on a teammate laptop. A small `--limit 32` run is used
only after the automatic worker teardown path exists.

## Repository map

```text
backend/
  dapta/                 ledger, jobs, agents, receipt adapter, training
  tests/                 fast deterministic unit tests
frontend/                demo application boundary
docs/                    architecture, build plan, UX, and durable context
research/adapter-assets/
  sources.lock.json      immutable remote dataset catalog
  skills/                reviewed contracts and public reference skills
```

Generated datasets, SQLite ledgers, model caches, checkpoints, and adapters are
runtime artifacts and must remain outside Git.

## Immediate next sequence

```text
Finish Step 7 deterministic receipt validation
  -> implement Step 8 guarded Vultr worker lifecycle
  -> run Step 9 tiny cloud smoke test
  -> review cost and output
  -> start Step 10 full receipt training
  -> build frontend and sponsor integrations while training runs
```

No paid GPU should be created before the validation report, budget checks, and
automatic deletion path all pass.
