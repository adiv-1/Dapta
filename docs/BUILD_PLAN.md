# Dapta team build plan

Last updated: October 17, 2026

## Goal

Dapta converts stable, repetitive AI workflows into evaluated LoRA adapters
for `Qwen/Qwen3-VL-8B-Instruct`.

The first complete workflow is receipt images to a structured expense report.
The demo compares the specialized Qwen adapter with frontier models on
accuracy, time and cost.

## Model deployment decision

Only Qwen runs on Vultr.

- **Qwen3-VL-8B:** trained and served on a Vultr GPU.
- **Kimi K3:** called through OpenRouter as Skill Architect and Data Builder.
- **gpt-oss-120b:** called through OpenRouter as the Independent Evaluator.
- **Claude and GPT:** called through OpenRouter as comparison baselines only.
- **JEV:** selects an existing adapter, requests a new adapter or chooses the
  general fallback.

Kimi K3 and gpt-oss-120b are not downloaded to Vultr. Hosting them would use
the compute needed for Qwen without improving the product demonstration.

## Initial cloud configuration

Use one GPU for the first smoke test and first adapter:

```dotenv
VULTR_REGION=ewr
VULTR_GPU_PLAN_ID=vcg-a40-24c-120g-48vram
VULTR_OS_ID=1743
```

The selected plan is one NVIDIA A40 with 48 GB VRAM. The four-A40 plan should
be used only after several validated jobs are ready to run concurrently.

Block storage is not needed initially. Model weights and datasets live on the
worker's temporary disk. Approved LoRA adapters are uploaded to Hugging Face,
and the Vultr instance is destroyed after the job.

## Complete workflow

```text
User describes workflow
        |
        +-- Sentry Session Replay records the progress UI
        v
ROOT TRACE: dapta.adapter_build
        |
        v
JEV Router
Does an adapter exist? Is the workflow stable and testable?
        |
        +-- Existing adapter --> run Qwen + LoRA
        +-- Unsafe/unstable --> general fallback or reject
        v
Skill Architect -- Kimi K3
        |  skill_spec.json
        |  output_schema.json
        |  evaluation_plan.json
        v
Skill Integration Engineer -- Devin
        |  deterministic plugin, tests and registry entry
        |  reviewable PR and human approval
        v
Data Builder -- Kimi K3
        |  TRAIN / DEV / TEST / OOD candidates
        v
Deterministic Dataset Validator
        |  schema, arithmetic, duplicates and split isolation
        |  freezes TEST and records its hash
        v
Training Worker -- Vultr
        |  Qwen3-VL-8B QLoRA
        |  checkpoints, loss, GPU-hours and cost
        v
Independent Evaluator -- code + gpt-oss-120b
        |  accuracy, cost, time and failure report
        v
PASS --> Adapter Registry / Hugging Face
FAIL --> targeted DEV feedback or scoped Devin repair
```

## Exact build order

### 1. Correct and verify cloud access

Completed October 17:

- The first-run plan is one A40 in `ewr`.
- The local public SSH key is registered with Vultr; the private key never left
  the computer.
- Vultr, Hugging Face, OpenRouter, and JEV pass the read-only preflight.
- The Hugging Face token has write access to the private adapter repository.
- The configured hourly rate matches Vultr's current plan catalog.

### 2. Build the cost and usage ledger

Completed October 17. See `docs/COST_AND_USAGE_LEDGER.md` and
`backend/dapta/ledger.py`.

Every model request and GPU job must record:

- job ID and stage;
- model and provider;
- input, output and reasoning tokens;
- provider-reported API cost;
- latency and time-to-first-token;
- Vultr instance ID, hourly rate and running time;
- estimated GPU cost; and
- dataset, code and adapter versions.

The Performance tab reads these stored measurements. It must not ask another
LLM to summarize the winner.

### 3. Build the adapter-job orchestrator

Completed October 17. Jobs, transitions, immutable events, artifacts, adapter
registry entries, idempotency keys, and one-time paid-stage claims are persisted
in the same SQLite database as the usage ledger.

Persist this state machine:

```text
queued
  -> architecting
  -> integrating
  -> building_dataset
  -> validating
  -> provisioning_gpu
  -> training
  -> evaluating
  -> publishing
  -> ready | failed
```

Initial API surface:

```text
POST /jobs
GET  /jobs/{job_id}
GET  /adapters
POST /compare
GET  /benchmarks
```

### 4. Connect the OpenRouter agents

Completed October 17 with mocked requests only; no OpenRouter credit was used.
The client requests response usage, reconciles each generation, and records each
role independently in the cost ledger.

Build one OpenRouter client that captures usage and cost independently for
every request. Use role-bound sessions for:

- Kimi K3 Skill Architect;
- Kimi K3 Data Builder; and
- gpt-oss-120b Independent Evaluator.

These are logical agents, not separate GPU deployments.

### 5. Generate and approve the receipt skill contract

Implemented October 17. Kimi K3 generated the contract through the metered
OpenRouter path. The versioned artifacts and review record live in
`research/adapter-assets/skills/receipt-to-expense/`. They are approved for
implementation; project-owner signoff remains required before paid training.

The Skill Architect creates:

```text
skill_spec.json
output_schema.json
evaluation_plan.json
```

The contract defines inputs, normalized expense output, rules, exclusions and
deterministic metrics. It must be approved before dataset construction.

### 6. Build the remote CORD/SROIE adapter

Implemented and fixture-tested October 17 in `backend/dapta/receipts.py`. Real
dataset materialization remains remote-gated and will run only through the
guarded Vultr worker lifecycle in step 8.

The worker reads `research/adapter-assets/sources.lock.json` and then:

1. Streams or temporarily downloads the pinned datasets on Vultr.
2. Converts them into Qwen multimodal conversations.
3. Creates TRAIN/DEV/TEST in the worker's temporary job directory.
4. Keeps frozen TEST inaccessible to the Data Builder.
5. Deletes the source cache when the worker is destroyed.

No dataset rows, images, archives or model weights are stored on a teammate's
computer or committed to Git.

### 7. Finish deterministic validation

Before a GPU is provisioned, validate:

- output schema and required fields;
- receipt totals and line-item arithmetic;
- image readability;
- duplicate examples and images;
- cross-split leakage;
- frozen TEST hash;
- safe paths; and
- dataset and preprocessing version hashes.

Any failure stops the job before GPU spending begins.

### 8. Build the Vultr worker lifecycle

The cloud worker must:

1. Check plan availability.
2. Create the GPU instance.
3. Install the pinned CUDA and Python environment.
4. check out the exact code commit.
5. Retrieve Qwen and temporary datasets.
6. Run smoke or full training.
7. Upload adapters, metrics and manifests.
8. Destroy the instance after success or terminal failure.

Stopping Linux does not end billing. The Vultr instance must be deleted.

### 9. Run the cloud smoke test

Only after the worker lifecycle is implemented:

1. Confirm `nvidia-smi`.
2. Download Qwen3-VL-8B on the worker.
3. Run base inference on one receipt.
4. Train QLoRA on approximately 16--32 examples.
5. Reload the adapter and run inference.
6. Upload the smoke adapter to Hugging Face.
7. Destroy the instance.

The smoke test must succeed before a long training run begins.

### 10. Train the first real adapter

1. Provision one A40.
2. Materialize the validated receipt dataset remotely.
3. Train the QLoRA.
4. Save loss, hyperparameters and GPU telemetry.
5. Upload the adapter and reproducibility manifest to Hugging Face.
6. Destroy the worker.

The durable artifact is the LoRA adapter, not another copy of Qwen.

### 11. Evaluate the adapter

Run the same frozen examples and decoding settings through:

```text
Base Qwen + short prompt
Base Qwen + full Skill
Qwen + Dapta LoRA
Claude + full Skill
GPT + full Skill
```

Evaluation order:

1. Deterministic schema and arithmetic checks.
2. SROIE precision, recall, F1 and exact match.
3. CORD field and line-item accuracy.
4. Dapta ExpenseBench workflow accuracy.
5. gpt-oss review only for qualities that cannot be scored with code.

Store accuracy, total latency, time-to-first-token, tokens and cost for every
condition.

### 12. Serve Qwen with adapters

The inference service loads base Qwen once and then loads the requested LoRA.
It accepts multiple receipt images, streams structured output and returns
measurement metadata. One GPU is sufficient for the hackathon inference demo.

### 13. Build the frontend

The demo has three tabs:

1. **Chat** races Qwen LoRA, Claude and GPT with live output and clocks.
2. **Performance** shows saved-run pills, performance over time, a
   deterministic winner and separate Time, Cost and Accuracy charts.
3. **Skills & Adapters** lists ready adapters and shows live build stages.

### 14. Add sponsor integrations while training runs

- **Sentry:** Tracing, Logs, Session Replay and AI Agent Monitoring.
- **Devin:** skill-plugin PRs and Sentry-driven repair PRs.
- **JEV:** existing-adapter, new-adapter and fallback routing.
- **ASI:One:** one Dapta Gateway Agent over the completed backend API.

ASI:One does not block training and can be connected after the backend actions
are stable.

## Critical path for October 17

```text
Cost ledger
  -> job orchestrator
  -> receipt skill contract
  -> remote CORD/SROIE conversion
  -> deterministic validation
  -> Vultr worker
  -> Qwen cloud smoke test
  -> full receipt adapter training
```

Once the full training job starts, work shifts to the frontend, evaluation,
Sentry and sponsor integrations.

## Required configuration status

Available for the first adapter:

- Vultr API key
- Hugging Face token and adapter repository
- OpenRouter API key
- JEV API key

No credential is currently missing for the first cloud smoke test. Provisioning
still waits for the worker lifecycle and its budget/destruction guards.

Required later but not blocking the first adapter:

- Sentry project values
- Devin credentials or hackathon access
- Agentverse / ASI:One credentials
- production database and Redis URLs

## Definition of done for the demo

- A receipt LoRA is versioned on Hugging Face.
- The frozen benchmark compares Qwen base, full Skill, LoRA and frontier
  baselines on identical examples.
- Time, cost and accuracy are stored per run and visible in the UI.
- The UI shows a real adapter being selected and a new build progressing.
- Sentry contains a real trace and at least one actionable finding.
- Devin produces a reviewable integration or repair PR.
- The ready receipt workflow runs through the ASI:One Gateway Agent.
- Every Vultr training worker is automatically deleted after completion.
