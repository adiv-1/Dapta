# Dapta LA Hacks 2026 master plan

Last updated: October 17, 2026

This is the current source of truth for the hackathon. When an older planning
document conflicts with this file, use this file.

## Product thesis

Dapta compiles stable, repetitive AI workflows into verified LoRA specialists
for a shared open-weight multimodal model.

> Pay for reasoning once, then reuse the specialist.

The claim is not that inference becomes free. Dapta measures the up-front cost
of specification, dataset creation, validation, training and evaluation, then
compares the amortized result with repeatedly prompting general models.

The first complete workflow is receipt images to a structured expense report.
The student and runtime model is `Qwen/Qwen3-VL-8B-Instruct` with a QLoRA
adapter. Closed frontier models are comparison baselines only; they do not
create training labels, repair data or evaluation truth.

## Complete adapter-build workflow

```text
User describes a workflow
        |
        +-- Sentry Session Replay records the UI flow with sensitive data masked
        v
ROOT TRACE: dapta.adapter_build
job_id / adapter_id / workflow_type / dataset_hash
        |
        v
JEV Router                                      span: router.classify
Does an adapter already exist?
Is the workflow stable, repeated and testable?
        |
        +-- Existing adapter --> run Qwen + LoRA
        +-- Unsafe/unstable --> general fallback or reject
        |
        v
1. Skill Architect -- Kimi K3                   span: gen_ai.skill_architect
        |  skill_spec.json
        |  output_schema.json
        |  evaluation_plan.json
        v
2. Skill Integration Engineer -- Devin          span: agent.devin.integrate
        |  implements deterministic oracle/validator plugin
        |  adds registry manifest and unit tests
        |  runs tests and opens a reviewable PR
        |  human/code-review approval gate
        v
3. Data Builder -- Kimi K3                      span: gen_ai.data_builder
        |  TRAIN / DEV / TEST / OOD candidates
        |  uses the merged deterministic plugin for truth where possible
        v
Deterministic Dataset Validator                 span: dataset.validate
        |  schema, labels, arithmetic, paths, duplicates and split isolation
        |  freezes TEST and records its hash
        v
Deterministic Training Worker -- Vultr          span: model.train_qlora
        |  trains Qwen3-VL 8B QLoRA
        |  checkpoints, loss, GPU-hours and cost
        v
4. Independent Evaluator -- code + gpt-oss     span: model.evaluate
        |  public benchmark, product accuracy, cost, time and failures
        v
PASS --> Adapter Registry / Hugging Face        span: registry.publish
FAIL --> classify the failure
          |
          +-- Model-quality failure --> targeted DEV feedback to Data Builder
          +-- Pipeline/code failure --> Sentry issue --> scoped Devin repair PR
```

The numbered AI roles are logical agents. They do not require one model server
or GPU each.

## Why Devin is part of the product

Using Devin merely to write the hackathon project is not the product
integration. Dapta uses Devin in two explicit, auditable roles.

### A. Skill Integration Engineer

After Kimi creates an approved skill specification, Devin receives a scoped
software-engineering task against the Dapta repository. It may:

- implement the skill's deterministic schema and validator plugin;
- implement arithmetic or rule-based oracle functions;
- add test fixtures and unit tests derived from the approved specification;
- register the new skill with the backend; and
- run tests and open a PR for approval.

Devin does not invent training labels, approve its own PR, alter the frozen TEST
set or decide whether the LoRA passes. Kimi remains the open-weight data agent,
code remains the preferred source of objective truth, and the independent
evaluator owns the quality gate.

The integration is complete only after the PR passes deterministic tests and is
approved. The resulting commit SHA is stored in the adapter-build manifest.

### B. Sentry-driven Pipeline Repair Engineer

Sentry may identify a genuine code or infrastructure failure such as a broken
schema migration, duplicate job launch, publishing error or reproducible
training crash. Dapta can start a scoped Devin session containing:

- the Sentry issue and trace identifiers;
- the failing job and stage;
- sanitized structured logs;
- the failing test or reproduction command; and
- the exact files or component Devin is allowed to change.

Devin investigates, writes a regression test, implements a fix, runs tests and
opens a PR. A human approves it. Accuracy failures are never sent to Devin;
they follow the bounded DEV-feedback loop instead.

This creates a strong combined story: Sentry finds and explains a real weakness
in the agent factory, and Devin turns that evidence into a tested repair.

## Sentry observability design

The required products are Tracing, Logs, AI Agent Monitoring and Session
Replay. Every adapter build uses one trace, with trace context propagated into
asynchronous workers.

Record low-cardinality metadata rather than prompts, documents or secrets:

- `job_id`, `adapter_id`, model and dataset versions;
- stage duration, retry count and failure status;
- OpenRouter input/output tokens, reported cost and latency;
- dataset sizes, invalid records, duplicates and leakage count;
- GPU type, GPU-hours, estimated cost, loss and checkpoint;
- evaluation F1, exact-match, schema validity and pass/fail; and
- Devin session ID, PR URL, test result and repair duration.

Sentry observes Dapta; it is not Dapta's job or benchmark database. The product
database remains authoritative. Receipt images, prompt contents, API keys and
personal expense data must be scrubbed or masked.

The judging story must use a real finding. Examples to watch for include a
trace showing repeated prompt cost, a duplicate GPU job visible in logs, a slow
model download, a failed publish operation or a UI progress state caught in
Session Replay. Record measured before/after results once a real issue is fixed.

## Runtime and ASI:One integration

The internal agents are not embedded in Qwen and do not all need Agentverse
registrations. Register one public **Dapta Gateway Agent**:

```text
User in ASI:One
    -> ASI:One discovers Dapta through Agentverse
    -> Agent Chat Protocol request
    -> Dapta Gateway Agent
    -> Dapta backend and model endpoints
    -> completed result returned inside the ASI:One conversation
```

Its stable actions are:

- `list_skills`
- `run_skill`
- `request_adapter`
- `get_build_status`
- `get_benchmark`

The complete primary ASI:One demo is execution of an already trained receipt
adapter, because a multi-hour training job cannot finish during judging. A new
adapter request may return a job ID and progress card as a secondary workflow.
The gateway is a thin protocol adapter over the same backend used by the custom
frontend.

Fetch requires at least one Agentverse agent, Agent Chat Protocol support,
ASI:One discovery, meaningful execution or orchestration, a complete primary
workflow inside ASI:One and a public repository with test instructions. The
submission also needs the ASI:One Submission Agent and a 3--5 minute Fetch demo.
See the [official hackpack](https://innovationlab.fetch.ai/events/hackathons/la-hacks-ai-hackathon-2026/hackpack).

## Evaluation plan

The first adapter uses a public benchmark plus a product-specific held-out set.

1. **SROIE Task 3** is the headline public receipt key-information extraction
   benchmark. Report precision, recall, F1 and exact match.
2. **CORD** measures richer receipt structure and line-item extraction.
3. **Dapta ExpenseBench** measures the actual multi-image expense workflow,
   including normalization, duplicates, totals, categories and valid output.
4. A small **DocVQA or OCRBench** slice checks that specialization did not
   destroy general document understanding.

Objective scores come from code. A model judge is used only for qualities that
cannot be scored deterministically. TEST is frozen before iterative repair and
never used to generate feedback.

Compare identical examples and decoding settings across:

- base Qwen with a short prompt;
- base Qwen with the full Skill;
- Qwen plus the Dapta LoRA;
- Claude with the full Skill; and
- GPT with the full Skill.

The judge-facing result should report accuracy, total latency, TTFT, token use,
per-request cost, adapter-build cost and break-even request volume. Prompt-token
reduction is not presented as identical to dollar savings.

## Cloud and artifact lifecycle

- Vultr hosts ephemeral GPU training workers and, when needed, inference.
- A CPU control plane owns the API, job state, registry metadata and Agentverse
  bridge.
- Hugging Face stores versioned adapter artifacts independently of a particular
  Vultr account.
- Each adapter manifest records the base model, dataset hash, TEST hash,
  hyperparameters, metrics, code commit and artifact URI.
- A worker downloads inputs, trains, uploads durable outputs and is destroyed
  after completion or terminal failure. Shutdown alone is not treated as a
  billing stop.
- Start with one isolated QLoRA job per GPU. Parallel jobs use separate GPUs;
  the model is not silently spread across every available GPU.

## Demo product

The frontend has three tabs.

1. **Chat** races Qwen LoRA, Claude and GPT on the same receipt images. Every
   lane shows output, elapsed time and cost.
2. **Performance** provides saved-run pills, performance over time, a
   deterministic winner summary and separate Time, Cost and Accuracy charts.
3. **Skills & Adapters** lists ready adapters and shows live build stages for a
   requested skill.

The demo sequence is:

1. Upload receipts and run the three-model comparison.
2. Show the structured reports and live clocks.
3. Open Performance and show held-out results, not just one favorable example.
4. Open Skills & Adapters and request an unknown stable workflow.
5. Show the Kimi -> Devin -> Kimi -> validator -> Vultr -> evaluator trace and
   the Devin PR/test evidence.
6. Show one real Sentry finding and the resulting improvement or Devin repair.
7. Run the ready receipt workflow inside ASI:One.

## Target submissions

Primary targets:

- Intuit QuickBooks / System One
- Fetch ASI:One Agent Challenge
- Veri open-source model / post-training
- MLH Best Use of Vultr
- Sentry
- Cognition Best Use of Devin
- Model Learning
- Applied Intelligence

Paper Best Designed Hack is optional if the interface reaches sufficient
polish. Browser Use is optional only if expense entry into a real web workflow
is completed without destabilizing the core demo.

## Required external configuration

Secrets stay in `.env` and never enter Git, Sentry events or browser bundles.
The expected integrations are:

```dotenv
VULTR_API_KEY=
HF_TOKEN=
OPENROUTER_API_KEY=
JEV_API_KEY=

DEVIN_API_KEY=
DEVIN_ORG_ID=

SENTRY_DSN=
VITE_SENTRY_DSN=
SENTRY_AUTH_TOKEN=
SENTRY_ORG=
SENTRY_PROJECT=
SENTRY_ENVIRONMENT=development

AGENTVERSE_API_KEY=
ASI_ONE_API_KEY=
```

Create service-scoped credentials with minimum privileges where the provider
supports them. `SENTRY_AUTH_TOKEN` is build-only and must never be exposed to
the frontend. Devin automation should use a service user rather than a personal
credential when the hackathon account supports it.

## Scope and safety gates

- Compile only stable, repeated and objectively testable workflows.
- Reject or fall back for live-information, one-off, unsafe or underspecified
  requests.
- Never allow the Data Builder to approve its own TEST set.
- Never allow Devin to change TEST labels or quality thresholds.
- Require an idempotency key before provisioning a GPU.
- Do not publish an adapter until validation and evaluation pass.
- Preserve an immutable record of dataset, code, adapter and metric versions.
- Keep proprietary frontier models out of supervision generation.
- Sanitize the repository before making it public for submission.

## Implementation order

1. Freeze the expense Skill, schema and deterministic scoring contract.
2. Implement SROIE and Dapta ExpenseBench evaluation harnesses.
3. Instrument the backend with Sentry tracing, logs and AI monitoring.
4. Connect the Kimi Skill Architect and Data Builder.
5. Add the Devin session boundary for skill integration and scoped repairs.
6. Validate a tiny multimodal dataset and run a short Vultr smoke train.
7. Train and evaluate the full receipt adapter, then publish it.
8. Serve the adapter and connect the three-tab frontend.
9. Add Session Replay and capture a real Sentry-led improvement.
10. Add the thin Agentverse/ASI:One gateway and rehearse the full submission.

## Final positioning

> Dapta is an observable neural-skill compiler. Open-weight agents turn a stable
> workflow into data, Devin turns its specification into tested software,
> Vultr trains a Qwen specialist, independent evaluation decides whether it is
> safe to publish, Sentry proves where the system spends time and money, and
> ASI:One makes the finished specialist discoverable and executable.
