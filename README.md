# Dapta

> New engineer or agent: start with [`docs/AGENT_HANDOFF.md`](docs/AGENT_HANDOFF.md)
> and [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md). They describe the
> live deployment and the safe recovery path for paid GPU work.

Dapta turns stable, repetitive workflows into measured LoRA specialists for
`Qwen/Qwen3-VL-8B-Instruct`. It compares the adapter with configured frontier
models on output quality, time, tokens, and cost, and preserves the evidence for
the demo instead of making another model call to summarize it.

The default live comparison now uses `anthropic/claude-opus-5.5` and
`openai/gpt-5.6-sol` through OpenRouter. Override either provider ID with
`OPENROUTER_FRONTIER_MODEL_A` or `OPENROUTER_FRONTIER_MODEL_B`.

## Current state

Implemented and tested:

- FastAPI controller with routing, comparison runs, jobs, progress events,
  adapter registry, benchmark history, document extraction, and ASI-ready tools.
- SQLite cost ledger with request-level tokens, provider cost, latency, GPU
  active time, idempotency, and separate factory/training/inference attribution.
- JEV routing with a registry-similarity fallback that requires confirmation.
- Kimi K3 skill-architect and data-builder stages through OpenRouter.
- Devin v3 skill-integration sessions with generated skill evidence stored at `generated/<skill_id>/<version>/skill/` beside each adapter in Hugging Face; generated skills never modify GitHub.
- deterministic schema, label, leakage, duplicate, split, and frozen-test validation.
- RunPod single-GPU QLoRA training with a hard time/cost preflight, live telemetry,
  inference-model unload/reload, and exact GPU-active cost accounting.
- independent base/base+Skill/LoRA evaluation, user-approved DEV-only quality
  repairs, and benchmark-only regeneration when output truncation—not model
  quality—invalidates a score.
- private Supabase mirroring for factory stages and agent artifacts; uploaded
  judge documents, credentials, and internal reasoning are never mirrored.
- RunPod controller/model worker with dynamic Hugging Face LoRA loading.
- private Base44 React app with a conversational JEV workflow, guided specialist
  creation, a prior-adapter drawer, inline build progress, and a shared
  speed/token/cost comparison with saved history.
- Base44 authentication boundary, secret proxy, and durable run/job snapshots.
- Sentry backend traces/logs plus masked React Session Replay configuration and
  an explicit factory feedback panel that separates operational repair,
  benchmark repair, and genuine retraining decisions.
- NDA seed adapter and the frozen 83-document benchmark produced on Colab.

The NDA adapter is stored at
`adityave/dapta-lora-adapters/nda-intake/qwen3-vl-8b-v1`. Its frozen benchmark
improved field F1 from 57.58% (base) and 61.28% (base + Skill) to 70.75% with
the LoRA; exact-form accuracy improved from 6.02% to 33.73%.

The CSV cleaner/dashboard adapter is stored at
`generated/csv-clean-dashboard-v1/v20261018093318`. Its weights and frozen test
generations exist, but the original 768-token benchmark outputs were truncated.
It is truthfully marked `needs_repair`; use evaluation-only repair rather than
retraining it.

Remaining external step: register the already-exposed tools as an
ASI:One/Agentverse agent after the judging API is frozen.

## Runtime architecture

```text
Invite-only Base44 app
        |
        v
Base44 secret proxy + durable snapshots
        |
        v
FastAPI controller on RunPod
        |-- JEV: adapter routing
        |-- OpenRouter: Kimi, GPT-OSS, Claude, GPT
        |-- Devin: scoped skill integration evidence (no GitHub writes)
        |-- Qwen worker: base model + selected LoRA
        |-- SQLite: cost, usage, jobs, benchmarks
        |-- Supabase: durable agent stages and artifacts
        |-- Hugging Face: adapters and manifests
        `-- Sentry: replay, traces, logs, AI spans
```

New adapter jobs follow:

```text
queued -> architecting -> integrating -> building_dataset -> validating
       -> awaiting_compute -> provisioning_gpu -> training -> evaluating
       -> publishing -> ready | failed
```

`awaiting_compute` is a real budget gate. The autonomous runner advances after
the cost cap and connected GPU checks pass; it never records training before
the worker starts.

## Product experience

Chat has one message composer with optional attachments. Each message and
clarification calls the routing endpoint and shows its decision inline,
including whether it used JEV or the registry fallback. Existing specialists
request missing input in the conversation. New specialists collect the output
contract and representative examples, then wait for **Build specialist** before
creating a paid factory job. Follow-up questions are a guided intake flow;
the existing router remains responsible for choosing the path.
JEV calls use the [System One API](https://docs.typesafe.ai/introduction/quickstart)
with typed choices for each registered specialist, a new build, or an
unsupported workflow. Low-confidence specialist matches require confirmation.

Comparisons show latency, total tokens, and recorded cost once in an aligned
three-model matrix, with quality checks and expandable outputs underneath.
Live calls show their actual pending/completed/failed status; unavailable costs
stay blank and metric winners appear only after all three calls complete.
Playfair Display is reserved for the brand; Inter is used throughout the
working interface with tabular numerals for metrics.

Model responses share the same **Preview / JSON** renderer across LoRA, Claude,
and GPT. Self-contained Vega-Lite `dashboard_spec` outputs render as charts;
`cleaned_rows` / `cleaned_csv` and object arrays become paginated tables. Receipt
and expense outputs show reported totals, currencies, extracted fields, and line
items; support routing displays queue, priority, and response-code badges.
Every result can be expanded, copied, or downloaded as its original JSON, with
CSV table and SVG chart exports. Missing values remain missing; the preview is
not a quality score. Generated Python/HTML is source-only and never executed.
Chart rendering runs in a bounded worker using interpreted expressions and
inline data only; invalid or blocked specs leave the response available in JSON.
This is presentation support, not a change to training or receipt-image ingestion.

The Performance tab uses unambiguous **Comparison runs** and **Training
benchmarks** views. Equal accuracy is reported as `Equal` instead of awarding a
false winner. Completed comparisons are listed newest-first with the speed,
cost, accuracy, and overall winner for each run.

**Prior adapters** opens a bottom drawer containing reusable specialists and
build history. Selecting a specialist returns to chat for routing. Build
progress stays in the conversation, with each stage and its provider visible.
Every active build has a sticky **Stop build** control with confirmation. Stop
requests persist server-side: no later stage starts, active Devin sessions are
terminated, and a training/evaluation worker is stopped and reaped before Qwen
inference reloads. **Stopping safely** remains visible until cleanup is confirmed;
failed cleanup offers **Retry stop**. An already-running provider request may
finish and incur its existing charge. Uploaded/published artifacts and prior
adapters are not deleted. Use **Back to chat** after stopping to reset the view;
refreshing or starting a new conversation does not silently abandon active builds.
The factory feedback panel applies this decision order:

1. Repair infrastructure or an incomplete evaluation before judging quality.
2. Rerun a truncated benchmark without retraining existing weights.
3. Publish immediately when the frozen benchmark passes.
4. Apply DEV-only evaluator feedback and retrain only when verified quality
   misses the frozen acceptance threshold.

## API

The judge-facing app uses only these authenticated controller actions:

```text
POST /v1/route
POST /v1/documents/extract
POST /v1/comparisons
GET  /v1/runs
GET  /v1/runs/{id}
GET  /v1/runs/{id}/events
GET  /v1/benchmarks
GET  /v1/adapters
POST /v1/jobs
GET  /v1/jobs/{id}
POST /v1/jobs/{id}/cancel
GET  /v1/jobs/{id}/events
POST /v1/jobs/{id}/repair-evaluation
GET  /v1/agent/tools
POST /v1/agent/tools/{tool_name}
```

Mutating actions use idempotency keys. Provider secrets remain in the
controller or Base44 function environment and never enter the browser bundle.

## Local setup

```bash
cp .env.example .env
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-controller.txt
PYTHONPATH=backend .venv/bin/uvicorn dapta.api:app --reload
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

`VITE_DAPTA_API_URL=/api` uses the local Vite proxy. The deployed Base44 build
omits that variable and invokes `dapta-proxy`, so controller credentials remain
server-side.

Run verification:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=backend \
  .venv/bin/python -m unittest discover -s backend/tests -v
cd frontend && npm run build && npm test
```

## RunPod deployment

Use a persistent `/workspace` volume, clone the repository to
`/workspace/dapta`, install the full pinned runtime, and create
`/workspace/dapta/runtime.env` from `.env.example`:

```bash
python3 -m venv /workspace/dapta/.venv
/workspace/dapta/.venv/bin/pip install -r backend/requirements.txt
./deploy/runpod-start.sh
curl http://127.0.0.1:8000/health
```

Only HTTP port `8000` is public through the RunPod proxy. The Qwen worker binds
to loopback on `8002`. Set a long random `DAPTA_CONTROL_TOKEN`, then save the
proxy URL and the same token as Base44 project secrets. `runpod-stop.sh` stops
both local services but does not stop or delete the RunPod pod.

## Base44

The linked Base44 project lives in `frontend/base44`. Deployment is:

```bash
cd frontend
npx base44 deploy --build --yes
npx base44 visibility private
npx base44 secrets set DAPTA_CONTROLLER_URL=<https-controller-url>
npx base44 secrets set DAPTA_CONTROL_TOKEN=<same-random-token>
```

The `DaptaSnapshot` entity is service-role-only. The function mirrors UI result
records but never stores API keys or uploads user documents to Hugging Face.

## Environment ownership

- `.env` is ignored and is the only local location for provider credentials.
- Sentry auth/user tokens are server-side secrets. Never expose one through a
  `VITE_` variable; the browser only needs `VITE_SENTRY_DSN`, release, and
  sampling configuration.
- Base44 receives only the controller URL/token and its Sentry browser DSN.
- RunPod receives OpenRouter, JEV, Devin, Hugging Face, Supabase, Qwen, and Sentry
  server credentials.
- Supabase receives generated factory artifacts and stage evidence—not browser
  uploads, provider secrets, or chain-of-thought.
- Hugging Face receives adapter weights, manifests, `SKILL.md`, and benchmark
  artifacts—not private judge documents.

Local IDE context, virtual environments, pytest caches, generated `artifacts/`,
and notebook files are ignored by Git. The two local LoRA notebooks remain
available for experimentation but are not part of the deployable repository.

Further project rationale and dataset research are in `docs/`.

The live NDA comparison uses [`backend/dapta/skills/nda-intake/SKILL.md`](backend/dapta/skills/nda-intake/SKILL.md)
as its source of truth. Claude and GPT receive the complete skill; the LoRA adapter receives the
compact four-field inference prompt derived from its frozen benchmark contract. All three must return exactly
`effective_date`, `jurisdiction`, `parties`, and `term`.
