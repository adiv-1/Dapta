# Dapta

Dapta turns stable, repetitive workflows into measured LoRA specialists for
`Qwen/Qwen3-VL-8B-Instruct`. It compares the adapter with configured frontier
models on output quality, time, tokens, and cost, and preserves the evidence for
the demo instead of making another model call to summarize it.

## Current state

Implemented and tested:

- FastAPI controller with routing, comparison runs, jobs, progress events,
  adapter registry, benchmark history, document extraction, and ASI-ready tools.
- SQLite cost ledger with request-level tokens, provider cost, latency, GPU
  active time, idempotency, and separate factory/training/inference attribution.
- JEV routing with a registry-similarity fallback that requires confirmation.
- Kimi K3 skill-architect and data-builder stages through OpenRouter.
- Devin v3 skill-integration sessions and PR evidence synchronization.
- deterministic split validation with frozen test hashes and `awaiting_compute`.
- Hostinger A100 controller/model-worker containers and LoRA inference loading.
- private Base44 React app with Chat, Performance, and Skills & Adapters tabs.
- Base44 authentication boundary, secret proxy, and durable run/job snapshots.
- Sentry backend traces/logs plus masked React Session Replay configuration.
- NDA seed adapter and the frozen 83-document benchmark produced on Colab.

The NDA adapter is stored at
`adityave/dapta-lora-adapters/nda-intake/qwen3-vl-8b-v1`. Its frozen benchmark
improved field F1 from 57.58% (base) and 61.28% (base + Skill) to 70.75% with
the LoRA; exact-form accuracy improved from 6.02% to 33.73%.

Still external/deployment-dependent:

- provision or provide SSH access to the Hostinger A100;
- point the Base44 `DAPTA_CONTROLLER_URL` secret at that deployment;
- add the Sentry DSNs and confirm telemetry in the Sentry UI;
- register the already-exposed tools as an ASI:One/Agentverse agent.

## Runtime architecture

```text
Invite-only Base44 app
        |
        v
Base44 secret proxy + durable snapshots
        |
        v
FastAPI controller on Hostinger
        |-- JEV: adapter routing
        |-- OpenRouter: Kimi, GPT-OSS, Claude, GPT
        |-- Devin: tested skill integration PR
        |-- Qwen worker: base model + selected LoRA
        |-- SQLite: cost, usage, jobs, benchmarks
        |-- Hugging Face: adapters and manifests
        `-- Sentry: replay, traces, logs, AI spans
```

New adapter jobs follow:

```text
queued -> architecting -> integrating -> building_dataset -> validating
       -> awaiting_compute -> provisioning_gpu -> training -> evaluating
       -> publishing -> ready | failed
```

The pause at `awaiting_compute` is deliberate: an agent can finish the safe,
deterministic preparation without claiming GPU work happened.

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
PYTHONPATH=backend .venv/bin/python -m unittest discover -s backend/tests -v
cd frontend && npm run build
```

## Hostinger A100 deployment

On the GPU VM, clone the repository, create `.env`, and run:

```bash
docker compose -f deploy/docker-compose.hostinger.yml up -d --build
curl http://127.0.0.1:8000/health
```

Only port `8000` should be public. The Qwen worker stays on the private Compose
network. Set a long random `DAPTA_CONTROL_TOKEN`, expose the controller through
Hostinger HTTPS, then save the same URL and token as Base44 project secrets.

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
- Base44 receives only the controller URL/token and its Sentry browser DSN.
- Hostinger receives OpenRouter, JEV, Devin, Hugging Face, Qwen, and Sentry
  server credentials.
- Hugging Face receives adapter weights, manifests, `SKILL.md`, and benchmark
  artifacts—not private judge documents.

Further project rationale and dataset research are in `docs/`.
