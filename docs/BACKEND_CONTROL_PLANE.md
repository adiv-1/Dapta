# Backend control plane

Parts 3 and 4 of the build plan are implemented without provisioning a GPU or
making a paid model request.

## Job lifecycle

`JobStore` and `UsageLedger` share the SQLite file at `DAPTA_LEDGER_PATH`.
Every adapter build follows this persisted sequence:

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

Invalid skips fail before work begins. Each transition has an immutable event,
and each paid agent stage has a one-time database claim. A crashed worker cannot
silently repeat the same paid stage; recovery requires an explicit operator
decision or a new job.

Job creation requires an idempotency key. Repeating the same request returns the
same job, while reusing that key for different work returns a conflict.

The store also contains:

- content-addressed job artifacts;
- versioned adapter registry entries; and
- failure codes without prompt or document content in the usage ledger.

## OpenRouter agents

`FactoryAgents` exposes three role-bound sessions:

- Skill Architect: Kimi K3;
- Data Builder: Kimi K3; and
- Independent Evaluator: gpt-oss-120b.

Each call requests strict JSON and `usage.include`. After the response, Dapta
retrieves the generation record and writes native prompt, completion, reasoning,
and cached tokens, provider identity, latency, generation ID, and final cost to
the ledger. The agent response text stays outside the usage ledger.

The independent evaluator receives deterministic results, not raw TEST labels,
and cannot change objective scores or thresholds.

## HTTP surface

Implemented now:

```text
GET  /health
POST /jobs
GET  /jobs
GET  /jobs/{job_id}
GET  /adapters
GET  /benchmarks
```

`POST /compare` exists but intentionally returns `501` until Qwen inference is
implemented. A successful `POST /jobs` only persists a queued job; it does not
call OpenRouter or create a Vultr instance.

Run the API after installing dependencies:

```bash
set -a
source .env
set +a
PYTHONPATH=backend uvicorn dapta.api:app --reload
```

## Next boundary

The receipt contract and remote CORD/SROIE dataset adapter are implemented. The
next boundary is receipt-specific deterministic validation, followed by the
guarded Vultr worker lifecycle. Qwen weights are downloaded inside an ephemeral
Vultr worker only when validation and guaranteed destruction both pass.
