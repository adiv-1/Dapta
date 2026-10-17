# Cost and usage ledger

The ledger is the source of truth for Dapta's cost, token, and timing claims.
It lives at `DAPTA_LEDGER_PATH` and uses SQLite until the backend moves to a
hosted database.

## What is recorded

Every workflow or benchmark has a stable `run_id`. Each model request records:

- stage, agent role, benchmark case, and comparison condition;
- requested and provider-resolved model;
- provider request and generation IDs;
- start, completion, total latency, and time-to-first-token;
- prompt, completion, reasoning, cached, and cache-write tokens;
- media counts, finish state, and sanitized provider usage details; and
- USD cost, upstream cost, and whether the value was provider-reported,
  calculated, or estimated.

Every Vultr allocation separately records:

- instance, plan, region, GPU type, GPU count, and captured hourly rate;
- creation, workload, failure, and destruction events; and
- the full deployment-to-destruction cost period.

The run also stores the job, skill, adapter, and benchmark IDs. Dataset hashes,
code commits, adapter versions, and decoding settings belong in sanitized run
metadata.

## Accuracy rules

1. Money enters the ledger as decimal strings, never binary floating-point.
2. A nanodollar integer is stored beside every USD value for exact aggregation.
3. A provider generation ID can occur only once, preventing double billing.
4. Repeating the same record is idempotent; conflicting reuse of an ID fails.
5. OpenRouter calls are recorded only after the response usage and
   `/api/v1/generation?id=...` metadata have been collected. Generation metadata
   wins because it identifies the actual provider, native tokens, and final
   provider-reported cost for that one request.
6. Vultr compute starts at instance creation, not workload start, and ends only
   at destruction. A provider-reported charge supersedes the calculated value.
7. Prompts, completions, documents, images, messages, API keys, and secrets are
   forbidden in ledger metadata.

## Benchmark isolation

Use one `benchmark_case_id` for the same frozen example and a separate
`condition` for each candidate:

```text
base_qwen_short_prompt
base_qwen_full_skill
qwen_lora
claude_full_skill
gpt_full_skill
```

This keeps tokens and cost attributable to the exact request instead of the
OpenRouter account as a whole. `run_summary()` aggregates the saved rows by
condition; the frontend reads this data directly and does not ask an LLM to
choose a winner.

## OpenRouter reconciliation

For every request:

1. Generate a Dapta `call_id` before sending the request.
2. Capture monotonic start, first-token, and completion times.
3. Save the returned generation ID and response usage.
4. Query OpenRouter generation metadata with that exact generation ID.
5. Call `record_openrouter_call()` once with both payloads.

The OpenRouter account usage before and after a benchmark is a reconciliation
check only. Per-generation rows remain authoritative because other team traffic
can change the account total during the run.

## Vultr reconciliation

The preflight validates the configured hourly price against Vultr's current
plan catalog and reports the account balance and pending-charge baseline without
printing credentials. Each worker later records its exact instance ID and all
lifecycle timestamps. Destroying the instance is mandatory; stopping it does
not stop billing.

Vultr does not expose an account hard-spend cap through this project. The worker
orchestrator must therefore enforce a software budget before provisioning and
destroy workers in a `finally` path. No instance is provisioned by Parts 1 or 2.

## Local commands

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=backend \
  python3 -m dapta.preflight --env .env

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=backend \
  python3 -m unittest discover -s backend/tests -v
```
