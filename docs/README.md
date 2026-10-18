# Dapta documentation index

Start here if you are a new engineer or coding agent:

1. [`AGENT_HANDOFF.md`](AGENT_HANDOFF.md) — authoritative current state, live
   services, ownership boundaries, and next actions.
2. [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md) — safe factory operation,
   failure diagnosis, recovery, and publication procedures.
3. [`ARCHITECTURE.md`](ARCHITECTURE.md) — component and data-flow design.
4. [`COST_AND_USAGE_LEDGER.md`](COST_AND_USAGE_LEDGER.md) — accounting rules.

`BUILD_PLAN.md`, `HACKATHON_MASTER_PLAN.md`, `PROJECT_CONTEXT.md`, and the UX
documents contain design history. Where they conflict with the handoff or
runbook, the handoff and current code win. Vultr and Hostinger references in
historical planning documents are not the active deployment path; production
currently runs on a persistent RunPod A40.

Never put API keys, access tokens, SSH private keys, `.env`, `runtime.env`, user
documents, or chain-of-thought in this directory.
