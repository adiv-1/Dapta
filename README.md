# Dapta

Dapta compiles stable, repeated AI workflows into evaluated LoRA adapters for
`Qwen/Qwen3-VL-8B-Instruct`.

## Repository

- `backend/` — factory agents, dataset validation, and Qwen3-VL training
- `frontend/` — demo application work
- `docs/` — durable product, architecture, and UX decisions
- `research/adapter-assets/` — remote dataset catalog and public skill references

Read `docs/HACKATHON_MASTER_PLAN.md` first. It is the current source of truth;
`docs/PROJECT_CONTEXT.md` and `docs/ARCHITECTURE.md` preserve earlier research
and implementation detail.

Share `docs/BUILD_PLAN.md` with teammates for the ordered implementation
checklist and current configuration requirements.

Run `python3 -m dapta.preflight` with `PYTHONPATH=backend` before creating cloud
resources. Cost attribution and reconciliation rules are documented in
`docs/COST_AND_USAGE_LEDGER.md`.

The persisted state machine, HTTP routes, and OpenRouter agent boundary are
documented in `docs/BACKEND_CONTROL_PLANE.md`.

No GPU is provisioned by this repository yet. Training starts only after a
dataset passes validation and a short cloud smoke test is ready.
