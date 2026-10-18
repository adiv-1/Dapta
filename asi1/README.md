# DAPTA Adapter Factory for ASI:One

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

This companion registers one public **DAPTA Adapter Factory** agent on Agentverse. It uses
the Agent Chat Protocol to expose DAPTA's existing adapter factory and private Qwen LoRA
worker without changing or publicly exposing either service.

## Capabilities

- list registered DAPTA adapters;
- route a stable workflow to an existing adapter or a new build;
- require explicit confirmation before starting a paid adapter build;
- report real factory job state; and
- execute a ready Qwen LoRA adapter on privacy-safe input.

Kimi, OpenRouter, JEV, Devin, RunPod training, evaluation, Hugging Face publishing, the
SQLite ledger and Sentry remain internal DAPTA services. They are not separate Agentverse
agents.

## Local verification

The orchestration tests use fake services and do not spend provider credits:

```bash
PYTHONPATH=. python -m unittest discover -s asi1/tests -v
```

## Run beside DAPTA

Use a separate environment so the existing backend dependency set stays unchanged:

```bash
python3.11 -m venv asi1/.venv
asi1/.venv/bin/pip install -r asi1/requirements.txt
cp asi1/.env.example asi1/.env
set -a
source asi1/.env
set +a
asi1/.venv/bin/python -m asi1.agent
```

Generate one unique seed, keep it private and never change it after registration. Connect
the Inspector URL printed at startup to Agentverse using **Mailbox**, then configure the
name, handle and Innovation Lab category. Keep this process running during the demo.
When it runs on RunPod, forward port `8003` over SSH before opening the Inspector URL on
your computer; registration through a localhost Inspector is not directly reachable from
the remote pod. Confirm the profile publishes `AgentChatProtocol` and is marked active.

`DAPTA_ASI1_ENABLE_FACTORY_BUILDS` defaults to `false`, so the public agent can execute
ready adapters without allowing arbitrary users to spend model or GPU credits. Set it to
`true` only for a supervised build window, then return it to `false`. DAPTA's existing
factory budget gates still apply.

## Safe demo

Use synthetic document text only. A reliable sequence is:

1. `list adapters`
2. `run adapter <ready-adapter-id>`
3. `Document: <synthetic input>`
4. During a supervised build window: `create a LoRA adapter that ...`
5. `confirm build` only after accepting the paid compute/model cost
6. `status`

The primary qualifying workflow should execute an already-ready adapter completely inside
ASI:One. A new build is a secondary workflow unless it genuinely reaches `ready` during the
same conversation.
