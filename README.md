# Dapta

Dapta turns stable, repetitive AI workflows into verified LoRA specialists that
run on an open-weight model with shorter prompts and lower recurring cost.

## About

Teams often resend the same long instructions to frontier models for work they
already understand. Dapta moves that effort upfront: it defines the workflow,
builds and validates training data, trains a lightweight LoRA adapter, and tests
the specialist against frontier models before publishing it.

Existing specialists run through Qwen3-VL on RunPod. Unfamiliar or complex work
can still fall back to frontier models.

## How it works

1. Base44 or ASI:One sends a workflow to the FastAPI controller.
2. JEV routes it to an existing specialist or starts a new build.
3. Kimi and Devin prepare the skill, integration, and training data.
4. RunPod trains the LoRA with QLoRA and evaluates it on a frozen test set.
5. Verified adapters are stored in Hugging Face and loaded for inference.

Sentry traces the UI, agent calls, training, inference, latency, token use,
cost, and operational failures. SQLite and Supabase remain the system of record.

## Stack

- Base44 and React
- FastAPI and SQLite
- JEV, Kimi, Devin, Claude, and GPT
- Qwen3-VL, QLoRA, RunPod, and vLLM
- Hugging Face and Supabase
- Sentry

## Run locally

Create the environment and start the controller:

```bash
cp .env.example .env
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-controller.txt
PYTHONPATH=backend .venv/bin/uvicorn dapta.api:app --reload
```

Start the frontend in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Run the checks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=backend \
  .venv/bin/python -m unittest discover -s backend/tests -v
cd frontend && npm run build && npm test
```

Provider credentials and optional integrations are documented in
`.env.example`. Full training and model serving use `backend/requirements.txt`
on the RunPod worker.

## License

MIT License. Copyright (c) 2026 Aditya Venkat. See [LICENSE](LICENSE).
