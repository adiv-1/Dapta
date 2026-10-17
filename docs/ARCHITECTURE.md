# Dapta architecture

Last updated: October 16, 2026

## Repository shape

```text
Dapta/
├── backend/
│   ├── dapta/          # agents, dataset contract, Qwen3-VL trainer
│   ├── tests/          # fast tests with no model download
│   └── requirements.txt
├── frontend/           # demo application
├── docs/               # context, architecture, UX, wireframe
└── README.md
```

Generated datasets, checkpoints, adapters, and model caches are artifacts, not
source code. They stay outside the committed package and are published to the
private Hugging Face adapter repository when approved.

## Adapter factory

The factory has four logical agents. They are role-bound sessions, not four
separate model deployments.

1. **Skill Architect — Kimi K3.** Converts a Skill and user examples into a
   structured specification: inputs, output schema, rules, exclusions, and
   deterministic checks.
2. **Data Builder — Kimi K3.** Produces TRAIN, DEV, TEST, and OOD candidates.
   It must call deterministic generators or validators for totals, currencies,
   dates, categories, and schema fields. It cannot approve its own TEST set.
3. **Adapter Builder — deterministic worker.** Validates split isolation,
   launches Qwen3-VL QLoRA, records metadata, and uploads the adapter.
4. **Independent Evaluator — code first, gpt-oss-120b only when needed.** Runs
   exact checks before any model judgment. TEST remains frozen; only DEV
   failures can create repair work.

JEV sits outside the factory. It decides whether an existing adapter applies,
whether a new stable workflow is compilable, or whether to use the general
fallback.

```text
request
  -> JEV route
  -> existing adapter: run Qwen + LoRA
  -> new compilable skill: Skill Architect -> Data Builder -> validation
                           -> Adapter Builder -> Independent Evaluator
                           -> registry
  -> unfamiliar or unsafe: full-Skill/general fallback
```

## Dataset contract

Each JSONL example contains `id`, one or more relative `images`, a `prompt`, and
the expected `completion`. The Data Builder may propose examples, but code owns
split validation and the TEST hash. Duplicate IDs, identical prompt/image cases,
missing images, unsafe paths, and a changed TEST hash stop training.

The first multimodal skill is expense-report generation from receipt images.
Its exact schema and oracle are deliberately not handwritten yet; they are the
Data Builder agent's first job after the Skill is finalized.

## Qwen training design

- Base: `Qwen/Qwen3-VL-8B-Instruct`
- Method: single-GPU 4-bit QLoRA, one isolated job per GPU
- Trainable targets: language attention projections only
- Frozen components: base weights and vision tower
- Precision: NF4 storage with BF16 compute
- Dataset: TRL conversational prompt/completion with one or more images
- Safety: no sequence truncation until image-token boundaries are measured
- Output: PEFT adapter, processor config, dataset hash, hyperparameters, metrics

Each training process receives one GPU through `CUDA_VISIBLE_DEVICES`. We do
not use `device_map="auto"` across the four A40s because Dapta wants independent
parallel adapter jobs, not one model silently sharded across every GPU.

## What survives from `skill2lora`

Keep the POC's experimental discipline:

- deterministic truth whenever possible;
- TRAIN/DEV/TEST/OOD separation;
- frozen TEST hashes and leak guards;
- base, full-Skill, cached-Skill, and LoRA conditions;
- bounded repair from DEV failures; and
- candid OOD reporting.

Replace its implementation assumptions:

- `AutoModelForCausalLM` and `AutoTokenizer` are text-only;
- its rows contain only `nda_text`, so receipt pixels never reach the model;
- `device_map="auto"` conflicts with isolated parallel jobs;
- it trains FP16 LoRA rather than the planned 4-bit QLoRA;
- metrics and paths are hard-coded to NDA triage; and
- it has no Hugging Face publishing or cloud lifecycle guard.

The original NDA experiment remains evidence and provenance in its own repo.
The hackathon repository uses a small, independently adapted multimodal core
instead of copying the POC wholesale.

## Build boundary for tonight

Completed here: POC audit, multimodal dataset contract, agent roles, Qwen3-VL
QLoRA entry point, Hugging Face publishing path, and fast dataset tests.

Not started intentionally: production dataset generation, GPU provisioning,
training, evaluation runs, runtime serving, JEV integration, and frontend code.
These begin after the expense-report Skill contract exists and the cloud smoke
test can run immediately after provisioning.
