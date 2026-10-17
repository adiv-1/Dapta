# Dapta Demo UX Specification

Last updated: October 16, 2026

## Demo promise

Dapta demonstrates that a reusable open-weight LoRA adapter can complete a
repetitive workflow faster and more cheaply than sending the entire Skill to a
frontier model on every request, while preserving measurable accuracy.

The primary demo is an expense-report workflow. The user uploads receipt images
and asks Dapta to create an expense report. Three models process the same input:

1. Qwen3-VL-8B with the selected expense-report LoRA and a short request.
2. Claude Opus 5.5 with the complete `SKILL.md` instructions.
3. GPT-6 Sol with the complete `SKILL.md` instructions.

Claude and GPT are comparison baselines only. They do not create training data,
labels, repair examples, or evaluation truth for the Qwen adapter.

## Information architecture

The product has three primary tabs.

### 1. Chat

Initial state:

- A quiet, centered chat composer is the only primary content.
- The user can attach multiple receipt or document images.
- A short example prompt may be shown inside the composer.

After submission:

- The composer moves to the top of the workspace.
- Three comparison lanes appear beneath it.
- Each lane has a compact model pill: Qwen LoRA, Claude Opus 5.5, and GPT-6 Sol.
- All three lanes receive the same user files and task.
- Qwen uses the selected LoRA and a short request.
- Claude and GPT receive the full Skill instructions.
- Each lane renders its expense report and exposes status, elapsed time, and
  cost.
- A clock remains at the bottom of every lane so latency is obvious during and
  after execution.

### 2. Performance

This is the most important tab in the demo. It opens with **Performance over
time**, followed by horizontally arranged pills for previously completed test
runs. Selecting a pill changes the full comparison below without making another
model call.

Every saved test contains one performance summary calculated directly from its
stored metrics. The summary identifies:

- the fastest model;
- the least expensive model;
- the most accurate model; and
- the best overall result for that test.

The summary is deterministic UI logic, not LLM-generated commentary.

Each selected test must show three dedicated bar charts. Every chart compares
all three models directly:

1. **Time** — total task duration, where lower is better.
2. **Cost** — per-task inference cost, where lower is better.
3. **Accuracy** — output correctness, where higher is better.

Required metrics:

- accuracy;
- schema validity;
- total latency;
- time to first token;
- input and output tokens;
- per-request inference cost; and
- amortized adapter cost, including training and factory costs.

Exactly one accuracy-checking agent evaluates output correctness across all
three model results. Exact totals, currencies, categories, and required fields
are still checked by deterministic code whenever possible. No other agent is
needed to construct the Performance tab: charts, winners, historical trends,
and summaries are derived from stored metrics.

### 3. Skills & Adapters

This tab is the registry of workflows that Dapta already knows.

Each adapter shows:

- skill name and version;
- base model;
- modality;
- readiness state;
- latest accuracy;
- creation time and cost; and
- an action to test the adapter.

An unknown but compilable workflow starts a new adapter build. Its registry card
shows a progress bar, estimated completion time, cloud provider, and current
stage:

1. Understanding Skill
2. Building Dataset
3. Validating Dataset
4. Provisioning Vultr GPU
5. Training LoRA
6. Evaluating
7. Publishing Adapter

The demo should make clear that a newly requested adapter is still training and
will take hours, while adapters trained earlier during the hackathon are ready
for immediate testing.

## Demo sequence

1. Open Chat in its minimal initial state.
2. Attach several receipt images and request an expense report.
3. Run the three model lanes simultaneously.
4. Call attention to the live clocks and completed reports.
5. Open Performance and compare accuracy, latency, tokens, and cost.
6. Open Skills & Adapters and show the ready expense adapter.
7. Submit an unfamiliar workflow and show a new Vultr-backed adapter build enter
   its multi-hour training state.
8. Return to a ready adapter and demonstrate that existing skills remain usable
   while the new adapter trains asynchronously.

## Wireframe

The interactive static reference is
[`docs/wireframes/dapta-demo-wireframe.html`](wireframes/dapta-demo-wireframe.html).
