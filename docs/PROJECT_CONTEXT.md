# Dapta / MoLE Project Context

Last updated: October 16, 2026

## Purpose of this document

This is the durable project brain for the LA Hacks repository. It records the
product idea, technical thesis, proof already completed, known limitations,
planned system, demo story, and unresolved decisions. Future work should read
this before changing architecture or making claims.

This document records the current implementation boundary as well as the product
decisions that future work must preserve.

## Repository situation

There are currently two separate repositories:

1. `skill2lora` is Pavan Srivathsa's private proof-of-concept repository. It
   contains a working, tested NDA-triage experiment on Qwen3-1.7B.
2. `Dapta` is Aditya Venkat's private GitHub repository and the intended
   home for the hackathon project. It started essentially empty.

Nothing from `skill2lora` should be moved blindly. First decide what the final
hackathon scope and architecture are, which POC components remain valid, and
what must be generalized or replaced. Preserve provenance and experiment
integrity during any later migration.

## Naming and the central idea

The product name is **Dapta**.

The underlying technical concept has also been called **MoLE (Mixture of LoRA
Experts)** and a **Neural Skill Compiler**. These names describe the mechanism;
Dapta is the product and pitch.

Dapta turns suitable repetitive AI workflows into small, verified,
hot-swappable LoRA adapters. Instead of sending the same long Skill, SOP,
policy, examples, and output rules with every request, the system spends compute
up front to specialize an open-weight base model and reuses that specialization
across later requests.

The memorable pitch is:

> Pay for reasoning once.

This is shorthand for amortizing the cost of specialization. It is **not** a
literal claim that later calls contain no reasoning, inference cost, prompt,
validation, tools, or fallback work.

The opening question for the pitch is:

> What if routine AI work could become faster, cheaper, and more accurate?

## The problem

Organizations repeatedly process documents and structured workflows such as:

- NDAs and other contracts
- invoices and receipts
- resumes and job-application forms
- underwriting materials
- internal policy and compliance workflows

Today a general model may repeatedly receive a large instruction package for
every task. This creates two connected problems:

1. Repeated instructions consume input tokens, context space, prefill compute,
   and potentially time and money on every request.
2. A broad generalist may still apply an organization's exact rules and output
   contract inconsistently.

Dapta's thesis is that a stable and frequently reused decision procedure can be
compiled into a lightweight specialist once, verified, and then served cheaply
on top of a shared base model. General models remain available for unfamiliar,
dynamic, or complex work.

## What makes a workflow suitable

A workflow is a promising specialization candidate when it is:

- repeated often enough to amortize build and evaluation cost;
- governed by stable rules rather than rapidly changing facts;
- learnable from examples;
- objectively or reliably evaluable;
- narrow enough to route with high confidence; and
- compatible with the capabilities of the chosen base model.

Dapta should not recommend compilation for a one-off request, live-information
question such as weather, poorly specified behavior, or a task whose essential
capability is an external tool.

LoRA can encode behavior such as classification rules, output structure,
analysis patterns, escalation rules, and domain terminology. It cannot replace
database access, web search, current prices, external APIs, file operations, or
other non-neural dependencies. Those still require tools and orchestration.

## Intended product flow

The envisioned orchestrator is **JEV**.

At build time:

1. JEV detects a repetitive, stable, evaluable workflow and asks the user for
   permission to create a specialist.
2. **Skill Architect** converts the Skill, SOP, examples, and corrections into a
   structured behavioral specification: explicit rules, exceptions, required
   fields, schema, and escalation behavior.
3. **Data Builder** creates training cases, edge cases, development cases, and a
   frozen test set. Where possible, it generates structured scenarios and uses
   code as the ground-truth oracle rather than letting one model invent both a
   question and its answer.
4. **Adapter Builder** trains and tunes a LoRA/QLoRA adapter for an open-weight
   base model.
5. **Independent Evaluator** measures deterministic correctness and schema
   compliance, plus model-judged qualities only where code cannot evaluate them.
   Development failures may produce targeted training data and another training
   round. The final held-out test remains untouched.
6. A human approves successful adapters for a versioned registry.

At runtime:

1. JEV evaluates whether a request fits a registered specialist.
2. A confident match uses the shared base model, the selected adapter, and a
   short prompt.
3. An uncertain or unfamiliar request falls back to the general model with the
   full Skill and any required tools, or to human review.
4. Runtime telemetry records quality, tokens, timing, routing, fallback, and
   cost inputs for later evaluation and retraining decisions.

Long-term, one multimodal base model could host multiple compatible adapters and
hot-swap among NDA triage, receipt-image extraction, resume form filling,
invoice checking, and other repeatable workflows.

## What the existing POC actually proves

The `skill2lora` repository implements a single-skill viability experiment, not
the full Dapta system.

It uses:

- student model: `Qwen/Qwen3-1.7B`;
- one text-only task: synthetic NDA triage;
- a long `SKILL.md` containing nine screening rules and a strict JSON contract;
- deterministic truth from `oracle/nda_rules.py`;
- generated Family A train/dev/test cases and a differently worded Family B OOD
  set;
- FP16 LoRA training through Transformers, PEFT, and TRL;
- an iterative DEV repair loop capped at three rounds;
- frozen TEST/OOD data protected by hashes and leak guards; and
- four evaluation conditions: base, full Skill, cached full Skill, and LoRA with
  a short prompt.

The configured dataset sizes are 600 TRAIN, 100 DEV, 150 TEST, and 50 OOD. The
short prompt is `Triage this NDA. Respond with JSON only.` Thinking is disabled
for every chat-template call.

Reported held-out TEST results (`n=150`) are:

- classification accuracy: LoRA **84.7%**, full Skill **38.7%**, base **0%**;
- criteria accuracy: LoRA **95.3%**, full Skill **72.7%**;
- mean input tokens: LoRA **297**, full Skill **2,230**, about **87% fewer**; and
- valid JSON: LoRA **100%**, full Skill **100%**.

These numbers passed the POC's four predefined GO gates. The gates tested LoRA
classification relative to the full Skill, criteria accuracy relative to the
full Skill, at least 70% input-token reduction, and at least a 15-point
classification improvement over base.

Reported Family B OOD results (`n=50`) are materially weaker:

- classification accuracy: LoRA **50%**, full Skill **46%**; and
- criteria accuracy: LoRA **84.7%**, full Skill **64.7%**.

The OOD set was not part of the GO rule. It shows that generalization is still a
major risk even though criterion-level performance remained better than the
prompted baseline.

## What has not been proven

Do not overstate the POC. It has **not** established:

- that the results transfer from Qwen3-1.7B to the final 8B or multimodal model;
- robust generalization across genuinely different documents and task families;
- a defensible time-to-first-token or total-latency improvement;
- a dollar-cost advantage after teacher calls, training, hosting, routing,
  validation, fallback, monitoring, and retraining;
- production-quality automatic Skill parsing or scenario generation;
- multi-adapter routing, hot-swapping, composition, or safe fallback;
- the proposed four-agent product workflow;
- multimodal receipt, resume, invoice, or underwriting performance;
- learning directly from stored reasoning traces; or
- security against prompt leakage or prompt injection.

The reported cached-Skill condition matched uncached accuracy and did not show a
credible latency improvement. Reported TTFT measurements appeared too small to
trust; approximately 7–11 seconds median total latency was the more meaningful
clock. Latency needs a controlled rerun using streaming measurements and the
actual serving stack.

An 87% input-token reduction must never be presented as an 87% dollar reduction.

## Experiment principles that must survive migration

- Prefer deterministic oracles for exact rules and labels.
- Never let one model freely invent both evaluation inputs and their supposedly
  correct answers when code can generate truth.
- Separate TRAIN, DEV, TEST, and OOD roles.
- Freeze TEST before iterative optimization and guard it against contamination.
- Use DEV failures for targeted repair; do not tune repeatedly on TEST.
- Compare the same cases and generation settings across all methods.
- Include the strongest relevant baselines: base only, full Skill, cached full
  Skill, and short prompt plus LoRA.
- Measure classification, criterion/rule accuracy, schema validity, prompt and
  completion tokens, streaming TTFT, total latency, throughput, and errors.
- Use code-based evaluation for objective requirements and a genuinely
  independent model family only for subjective qualities that code cannot score.
- Record hardware, model version, adapter version, prompt version, dataset hash,
  and serving configuration so results are reproducible.
- Report OOD separately and candidly.

## Model and infrastructure direction

The hackathon's adapter factory must remain open-weight end to end. Closed
frontier models must not author training examples, labels, rules, repair data,
or evaluation truth. They may appear only as untouched runtime baselines after
an adapter has already been trained.

Locked direction:

- use `Qwen/Qwen3-VL-8B-Instruct` as the shared multimodal student/base;
- use a strong open-weight model from a different family as the Skill Architect
  and Data Builder, initially `moonshotai/kimi-k3` through OpenRouter;
- optionally use `openai/gpt-oss-120b` as a second open-weight critic or
  self-hosted factory model;
- use deterministic code as ground truth whenever rules are expressible;
- reserve Claude/OpenAI proprietary frontier models for post-training benchmark
  comparisons only, never for producing supervision;
- use Transformers/PEFT/TRL for training and vLLM for runtime LoRA loading; and
- initially adapt the Qwen language backbone while freezing the vision tower,
  because multimodal vision-tower LoRA serving is more experimental.

The open-weight factory's agents are logical, role-bound sessions against one or
two shared model endpoints. They are not separate model deployments and do not
each occupy a GPU. Data/spec generation happens before the 6--8 hour LoRA job;
the factory endpoint can be remote through OpenRouter or temporarily self-hosted
and then unloaded before training.

## Hackathon scope under consideration

The strongest focused version is a system for repetitive, exactly verifiable
tasks on one open-weight base model. Candidate demonstrations are:

- NDA triage, carried over from the POC;
- receipt screenshot to CSV or spreadsheet; and
- resume to job-application form filling.

Possible hackathon capabilities include adapter training, a registry,
multi-LoRA serving, conservative routing, fallback to full-Skill general
execution, evaluation, and a side-by-side UI.

Earlier notes conflict on scope: the research plan says to prove one adapter
before building routing, while the later hackathon brief proposes routing and
multiple adapters for the demo. This must be resolved during planning based on
time, team size, hardware, and what judges can verify. It is not yet a decision.

Likewise, one brief includes an automatic compile-debug-recompile loop while a
later brief says "automatic learning from mistakes" is out of scope. A bounded
offline DEV repair loop may be different from continuous production learning,
but the distinction and demo scope need an explicit decision.

## Confirmed event, cloud, and sponsor constraints

As of October 16, 2026:

- The team has **$250 in Vultr promotional credit** available and is willing to
  exhaust it on useful training. Cost must still be measured because cheaper
  inference is a core product claim, not because $250 is a hard project cap.
- The Fetch.ai LA Hacks hackpack lists hacking from **October 17 at 12:00 PDT**
  through **October 18 at 12:00 PDT**. Planning and account/quota setup can be
  done beforehand, but implementation should not begin early unless organizers
  explicitly approve it.
- The published LA Hacks rules say work should be performed during the event and
  that reused pre-existing work must be open source and disclosed. The private
  `skill2lora` POC therefore cannot simply be copied without confirming its
  license, Pavan's permission, public availability, and what judges will count
  as new work. The public rules pages contain some inconsistent legacy dates, so
  the team should confirm the applicable rule text with organizers.
- The Fetch.ai prize requires at least one Agentverse-registered agent, Agent
  Chat Protocol support, ASI:One discovery and direct use, meaningful tool or
  multi-agent execution, and completion of the primary workflow entirely inside
  an ASI:One conversation.
- Fetch.ai submission requires both the normal Devpost submission and separate
  registration through the LA Hacks ASI:One Submission Agent.
- The general event submission requires a public source repository, project
  description, complete Devpost entry, and a 2–3 minute demo video.

The team is willing to use the available GPU compute aggressively. Training a
Qwen3-VL 8B LoRA is expected to take roughly 6--8 hours. Start with two parallel
single-GPU adapter jobs after a short smoke test; expand farther only when more
complete datasets and eval suites are ready.

Provisional cloud shape:

- a cheap CPU control-plane VM for the API, durable job state, Agentverse bridge,
  and registry metadata;
- ephemeral GPU workers capable of two parallel adapter-training jobs;
- object storage for datasets, adapters, reports, hashes, and recordings so the
  GPU VM can be destroyed when idle;
- ASI:One and a custom web interface as two clients of the same backend; and
- a strict budget cap with GPU destruction, not merely shutdown, when unused,
  because reserved stopped Vultr instances continue billing.

For the hackathon, the realistic on-demand promise is one constrained compiler
contract: stable structured classification/extraction Skills with explicit
schemas and programmatically verifiable outputs. Arbitrary unscorable workflows
must be rejected or routed to the general fallback.

## Demo concept

The demo should make the mechanism visible rather than merely describe it:

1. Show a company workflow being recognized as repetitive and approved for
   specialization.
2. Show Skill Architect, Data Builder, Adapter Builder, and Independent Evaluator
   working, then visually collapse their output into one small adapter.
3. Reveal several adapters around a shared model, if multi-adapter scope is
   actually completed.
4. On the same unseen input, race the base model with the full Skill against the
   selected Dapta adapter with a short prompt.
5. Show actual outputs plus correctness, input/output tokens, streaming TTFT,
   total latency, and a defensible cost estimate.
6. Submit an unfamiliar input and show conservative fallback rather than a forced
   specialist route.
7. Present full held-out aggregate results separately from the live example so
   the live race is not mistaken for the benchmark.
8. Keep a recorded backup.

The presentation should use sparse text, distinct visual ideas, a cost curve,
and an obvious transformation from repeated instructions to a reusable adapter.
Clearly label measured results, proposed features, and illustrative economics.

## Business-case framework

Compare cumulative cost, not merely prompt length:

- existing approach = task count × cost per general/prompted task;
- Dapta = build and evaluation cost + task count × specialized runtime cost.

Dapta's calculation must include teacher/data-generation calls, training,
hosting, routing, validation, fallback, monitoring, and retraining. Show the
break-even volume. Compare against prompt caching and a smaller prompted model,
not only against an expensive frontier API.

For self-hosted open models, distinguish logical token/compute usage from actual
infrastructure cost. A useful per-request estimate is GPU hourly price multiplied
by request duration divided by 3,600, but utilization and deployment assumptions
must be stated.

## Claims and positioning

Safe headline:

> Dapta compiles stable, repetitive AI workflows into verified LoRA specialists
> that can use much shorter prompts on a shared open-weight model, with a general
> path for unfamiliar work.

Strong but honest POC claim:

> In one synthetic NDA-triage experiment on Qwen3-1.7B, the LoRA improved
> held-out classification and criteria accuracy over the full-Skill prompt while
> using 87% fewer input tokens. OOD classification remained weak, and latency and
> dollar savings are not yet proven.

Avoid saying:

- inference is free;
- the adapter performs "no thinking" as a technical fact;
- all Skills can be compiled;
- 87% fewer tokens means 87% cheaper;
- latency is already proven;
- the 1.7B text results apply to an 8B multimodal system;
- prompts become impossible to leak; or
- routing and safe fallback already exist.

Prior work such as LatentSkill, Skill-to-LoRA, and trace-compilation research is
part of the landscape described by the team. The novelty claim should therefore
focus on building an understandable working system with qualification, routing,
fallback, rigorous evaluation, real workflows, and measured economics—not on
claiming the basic text-to-LoRA idea has no precedent. Exact citations and claims
need verification before the pitch.

## Major planning decisions still open

Before implementation or migration, decide:

1. What is the exact hackathon success criterion: one undeniable compiler proof,
   or a broader multi-adapter product demo?
2. Which specific multimodal base model satisfies training, serving, licensing,
   and available-GPU constraints?
3. Are pre-trained adapters allowed under LA Hacks rules, and what must be built
   during the event?
4. Which public benchmark can credibly test the core thesis?
5. Which first-party workflows will be included, and how will each obtain exact
   ground truth and OOD cases?
6. Is JEV only a router, or also the workflow-recognition and user-consent layer?
7. What routing method and confidence threshold trigger fallback?
8. Is the build pipeline truly agentic/live, a deterministic orchestrated
   workflow, or a visualized pre-run process?
9. Is the DEV repair loop in scope, and how is it distinguished from continuous
   online learning?
10. What is the independent evaluator, and how is family independence ensured?
11. What hardware is available for training and serving, and what can run
    concurrently during the demo?
12. What evidence is required before the team claims faster or cheaper?
13. What exact subset of `skill2lora` should be preserved, generalized, or
    discarded?
14. What security/privacy claims, if any, can be demonstrated rather than
    asserted?
15. Who owns training, serving/router, benchmarks/data, UI, and pitch/submission?

## Source-of-truth hierarchy

When sources disagree, use this order until the team makes a new explicit
decision:

1. Newly verified experiment artifacts and written team decisions.
2. This context document, updated to reflect those decisions.
3. The checked-in `skill2lora` code and configuration for what the POC actually
   does.
4. Attached product and research briefs for intent and candidate direction.
5. Pitch shorthand, which must not override technical evidence.

Any future material change should update this document with the decision, date,
rationale, and effect on scope or claims.
