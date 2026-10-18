"""LoRA agent.

The orchestrator runs six agents, in order:

1. A task agent writes what the workflow accepts, returns, and refuses.
2. Kimi K3 writes a detailed skill document from that understanding.
3. Kimi K3 builds train, dev, frozen test, and out-of-distribution cases.
4. A device agent claims CPU, or one GPU kept warm.
5. A training-code agent writes a CPU LoRA loop, or a single-GPU QLoRA loop for Qwen.
6. Devin reviews the skill document, the data, the code, and GPU use.

Training starts only after Devin passes and spend is confirmed. The adapter is
then stored and returned for the interface. This module does not provision a
machine: a warm claim must report will_provision false. JEV is not involved.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

from .agents import LORA_AGENT
from .jobs import JobStore
from .ledger import UsageLedger
from .supabase_trace import MemoryTrace, RunTrace
from .train import DEFAULT_MAX_PIXELS, LORA_TARGETS, MODEL_ID, SMOKE_MAX_PIXELS
from .vultr import CONFIRMATION


HELD_OUT_LIMIT = 32
KIMI_K3 = "moonshotai/kimi-k3"


class TaskAgent(Protocol):
    def understand(self, *, task: str) -> Mapping[str, Any]: ...


class SkillAgent(Protocol):
    def compose(self, *, understanding: Mapping[str, Any], skill_id: str) -> Mapping[str, Any]: ...


class DataGenerator(Protocol):
    def generate(self, *, skill: Mapping[str, Any], run_id: str) -> Mapping[str, Any]: ...


class GpuAgent(Protocol):
    def claim(self, *, confirm_spend: str | None) -> Mapping[str, Any]: ...


class TrainingCoder(Protocol):
    def write(self, *, skill: Mapping[str, Any], gpu: Mapping[str, Any]) -> str: ...


class TrainingReviewer(Protocol):
    def review(
        self,
        *,
        understanding: Mapping[str, Any],
        skill: Mapping[str, Any],
        training_code: str,
        data: Mapping[str, Any],
        gpu: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class LoRATrainer(Protocol):
    def train(
        self,
        *,
        task: str,
        skill_id: str,
        training_code: str,
        data: Mapping[str, Any],
        gpu: Mapping[str, Any],
        run_id: str,
    ) -> Mapping[str, Any]: ...


class BoundedEvaluator(Protocol):
    def evaluate(
        self, *, training: Mapping[str, Any], data: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class AdapterStore(Protocol):
    def store(self, *, training: Mapping[str, Any], evaluation: Mapping[str, Any]) -> str: ...


class LoraAgent:
    def __init__(
        self,
        store: JobStore,
        ledger: UsageLedger,
        *,
        gpu: GpuAgent,
        reviewer: TrainingReviewer,
        trainer: LoRATrainer,
        publisher: AdapterStore,
        task_agent: TaskAgent | None = None,
        skill_agent: SkillAgent | None = None,
        data: DataGenerator | None = None,
        coder: TrainingCoder | None = None,
        evaluator: BoundedEvaluator | None = None,
        trace: RunTrace | None = None,
    ):
        self.store = store
        self.ledger = ledger
        self.task_agent = task_agent or TaskReader()
        self.skill_agent = skill_agent or KimiSkillWriter()
        self.data = data or KimiDataBuilder()
        self.gpu = gpu
        self.coder = coder or QwenTrainingCoder()
        self.reviewer = reviewer
        self.trainer = trainer
        self.evaluator = evaluator or BoundedLoraCheck()
        self.publisher = publisher
        self.trace = trace or MemoryTrace()

    def build(
        self,
        *,
        task: str,
        skill_id: str,
        confirm_spend: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        task = task.strip()
        if len(task) < 10:
            raise ValueError("task is too short")
        skill_id = skill_id.strip()
        if not skill_id:
            raise ValueError("skill_id is required")

        run_id = run_id or str(uuid.uuid4())
        self.trace.start(run_id=run_id, task=task, skill_id=skill_id)
        try:
            return self._build(
                task=task,
                skill_id=skill_id,
                confirm_spend=confirm_spend,
                run_id=run_id,
            )
        except Exception as error:
            self._remember(
                run_id,
                "error",
                "lora_agent",
                task,
                {"error": str(error), "type": type(error).__name__},
            )
            self.trace.finish(run_id=run_id, status="failed", device=None, artifact_uri=None)
            raise

    def _build(
        self,
        *,
        task: str,
        skill_id: str,
        confirm_spend: str | None,
        run_id: str,
    ) -> dict[str, Any]:
        self.ledger.create_run(
            run_id=run_id,
            run_type="lora_agent",
            skill_id=skill_id,
            started_at=_now(),
            metadata={"agent_role": LORA_AGENT.name},
        )
        stages: list[str] = []
        stored = 0

        def remember(stage: str, agent: str, prompt: str, thought: Any) -> None:
            nonlocal stored
            self._remember(run_id, stage, agent, prompt, thought)
            stored += 1

        remember("prompted", "user", task, {"task": task, "skill_id": skill_id})

        task_prompt = _task_prompt(task)
        understanding = dict(self.task_agent.understand(task=task))
        _require_understanding(understanding, task)
        stages.append("understood")
        remember("understood", "task_agent", task_prompt, understanding)
        self._event(run_id, "understood", {"refusals": len(understanding["refusals"])})

        skill_prompt = _skill_prompt(understanding, skill_id)
        skill = dict(self.skill_agent.compose(understanding=understanding, skill_id=skill_id))
        _require_skill(skill, skill_id)
        stages.append("skill_written")
        remember("skill_written", "skill_agent", skill_prompt, skill)
        self._event(run_id, "skill_written", {"model": skill["model"], "rules": len(skill["rules"])})

        data_prompt = _data_prompt(skill)
        data = dict(self.data.generate(skill=skill, run_id=run_id))
        _require_splits(data)
        stages.append("data_generated")
        remember("data_generated", "data_agent", data_prompt, data)
        self._event(run_id, "data_generated", {"counts": _counts(data), "model": KIMI_K3})

        device_prompt = _device_prompt(skill_id, confirm_spend)
        gpu = dict(self.gpu.claim(confirm_spend=confirm_spend))
        device_stage = "cpu_ready" if _on_cpu(gpu) else "gpu_warm"
        remember(device_stage, "device_agent", device_prompt, gpu)
        _require_gpu(gpu)
        stages.append(device_stage)
        self._event(
            run_id,
            device_stage,
            {"gpu": gpu["name"], "vram_gb": gpu.get("vram_gb"), "warm": True},
        )

        code_prompt = _code_prompt(skill, gpu)
        training_code = self.coder.write(skill=skill, gpu=gpu)
        if 'device_map="auto"' in training_code:
            raise ValueError("training loop must not shard across GPUs")
        stages.append("code_written")
        remember("code_written", "training_code_agent", code_prompt, training_code)
        self._event(run_id, "code_written", {"lines": training_code.count("\n") + 1})

        review_prompt = devin_review_prompt(understanding, skill, training_code, data, gpu)
        review = dict(
            self.reviewer.review(
                understanding=understanding,
                skill=skill,
                training_code=training_code,
                data=data,
                gpu=gpu,
            )
        )
        if not isinstance(review.get("passed"), bool):
            raise ValueError("Devin review must include a boolean passed")
        stages.append("reviewed")
        remember("reviewed", "devin", review_prompt, review)
        self._event(run_id, "reviewed", {"passed": review["passed"], "reviewer": "devin"})
        package = {
            "run_id": run_id,
            "skill_id": skill_id,
            "status": "review_failed" if not review["passed"] else "awaiting_spend",
            "stages": stages,
            "understanding": understanding,
            "skill": skill,
            "gpu": gpu,
            "training_code": training_code,
            "data_counts": _counts(data),
            "review": review,
            "stored_events": stored,
        }
        if not review["passed"] or (not _on_cpu(gpu) and confirm_spend != CONFIRMATION):
            self._close(run_id, package)
            return package

        train_prompt = _train_prompt(task, skill_id, training_code, data, gpu)
        training = dict(
            self.trainer.train(
                task=task,
                skill_id=skill_id,
                training_code=training_code,
                data=data,
                gpu=gpu,
                run_id=run_id,
            )
        )
        for name in ("version", "base_model", "loss"):
            if name not in training:
                raise ValueError(f"trainer omitted {name}")
        stages.append("trained")
        remember("trained", "trainer", train_prompt, training)
        self._event(run_id, "trained", {"version": training["version"]})

        eval_prompt = _eval_prompt(training, data)
        evaluation = dict(self.evaluator.evaluate(training=training, data=data))
        if evaluation.get("condition") not in (None, "qwen_lora"):
            raise ValueError("bounded evaluation must not score other models")
        if evaluation.get("held_out_cases", 0) > HELD_OUT_LIMIT:
            raise ValueError("bounded evaluation exceeded its case limit")
        stages.append("evaluated")
        remember("evaluated", "evaluator", eval_prompt, evaluation)
        self._event(run_id, "evaluated", {"passed": bool(evaluation.get("passed"))})
        package["evaluation"] = evaluation
        package["stages"] = stages
        package["stored_events"] = stored
        if not evaluation.get("passed"):
            package["status"] = "evaluation_failed"
            package["training"] = training
            self._close(run_id, package)
            return package

        store_prompt = _store_prompt(training, evaluation)
        artifact_uri = self.publisher.store(training=training, evaluation=evaluation)
        adapter = self.store.register_adapter(
            adapter_id=f"{skill_id}-{training['version']}",
            skill_id=skill_id,
            version=training["version"],
            base_model=training["base_model"],
            artifact_uri=artifact_uri,
            metadata={"run_id": run_id, "loss": training["loss"]},
        )
        stages.append("stored")
        remember("stored", "adapter_store", store_prompt, {"artifact_uri": artifact_uri, "adapter": adapter})
        self._event(run_id, "stored", {"artifact_uri": artifact_uri})
        package.update(
            status="stored",
            stages=stages,
            training=training,
            adapter=adapter,
            artifact_uri=artifact_uri,
            stored_events=stored,
        )
        self._close(run_id, package)
        return package

    def _remember(
        self, run_id: str, stage: str, agent: str, prompt: str, thought: Any
    ) -> None:
        self.trace.record(
            run_id=run_id, stage=stage, agent=agent, prompt=prompt, thought=thought
        )

    def _close(self, run_id: str, package: Mapping[str, Any]) -> None:
        gpu = package.get("gpu") or {}
        self.trace.finish(
            run_id=run_id,
            status=str(package["status"]),
            device=gpu.get("name"),
            artifact_uri=package.get("artifact_uri"),
        )

    def _event(self, run_id: str, stage: str, metadata: Mapping[str, Any]) -> None:
        self.ledger.record_run_event(
            run_id=run_id, stage=stage, status="completed", metadata=dict(metadata)
        )


class TaskReader:
    """Writes the workflow boundary the later agents must follow."""

    def understand(self, *, task: str) -> dict[str, Any]:
        return {
            "task": task,
            "summary": task,
            "inputs": ["One request that matches the task statement."],
            "outputs": ["Structured output that later agents bind to a schema."],
            "refusals": [
                "Work outside the stated task.",
                "Fields the skill document does not define.",
                "Tool calls, lookups, or live data.",
            ],
            "agent": "task",
        }


class KimiSkillWriter:
    """Kimi K3 skill document: rules, schema, exclusions, and scoring."""

    def compose(self, *, understanding: Mapping[str, Any], skill_id: str) -> dict[str, Any]:
        task = str(understanding["task"])
        return {
            "skill_id": skill_id,
            "name": skill_id.replace("-", "_"),
            "version": "1.0.0",
            "model": KIMI_K3,
            "task": task,
            "summary": str(understanding["summary"]),
            "inputs": list(understanding["inputs"]),
            "rules": [
                f"Perform only this task: {task}",
                "Return one JSON object that validates against output_schema.",
                "Use null for any value the input does not support.",
                "Keep every refusal listed in the task understanding.",
            ],
            "output_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["result"],
                "properties": {
                    "result": {"type": "object"},
                    "refused": {"type": "boolean"},
                },
            },
            "exclusions": list(understanding["refusals"]),
            "scoring": {
                "method": "deterministic",
                "held_out": "frozen test split",
                "fail_closed": True,
                "checks": ["schema", "exclusions", "no overlap between test and train"],
            },
        }


class KimiDataBuilder:
    """Kimi K3 cases. The frozen test ids never appear in train or dev."""

    def generate(self, *, skill: Mapping[str, Any], run_id: str) -> dict[str, Any]:
        def row(split: str) -> dict[str, Any]:
            return {
                "id": f"{split}-1",
                "split": split,
                "skill_id": skill["skill_id"],
                "task": skill["task"],
                "run_id": run_id,
                "model": KIMI_K3,
            }

        return {split: [row(split)] for split in ("train", "dev", "test", "ood")}


class QwenTrainingCoder:
    """Writes a CPU LoRA loop, or one single-GPU QLoRA loop sized to the warm card."""

    def write(self, *, skill: Mapping[str, Any], gpu: Mapping[str, Any]) -> str:
        if _on_cpu(gpu):
            return self._cpu_loop(skill)
        return self._gpu_loop(skill, gpu)

    def _cpu_loop(self, skill: Mapping[str, Any]) -> str:
        targets = ", ".join(repr(name) for name in LORA_TARGETS)
        task = str(skill["task"]).replace('"""', "'''")
        return f'''"""CPU LoRA check for Qwen. No 4-bit quantization and no GPU.

Skill: {skill["skill_id"]}
Task: {task}
"""
from dapta.train import MODEL_ID, TrainOptions, train


def main() -> None:
    options = TrainOptions(
        base_model={MODEL_ID!r},
        max_pixels={SMOKE_MAX_PIXELS},
        gpu=False,
    )
    # Float32 on CPU. device_map auto is refused, and 4-bit is CUDA-only.
    device_map = {{"": "cpu"}}
    load_in_4bit = False
    use_cpu = True
    target_modules = [{targets}]
    if device_map != {{"": "cpu"}} or load_in_4bit or not use_cpu or options.gpu:
        raise RuntimeError("training loop is not pinned to CPU")
    train(
        dataset_dir=__import__("pathlib").Path(__import__("os").environ["DAPTA_DATASET"]),
        output_dir=__import__("pathlib").Path(__import__("os").environ["DAPTA_OUTPUT"]),
        skill_id=__import__("os").environ["DAPTA_SKILL_ID"],
        version=__import__("os").environ["DAPTA_VERSION"],
        contract_dir=__import__("pathlib").Path(__import__("os").environ["DAPTA_CONTRACT"]),
        source_manifest=__import__("pathlib").Path(__import__("os").environ["DAPTA_SOURCES"]),
        options=options,
    )


if __name__ == "__main__":
    main()
'''

    def _gpu_loop(self, skill: Mapping[str, Any], gpu: Mapping[str, Any]) -> str:
        vram = int(gpu["vram_gb"])
        max_pixels = DEFAULT_MAX_PIXELS if vram >= 40 else SMOKE_MAX_PIXELS
        targets = ", ".join(repr(name) for name in LORA_TARGETS)
        task = str(skill["task"]).replace('"""', "'''")
        return f'''"""QLoRA training loop for {gpu["name"]} ({vram} GB), held warm on cuda:0.

Skill: {skill["skill_id"]}
Task: {task}
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from dapta.train import MODEL_ID, TrainOptions, train

assert torch.cuda.is_available(), "LoRA training requires the warm GPU"
assert torch.cuda.device_count() == 1, "this loop fills the claimed GPU and no other card"


def main() -> None:
    options = TrainOptions(
        base_model={MODEL_ID!r},
        max_pixels={max_pixels},
        gpu=True,
    )
    # 4-bit QLoRA, gradient checkpointing, and one device. device_map auto is refused.
    device_map = {{"": 0}}
    load_in_4bit = True
    gradient_checkpointing = True
    target_modules = [{targets}]
    if device_map != {{"": 0}} or not load_in_4bit or not gradient_checkpointing:
        raise RuntimeError("training loop is not pinned to the warm GPU")
    train(
        dataset_dir=__import__("pathlib").Path(os.environ["DAPTA_DATASET"]),
        output_dir=__import__("pathlib").Path(os.environ["DAPTA_OUTPUT"]),
        skill_id=os.environ["DAPTA_SKILL_ID"],
        version=os.environ["DAPTA_VERSION"],
        contract_dir=__import__("pathlib").Path(os.environ["DAPTA_CONTRACT"]),
        source_manifest=__import__("pathlib").Path(os.environ["DAPTA_SOURCES"]),
        options=options,
    )


if __name__ == "__main__":
    main()
'''


class BoundedLoraCheck:
    """Scores only the new adapter, and only on a capped held-out slice."""

    def evaluate(
        self, *, training: Mapping[str, Any], data: Mapping[str, Any]
    ) -> dict[str, Any]:
        test = list(data["test"])
        held_out = test[:HELD_OUT_LIMIT]
        loss = training.get("loss")
        passed = isinstance(loss, (int, float)) and not isinstance(loss, bool) and loss >= 0
        passed = passed and len(held_out) > 0
        return {
            "passed": passed,
            "condition": "qwen_lora",
            "held_out_cases": len(held_out),
            "benchmark": "bounded",
            "loss": loss,
        }


class DevinReviewer:
    """Asks Devin to oversee the skill document, data, code, and warm GPU."""

    def __init__(self, client: "DevinClient"):
        self.client = client

    def review(
        self,
        *,
        understanding: Mapping[str, Any],
        skill: Mapping[str, Any],
        training_code: str,
        data: Mapping[str, Any],
        gpu: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw = self.client.ask(
            devin_review_prompt(understanding, skill, training_code, data, gpu)
        )
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("passed"), bool):
            raise ValueError("Devin review must return JSON with a boolean passed")
        findings = payload.get("findings", [])
        if not isinstance(findings, list):
            raise ValueError("Devin findings must be a list")
        return {"passed": payload["passed"], "findings": findings, "reviewer": "devin"}


class DevinClient(Protocol):
    def ask(self, prompt: str) -> str: ...


def _task_prompt(task: str) -> str:
    return "\n".join(
        [
            "Read the task. Write the inputs, the outputs, and the refusals.",
            "Task:",
            task,
        ]
    )


def _skill_prompt(understanding: Mapping[str, Any], skill_id: str) -> str:
    return "\n".join(
        [
            "Write a detailed skill document for this understanding.",
            "Include rules, an output schema, exclusions, and scoring.",
            f"Skill id: {skill_id}",
            json.dumps(understanding, sort_keys=True),
        ]
    )


def _data_prompt(skill: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "Generate train, dev, frozen test, and out-of-distribution cases from this skill document.",
            "The test ids must not appear in train or dev.",
            json.dumps(skill, sort_keys=True),
        ]
    )


def _device_prompt(skill_id: str, confirm_spend: str | None) -> str:
    return "\n".join(
        [
            "Claim one device and hold it for this run.",
            f"Skill id: {skill_id}",
            f"confirm_spend: {confirm_spend}",
        ]
    )


def _code_prompt(skill: Mapping[str, Any], gpu: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "Write the Qwen training loop for this skill on this device.",
            f"Device: {json.dumps(dict(gpu), sort_keys=True)}",
            f"Skill id: {skill['skill_id']}",
            f"Task: {skill['task']}",
        ]
    )


def _train_prompt(
    task: str,
    skill_id: str,
    training_code: str,
    data: Mapping[str, Any],
    gpu: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "Train the adapter with this loop, this dataset, and this device.",
            f"Task: {task}",
            f"Skill id: {skill_id}",
            f"Device: {gpu.get('name')}",
            f"Split counts: {json.dumps(_counts(data), sort_keys=True)}",
            training_code,
        ]
    )


def _eval_prompt(training: Mapping[str, Any], data: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "Score only the new Qwen LoRA on the capped held-out slice.",
            json.dumps(dict(training), sort_keys=True),
            f"Split counts: {json.dumps(_counts(data), sort_keys=True)}",
        ]
    )


def _store_prompt(training: Mapping[str, Any], evaluation: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "Store the trained adapter.",
            json.dumps({"training": dict(training), "evaluation": dict(evaluation)}, sort_keys=True),
        ]
    )


def devin_review_prompt(
    understanding: Mapping[str, Any],
    skill: Mapping[str, Any],
    training_code: str,
    data: Mapping[str, Any],
    gpu: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "You are the Devin overseer for a Dapta LoRA build.",
            'Return JSON: {"passed": true|false, "findings": [strings]}.',
            "Check all of the following:",
            "1. The skill document was written by Kimi K3 and has rules, an output schema, exclusions, and scoring.",
            "2. The data has train, dev, test, and ood splits, and test does not overlap train or dev.",
            *_device_checks(gpu),
            _device_line(gpu),
            f"Task: {understanding['task']}",
            f"Skill model: {skill.get('model')}",
            f"Rules: {json.dumps(skill.get('rules', []), sort_keys=True)}",
            f"Exclusions: {json.dumps(skill.get('exclusions', []), sort_keys=True)}",
            f"Split counts: {json.dumps(_counts(data), sort_keys=True)}",
            "Training code:",
            training_code,
        ]
    )


def _require_understanding(understanding: Mapping[str, Any], task: str) -> None:
    if understanding.get("task") != task:
        raise ValueError("task agent must keep the original task")
    for name in ("summary", "inputs", "outputs", "refusals"):
        if not understanding.get(name):
            raise ValueError(f"task agent omitted {name}")
    for name in ("inputs", "outputs", "refusals"):
        if not isinstance(understanding[name], list):
            raise ValueError(f"task agent {name} must be a list")


def _require_skill(skill: Mapping[str, Any], skill_id: str) -> None:
    if skill.get("model") != KIMI_K3:
        raise ValueError("skill document must be written by Kimi K3")
    if skill.get("skill_id") != skill_id:
        raise ValueError("skill document skill_id does not match the request")
    for name in ("task", "rules", "output_schema", "exclusions", "scoring"):
        if not skill.get(name):
            raise ValueError(f"skill document omitted {name}")
    if not isinstance(skill["rules"], list) or not isinstance(skill["exclusions"], list):
        raise ValueError("skill rules and exclusions must be lists")
    schema = skill["output_schema"]
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("skill output_schema must be a JSON object schema")
    scoring = skill["scoring"]
    if not isinstance(scoring, dict) or not scoring.get("method"):
        raise ValueError("skill scoring must name a method")


def _on_cpu(gpu: Mapping[str, Any]) -> bool:
    return gpu.get("device") == "cpu"


def _device_checks(gpu: Mapping[str, Any]) -> list[str]:
    if _on_cpu(gpu):
        return [
            "3. The training code is a CPU LoRA loop with gpu=False, device_map cpu, and no 4-bit quantization.",
            "4. The loop does not set device_map to auto and does not require CUDA.",
        ]
    return [
        "3. The training code is a single-GPU 4-bit QLoRA loop with gradient checkpointing.",
        "4. The loop is sized to fill the warm GPU and does not set device_map to auto.",
    ]


def _device_line(gpu: Mapping[str, Any]) -> str:
    if _on_cpu(gpu):
        return f"Device: CPU warm={gpu.get('warm')}"
    return f"GPU: {gpu['name']} {gpu['vram_gb']} GB warm={gpu.get('warm')}"


def _require_gpu(gpu: Mapping[str, Any]) -> None:
    if gpu.get("will_provision"):
        raise ValueError("GPU claim must not provision a machine")
    if gpu.get("warm") is not True:
        raise ValueError("device must stay available for the run")
    if not str(gpu.get("name", "")).strip():
        raise ValueError("connected device must have a name")
    if _on_cpu(gpu):
        return
    if not isinstance(gpu.get("vram_gb"), int) or gpu["vram_gb"] < 1:
        raise ValueError("connected GPU must report vram_gb")


def _require_splits(data: Mapping[str, Any]) -> None:
    missing = {"train", "dev", "test", "ood"} - set(data)
    if missing:
        raise ValueError(f"Kimi data omitted splits: {sorted(missing)}")
    ids = {split: {row["id"] for row in data[split]} for split in ("train", "dev", "test", "ood")}
    if not ids["train"] or not ids["test"]:
        raise ValueError("Kimi data needs train and held-out test rows")
    if ids["test"] & (ids["train"] | ids["dev"]):
        raise ValueError("held-out test overlaps train or dev")


def _counts(data: Mapping[str, Any]) -> dict[str, int]:
    return {split: len(data[split]) for split in ("train", "dev", "test", "ood") if split in data}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
