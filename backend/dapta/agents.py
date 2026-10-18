from dataclasses import dataclass

from .jobs import JOB_STAGES


@dataclass(frozen=True)
class AgentRole:
    name: str
    model: str
    output: str


FACTORY_AGENTS = (
    AgentRole(
        "skill_architect",
        "moonshotai/kimi-k3",
        "A structured skill specification with rules, schema, and exclusions.",
    ),
    AgentRole(
        "data_builder",
        "moonshotai/kimi-k3",
        "TRAIN, DEV, frozen TEST, and OOD cases that follow the dataset contract.",
    ),
    AgentRole(
        "adapter_builder",
        "deterministic-worker",
        "A versioned Qwen3-VL LoRA plus reproducible training metadata.",
    ),
    AgentRole(
        "independent_evaluator",
        "openai/gpt-oss-120b",
        "Code-first metrics and a bounded qualitative review for subjective fields.",
    ),
)

LORA_AGENT = AgentRole(
    "lora_agent",
    "orchestrator",
    "Runs task understanding, a Kimi K3 skill document, Kimi K3 data, a warm GPU, Qwen training code, and a Devin review.",
)

LORA_PIPELINE = (
    AgentRole(
        "task_agent",
        "task-reader",
        "Inputs, outputs, and refusals for the requested workflow.",
    ),
    AgentRole(
        "skill_agent",
        "moonshotai/kimi-k3",
        "A detailed skill document with rules, schema, exclusions, and scoring.",
    ),
    AgentRole(
        "data_agent",
        "moonshotai/kimi-k3",
        "Train, dev, frozen test, and out-of-distribution cases for that skill.",
    ),
    AgentRole(
        "gpu_agent",
        "warm-gpu",
        "One GPU held warm for the rest of the run.",
    ),
    AgentRole(
        "training_code_agent",
        "Qwen/Qwen3-VL-8B-Instruct",
        "A single-GPU QLoRA training loop sized to the warm GPU.",
    ),
    AgentRole(
        "devin_overseer",
        "devin",
        "A pass or fail on the skill document, data, training code, and GPU use.",
    ),
)

BUILD_STAGES = JOB_STAGES
