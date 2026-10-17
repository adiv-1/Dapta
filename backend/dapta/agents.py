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

BUILD_STAGES = JOB_STAGES
