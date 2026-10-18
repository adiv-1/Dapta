from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    environment: str
    controller_token: str | None
    qwen_url: str | None
    qwen_api_token: str | None
    qwen_model: str
    frontier_model_a: str
    frontier_model_b: str
    frontier_reasoning_a: str | None
    frontier_reasoning_b: str | None
    evaluator_model: str
    hostinger_hourly_cost_usd: str
    sentry_dsn: str | None
    sentry_release: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            environment=os.environ.get("DAPTA_ENV", "development"),
            controller_token=os.environ.get("DAPTA_CONTROL_TOKEN") or None,
            qwen_url=os.environ.get("DAPTA_QWEN_URL") or None,
            qwen_api_token=os.environ.get("DAPTA_QWEN_TOKEN") or None,
            qwen_model=os.environ.get("QWEN_BASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
            frontier_model_a=os.environ.get(
                "OPENROUTER_FRONTIER_MODEL_A", "anthropic/claude-opus-5.5"
            ),
            frontier_model_b=os.environ.get(
                "OPENROUTER_FRONTIER_MODEL_B", "openai/gpt-5.6-sol"
            ),
            frontier_reasoning_a=os.environ.get("OPENROUTER_FRONTIER_REASONING_A") or None,
            frontier_reasoning_b=os.environ.get("OPENROUTER_FRONTIER_REASONING_B") or None,
            evaluator_model=os.environ.get(
                "OPENROUTER_CRITIC_MODEL", "openai/gpt-oss-120b"
            ),
            hostinger_hourly_cost_usd=os.environ.get(
                "HOSTINGER_GPU_HOURLY_COST_USD", "1.43"
            ),
            sentry_dsn=os.environ.get("SENTRY_DSN") or None,
            sentry_release=os.environ.get("SENTRY_RELEASE") or None,
        )
