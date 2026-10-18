"""Local LoRA-agent bench.

Runs the real orchestrator. Task understanding, the Kimi skill document, and
the Kimi dataset are local writers. The default device is CPU. Devin, training,
and the adapter store are stand-ins, so a run does not load Qwen or spend money.
"""

from __future__ import annotations

import argparse
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Mapping

from fastapi.responses import FileResponse

from .api import create_app
from .jobs import JobStore
from .preflight import load_env
from .ledger import UsageLedger
from .lora import (
    BoundedLoraCheck,
    KimiDataBuilder,
    KimiSkillWriter,
    LoraAgent,
    QwenTrainingCoder,
    TaskReader,
)
from .supabase_trace import supabase_trace_from_env
from .train import MODEL_ID


REVIEW = ContextVar("demo_review", default="pass")
GPU = ContextVar("demo_gpu", default="cpu")
FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "index.html"

GPUS = {
    "cpu": {"name": "CPU", "device": "cpu", "warm": True, "will_provision": False},
    "a40": {
        "name": "NVIDIA A40",
        "device": "cuda",
        "vram_gb": 48,
        "warm": True,
        "will_provision": False,
    },
    "a16": {
        "name": "NVIDIA A16",
        "device": "cuda",
        "vram_gb": 16,
        "warm": True,
        "will_provision": False,
    },
}


class DemoGpu:
    """Warm stand-in. confirm_spend never creates a Vultr instance here."""

    def claim(self, *, confirm_spend: str | None) -> dict[str, Any]:
        choice = GPU.get()
        if choice not in GPUS:
            raise ValueError("demo device must be cpu, a40, or a16")
        card = dict(GPUS[choice])
        card["confirm_spend"] = confirm_spend
        return card


class DemoDevin:
    def review(
        self,
        *,
        understanding: Mapping[str, Any],
        skill: Mapping[str, Any],
        training_code: str,
        data: Mapping[str, Any],
        gpu: Mapping[str, Any],
    ) -> dict[str, Any]:
        if REVIEW.get() == "fail":
            return {
                "passed": False,
                "findings": ["Devin rejected this package from the test bench."],
                "reviewer": "devin",
            }
        findings = []
        if skill.get("model") != "moonshotai/kimi-k3" or not skill.get("rules"):
            findings.append("skill document is missing Kimi K3 rules")
        if not skill.get("exclusions") or not skill.get("scoring"):
            findings.append("skill document is missing exclusions or scoring")
        if gpu.get("device") == "cpu":
            if "gpu=False" not in training_code:
                findings.append("CPU training must set gpu=False")
            if 'device_map = {"": "cpu"}' not in training_code:
                findings.append("training loop is not pinned to CPU")
            if "load_in_4bit = True" in training_code:
                findings.append("CPU training must not use 4-bit")
        else:
            if "load_in_4bit = True" not in training_code:
                findings.append("training loop is not 4-bit")
            if 'device_map="auto"' in training_code:
                findings.append("training loop shards across GPUs")
        if not data["test"] or {row["id"] for row in data["test"]} & {
            row["id"] for row in data["train"]
        }:
            findings.append("held-out test overlaps train")
        if gpu.get("warm") is not True:
            findings.append("device is not held for the run")
        if not understanding.get("task"):
            findings.append("task understanding is empty")
        return {"passed": not findings, "findings": findings, "reviewer": "devin"}


class DemoTrainer:
    def train(
        self,
        *,
        task: str,
        skill_id: str,
        training_code: str,
        data: Mapping[str, Any],
        gpu: Mapping[str, Any],
        run_id: str,
    ) -> dict[str, Any]:
        return {
            "version": f"demo-{uuid.uuid4().hex[:8]}",
            "base_model": MODEL_ID,
            "loss": 0.42,
            "gpu": gpu["name"],
        }


class DemoPublisher:
    def store(self, *, training: Mapping[str, Any], evaluation: Mapping[str, Any]) -> str:
        return f"hf://dapta-demo/adapters/{training['version']}"


def demo_agent(database: Path | str) -> LoraAgent:
    store = JobStore(database)
    return LoraAgent(
        store,
        UsageLedger(database),
        task_agent=TaskReader(),
        skill_agent=KimiSkillWriter(),
        data=KimiDataBuilder(),
        gpu=DemoGpu(),
        coder=QwenTrainingCoder(),
        reviewer=DemoDevin(),
        trainer=DemoTrainer(),
        evaluator=BoundedLoraCheck(),
        publisher=DemoPublisher(),
        trace=supabase_trace_from_env(),
    )


def demo_app(database: Path | str) -> Any:
    app = create_app(database, lora_agent=demo_agent(database))

    @app.middleware("http")
    async def demo_controls(request, call_next):
        review = REVIEW.set(request.headers.get("x-devin-review", "pass"))
        gpu = GPU.set(request.headers.get("x-demo-gpu", "a40"))
        try:
            return await call_next(request)
        finally:
            REVIEW.reset(review)
            GPU.reset(gpu)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND)

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--database", type=Path, default=Path("artifacts/lora-demo.sqlite3"))
    args = parser.parse_args()
    load_env(Path(".env"))
    args.database.parent.mkdir(parents=True, exist_ok=True)
    uvicorn.run(demo_app(args.database), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
