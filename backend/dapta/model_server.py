from __future__ import annotations

import json
import hashlib
import gc
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .observability import initialize_sentry


class GenerateRequest(BaseModel):
    run_id: str
    workflow: str = Field(min_length=3, max_length=20_000)
    document_text: str = Field(min_length=1, max_length=500_000)
    adapter_id: str


NDA_ADAPTER_ID = "nda-intake/qwen3-vl-8b-v1"
NDA_SHORT_SYSTEM = (
    "Complete the NDA intake form using the learned workflow. Return JSON only. "
    "Use underscore-normalized strings; term must be a number_unit string, never a JSON number."
)
NDA_USER_INSTRUCTION = """Fill this NDA intake form.

Return exactly:
{
  "effective_date": "YYYY-MM-DD or null",
  "jurisdiction": "normalized_jurisdiction or null",
  "parties": ["normalized_party_names"],
  "term": "number_unit or null"
}

Replace spaces and colons inside extracted values with underscores. The term is a string such as
"3_years", not the number 3.

NDA:
"""


def generation_messages(request: GenerateRequest) -> list[dict[str, str]]:
    """Build the inference prompt; the NDA path intentionally matches its frozen benchmark."""
    if request.adapter_id == NDA_ADAPTER_ID:
        return [
            {"role": "system", "content": NDA_SHORT_SYSTEM},
            {"role": "user", "content": NDA_USER_INSTRUCTION + request.document_text},
        ]
    return [
        {
            "role": "system",
            "content": "Complete the requested learned workflow. Return valid JSON only.",
        },
        {
            "role": "user",
            "content": f"Workflow: {request.workflow}\n\nDocument:\n{request.document_text}",
        },
    ]


class ModelRuntime:
    """In-process transformers + PEFT generation (rollback path: DAPTA_QWEN_BACKEND=transformers)."""

    backend = "transformers"

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.adapter_repo = os.environ.get("QWEN_ADAPTER_REPO")
        self.loaded_adapters: dict[str, str] = {}
        self.lock = threading.Lock()

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        with self.lock:
            if self.model is not None:
                return
            model_id = os.environ.get("QWEN_BASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
            self.processor = AutoProcessor.from_pretrained(model_id)
            base = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation=os.environ.get("QWEN_ATTN_IMPLEMENTATION", "sdpa"),
            )
            self.model = base
            self.model.eval()
            self.loaded_adapters = {}
            default_adapter = os.environ.get("QWEN_ADAPTER_SUBFOLDER")
            if self.adapter_repo and default_adapter:
                self.select_adapter(default_adapter)

    def unload(self) -> None:
        import torch

        with self.lock:
            self.model = None
            self.processor = None
            self.loaded_adapters = {}
            gc.collect()
            torch.cuda.empty_cache()

    def select_adapter(self, adapter_id: str) -> None:
        if not self.adapter_repo:
            raise RuntimeError("QWEN_ADAPTER_REPO is not configured")
        if not re.fullmatch(r"[A-Za-z0-9._/-]{3,300}", adapter_id) or ".." in adapter_id:
            raise ValueError("invalid adapter id")
        alias = self.loaded_adapters.get(adapter_id)
        if alias:
            self.model.set_adapter(alias)
            return

        from peft import PeftModel

        alias = "adapter_" + hashlib.sha256(adapter_id.encode("utf-8")).hexdigest()[:12]
        if not isinstance(self.model, PeftModel):
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_repo,
                subfolder=adapter_id,
                adapter_name=alias,
            )
        else:
            self.model.load_adapter(
                self.adapter_repo,
                subfolder=adapter_id,
                adapter_name=alias,
            )
        self.model.set_adapter(alias)
        self.loaded_adapters[adapter_id] = alias
        self.model.eval()

    def generate(self, request: GenerateRequest) -> dict[str, Any]:
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("model is not loaded")
        messages = generation_messages(request)
        with self.lock, torch.inference_mode():
            self.select_adapter(request.adapter_id)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to("cuda")
            started = time.perf_counter()
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(os.environ.get("QWEN_MAX_NEW_TOKENS", "1200")),
                do_sample=False,
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        output_text = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        try:
            output: Any = json.loads(output_text)
        except json.JSONDecodeError:
            output = output_text
        return {
            "output": output,
            "input_tokens": int(inputs.input_ids.shape[1]),
            "output_tokens": int(generated.shape[1]),
            "latency_ms": elapsed_ms,
            "ttft_ms": None,
            "backend": self.backend,
        }


def create_runtime() -> Any:
    """vLLM by default; the transformers runtime stays available as a rollback."""
    backend = os.environ.get("DAPTA_QWEN_BACKEND", "vllm").strip().lower()
    if backend == "vllm":
        from .vllm_backend import VllmRuntime

        return VllmRuntime(build_messages=generation_messages)
    if backend == "transformers":
        return ModelRuntime()
    raise RuntimeError(f"unknown DAPTA_QWEN_BACKEND: {backend}")


runtime = create_runtime()


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    runtime.load()
    yield


# Before the app is created, so Sentry's FastAPI integration continues the controller's trace
# from the incoming sentry-trace/baggage headers on /generate.
initialize_sentry(service="qwen-worker")
app = FastAPI(title="Dapta Qwen Worker", lifespan=lifespan)


def authorize(authorization: Optional[str] = Header(default=None)) -> None:
    expected = os.environ.get("DAPTA_QWEN_TOKEN")
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": runtime.backend,
        "model_loaded": runtime.model is not None,
        "loaded_adapters": list(runtime.loaded_adapters),
    }


@app.post("/generate", dependencies=[Depends(authorize)])
def generate(request: GenerateRequest) -> dict[str, Any]:
    return runtime.generate(request)


@app.post("/admin/unload", dependencies=[Depends(authorize)])
def unload() -> dict[str, Any]:
    runtime.unload()
    return {"status": "ok", "model_loaded": False}


@app.post("/admin/load", dependencies=[Depends(authorize)])
def load() -> dict[str, Any]:
    runtime.load()
    return {"status": "ok", "model_loaded": True}
