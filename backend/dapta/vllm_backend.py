"""Qwen + LoRA inference through a local vLLM OpenAI-compatible server.

The Qwen worker (model_server.py) keeps its API, auth, prompts and Sentry instrumentation; this
runtime replaces only the in-process transformers generation. vLLM runs as a separate process on
loopback (see deploy/runpod-start.sh), in its own virtualenv because it pins a different torch
than the training stack.

- Adapters: each adapter's subfolder is downloaded from QWEN_ADAPTER_REPO and registered with
  vLLM at runtime (/v1/load_lora_adapter), so newly published factory adapters need no restart.
- Lifecycle: unload() puts vLLM to sleep (weights to CPU RAM, KV cache freed) so the factory can
  train on the same GPU; load() wakes it up.
- Metrics: token counts come only from vLLM's reported usage (None when absent); TTFT is measured
  from the first streamed token.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

from .observability import redact_for_sentry, span

ADAPTER_ID = re.compile(r"[A-Za-z0-9._/-]{3,300}")


class VllmError(RuntimeError):
    pass


def adapter_alias(adapter_id: str) -> str:
    return "adapter_" + hashlib.sha256(adapter_id.encode("utf-8")).hexdigest()[:12]


def _download_adapter(repo_id: str, adapter_id: str, root: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id, allow_patterns=[f"{adapter_id}/*"], local_dir=str(root),
        token=os.environ.get("HF_TOKEN") or None,
    )
    return root / adapter_id


class VllmRuntime:
    backend = "vllm"

    def __init__(
        self,
        *,
        build_messages: Callable[[Any], list[dict[str, str]]],
        base_url: str | None = None,
        api_key: str | None = None,
        adapter_repo: str | None = None,
        adapter_root: Path | str | None = None,
        downloader: Callable[[str, str, Path], Path] = _download_adapter,
        timeout: float | None = None,
    ) -> None:
        self.build_messages = build_messages
        self.base_url = (base_url or os.environ.get("VLLM_URL", "http://127.0.0.1:8003")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("VLLM_API_KEY") or None
        self.adapter_repo = adapter_repo if adapter_repo is not None else os.environ.get("QWEN_ADAPTER_REPO")
        self.adapter_root = Path(adapter_root or os.environ.get("VLLM_ADAPTER_DIR", "artifacts/vllm-adapters")).resolve()
        self.downloader = downloader
        self.timeout = timeout if timeout is not None else float(os.environ.get("VLLM_TIMEOUT_SECONDS", "300"))
        self.base_model = os.environ.get("QWEN_BASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        # `model` mirrors the transformers runtime: None means the worker cannot generate.
        self.model: str | None = None
        self.loaded_adapters: dict[str, str] = {}
        self.lock = threading.Lock()

    # --- lifecycle ---------------------------------------------------------------------------

    def load(self) -> None:
        with self.lock:
            if self._sleeping():
                with span("model.load", "Wake vLLM", **{"dapta.backend": "vllm"}):
                    self._request("POST", "/wake_up")
                    if os.environ.get("VLLM_SLEEP_LEVEL", "1") == "2":
                        # Level 2 discarded the weights; vLLM reloads them from the model files.
                        self._request("POST", "/collective_rpc", {"method": "reload_weights"})
            self._wait_ready(float(os.environ.get("VLLM_READY_TIMEOUT_SECONDS", "900")))
            self.model = self.base_model
            # Re-register adapters after a wake-up rather than assuming they survived sleep.
            self.loaded_adapters = {}
            default_adapter = os.environ.get("QWEN_ADAPTER_SUBFOLDER")
            if self.adapter_repo and default_adapter:
                self._register(default_adapter)

    def unload(self) -> None:
        with self.lock:
            level = os.environ.get("VLLM_SLEEP_LEVEL", "1")
            with span("model.unload", "Put vLLM to sleep", **{"dapta.backend": "vllm", "dapta.sleep_level": level}):
                self._request("POST", f"/sleep?level={level}")
            self.model = None
            self.loaded_adapters = {}

    # --- inference ---------------------------------------------------------------------------

    def generate(self, request: Any) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("model is not loaded")
        with self.lock:
            alias = self._register(request.adapter_id)
        payload = {
            "model": alias,
            "messages": self.build_messages(request),
            "temperature": 0,
            "max_tokens": int(os.environ.get("QWEN_MAX_NEW_TOKENS", "1200")),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        with span("model.inference", "vLLM chat completion", **{
            "dapta.backend": "vllm", "dapta.adapter_id": request.adapter_id,
        }) as active:
            started = time.perf_counter()
            ttft_ms: int | None = None
            parts: list[str] = []
            usage: dict[str, Any] = {}
            finish_reason = None
            for chunk in self._stream("/v1/chat/completions", payload):
                for choice in chunk.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        if ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - started) * 1000)
                        parts.append(text)
                    finish_reason = choice.get("finish_reason") or finish_reason
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
            latency_ms = round((time.perf_counter() - started) * 1000)
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
            if active is not None:
                active.set_data("dapta.ttft_ms", ttft_ms)
                active.set_data("dapta.finish_reason", finish_reason)
                if input_tokens is not None:
                    active.set_data("gen_ai.usage.input_tokens", input_tokens)
                if output_tokens is not None:
                    active.set_data("gen_ai.usage.output_tokens", output_tokens)
        output_text = "".join(parts)
        try:
            output: Any = json.loads(output_text)
        except json.JSONDecodeError:
            output = output_text
        return {
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
            "finish_reason": finish_reason,
            "backend": self.backend,
        }

    # --- adapters ----------------------------------------------------------------------------

    def _register(self, adapter_id: str) -> str:
        """Download (once) and register an adapter with vLLM; caller holds the lock."""
        if not self.adapter_repo:
            raise RuntimeError("QWEN_ADAPTER_REPO is not configured")
        if not ADAPTER_ID.fullmatch(adapter_id) or ".." in adapter_id:
            raise ValueError("invalid adapter id")
        alias = self.loaded_adapters.get(adapter_id)
        if alias:
            return alias
        alias = adapter_alias(adapter_id)
        with span("model.adapter.load", "Load LoRA adapter into vLLM", **{
            "dapta.backend": "vllm", "dapta.adapter_id": adapter_id,
        }):
            path = self.adapter_root / adapter_id
            if not (path / "adapter_config.json").is_file():
                path = self.downloader(self.adapter_repo, adapter_id, self.adapter_root)
            if not (path / "adapter_config.json").is_file():
                raise RuntimeError(f"adapter {adapter_id} has no adapter_config.json")
            # load_inplace makes registration idempotent across wake-ups and restarts of this worker.
            self._request("POST", "/v1/load_lora_adapter", {
                "lora_name": alias, "lora_path": str(path), "load_inplace": True,
            })
        self.loaded_adapters[adapter_id] = alias
        return alias

    # --- HTTP --------------------------------------------------------------------------------

    def _sleeping(self) -> bool:
        try:
            result = self._request("GET", "/is_sleeping")
        except VllmError:
            return False  # dev-mode endpoints disabled: vLLM cannot be asleep
        return bool(isinstance(result, dict) and result.get("is_sleeping"))

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._request("GET", "/health")
                return
            except VllmError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(2)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _open(self, method: str, path: str, payload: dict[str, Any] | None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method, headers=self._headers())
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            detail = ""
            try:
                body = json.loads(error.read().decode("utf-8", errors="replace"))
                message = (body.get("error") or {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
                detail = f": {str(message)[:200]}" if message else ""
            except (ValueError, AttributeError):
                pass
            failure = VllmError(f"vLLM {path.split('?')[0]} failed with HTTP {error.code}{detail}")
            # vLLM error messages may quote request content; Sentry gets the status only.
            raise redact_for_sentry(failure, f"vLLM {path.split('?')[0]} failed with HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise VllmError(f"vLLM is unreachable at {self.base_url}: {getattr(error, 'reason', error)}") from error

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        with self._open(method, path, payload) as response:
            body = response.read().decode("utf-8")
        if not body.strip():
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

    def _stream(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with self._open("POST", path, payload) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict):
                    if "error" in chunk:
                        raise VllmError(f"vLLM stream failed: {str(chunk['error'])[:200]}")
                    yield chunk
