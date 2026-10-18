from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping

from .observability import span


@dataclass(frozen=True)
class RouteDecision:
    decision: str
    adapter_id: str | None
    confidence: float
    reason: str
    workflow_fingerprint: str
    source: str
    requires_confirmation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorkflowRouter:
    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ.get("JEV_API_KEY")
        self.base_url = (base_url or os.environ.get("JEV_BASE_URL", "https://api.typesafe.ai")).rstrip("/")
        self.model = model or os.environ.get("JEV_MODEL", "jev-latest")

    def decide(self, workflow: str, adapters: Iterable[Mapping[str, Any]]) -> RouteDecision:
        workflow = workflow.strip()
        if not workflow:
            raise ValueError("workflow is required")
        fingerprint = hashlib.sha256(workflow.encode("utf-8")).hexdigest()
        adapter_list = list(adapters)
        with span("router.classify", "JEV workflow route", **{"dapta.workflow_hash": fingerprint}):
            if self.api_key:
                try:
                    return self._jev(workflow, adapter_list, fingerprint)
                except Exception:
                    pass
            return self._fallback(workflow, adapter_list, fingerprint)

    def _jev(
        self, workflow: str, adapters: list[Mapping[str, Any]], fingerprint: str
    ) -> RouteDecision:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Route the workflow to an adapter or request a new adapter. Return JSON "
                        "with decision (existing_adapter, create_adapter, unsupported), adapter_id, "
                        "confidence, and reason. Never execute the workflow."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"workflow": workflow, "adapters": adapter_list}),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as error:
            raise RuntimeError("JEV routing failed") from error
        content = result["choices"][0]["message"]["content"]
        value = json.loads(content) if isinstance(content, str) else content
        decision = str(value.get("decision", "create_adapter"))
        if decision not in {"existing_adapter", "create_adapter", "unsupported"}:
            decision = "create_adapter"
        adapter_id = value.get("adapter_id") if decision == "existing_adapter" else None
        return RouteDecision(
            decision=decision,
            adapter_id=adapter_id,
            confidence=max(0.0, min(1.0, float(value.get("confidence", 0)))),
            reason=str(value.get("reason", "JEV routing decision"))[:500],
            workflow_fingerprint=fingerprint,
            source="jev",
        )

    def _fallback(
        self, workflow: str, adapters: list[Mapping[str, Any]], fingerprint: str
    ) -> RouteDecision:
        normalized = workflow.lower()
        best: tuple[float, str | None, str] = (0.0, None, "No registered adapter matched")
        for adapter in adapters:
            metadata = adapter.get("metadata") or {}
            terms = " ".join(
                str(value)
                for value in (
                    adapter.get("adapter_id"),
                    adapter.get("skill_id"),
                    metadata.get("name"),
                    metadata.get("description"),
                    " ".join(metadata.get("keywords") or []),
                )
                if value
            ).lower()
            score = SequenceMatcher(None, normalized, terms).ratio()
            keyword_hits = sum(word in normalized for word in (metadata.get("keywords") or []))
            score = min(1.0, score + keyword_hits * 0.2)
            if score > best[0]:
                best = (score, adapter.get("adapter_id"), f"Registry similarity matched {adapter.get('skill_id')}")
        if best[0] >= 0.55 and best[1]:
            return RouteDecision(
                "existing_adapter", best[1], best[0], best[2], fingerprint, "registry_fallback", True
            )
        return RouteDecision(
            "create_adapter", None, max(0.0, 1 - best[0]), best[2], fingerprint, "registry_fallback", True
        )
