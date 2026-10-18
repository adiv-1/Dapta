"""Stores every prompt and every agent output for a LoRA run.

The bench uses the Supabase secret key from the server. That key is not sent
to the browser.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

import httpx


class RunTrace(Protocol):
    def start(self, *, run_id: str, task: str, skill_id: str) -> None: ...

    def record(
        self, *, run_id: str, stage: str, agent: str, prompt: str, thought: Any
    ) -> None: ...

    def finish(
        self,
        *,
        run_id: str,
        status: str,
        device: str | None,
        artifact_uri: str | None,
    ) -> None: ...


class MemoryTrace:
    """In-memory trace for tests."""

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.finished: list[dict[str, Any]] = []

    def start(self, *, run_id: str, task: str, skill_id: str) -> None:
        self.runs.append({"run_id": run_id, "task": task, "skill_id": skill_id, "status": "running"})

    def record(
        self, *, run_id: str, stage: str, agent: str, prompt: str, thought: Any
    ) -> None:
        self.events.append(
            {
                "run_id": run_id,
                "stage": stage,
                "agent": agent,
                "prompt": prompt,
                "thought": thought,
            }
        )

    def finish(
        self,
        *,
        run_id: str,
        status: str,
        device: str | None,
        artifact_uri: str | None,
    ) -> None:
        self.finished.append(
            {
                "run_id": run_id,
                "status": status,
                "device": device,
                "artifact_uri": artifact_uri,
            }
        )


class UnconfiguredSupabase:
    def start(self, *, run_id: str, task: str, skill_id: str) -> None:
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SECRET_KEY, "
            "and run supabase/lora_trace.sql in the SQL editor."
        )

    def record(self, **kwargs: Any) -> None:
        raise RuntimeError("Supabase is not configured")

    def finish(self, **kwargs: Any) -> None:
        raise RuntimeError("Supabase is not configured")


class SupabaseTrace:
    def __init__(self, url: str, secret: str, client: httpx.Client | None = None):
        self.url = url.rstrip("/")
        self.secret = secret
        self.client = client or httpx.Client(timeout=30.0)

    def start(self, *, run_id: str, task: str, skill_id: str) -> None:
        self._send(
            "POST",
            "/rest/v1/lora_runs",
            {
                "run_id": run_id,
                "task": task,
                "skill_id": skill_id,
                "status": "running",
            },
        )

    def record(
        self, *, run_id: str, stage: str, agent: str, prompt: str, thought: Any
    ) -> None:
        self._send(
            "POST",
            "/rest/v1/lora_events",
            {
                "run_id": run_id,
                "stage": stage,
                "agent": agent,
                "prompt": prompt,
                "thought": thought,
            },
        )

    def finish(
        self,
        *,
        run_id: str,
        status: str,
        device: str | None,
        artifact_uri: str | None,
    ) -> None:
        self._send(
            "PATCH",
            f"/rest/v1/lora_runs?run_id=eq.{run_id}",
            {"status": status, "device": device, "artifact_uri": artifact_uri},
        )

    def _send(self, method: str, path: str, body: dict[str, Any]) -> None:
        response = self.client.request(
            method,
            f"{self.url}{path}",
            headers={
                "apikey": self.secret,
                "Authorization": f"Bearer {self.secret}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json=body,
        )
        if response.status_code not in (200, 201, 204):
            detail = response.text.strip().replace("\n", " ")[:300]
            raise RuntimeError(f"Supabase {method} {path} failed ({response.status_code}): {detail}")


def supabase_trace_from_env() -> RunTrace:
    url = os.environ.get("SUPABASE_URL", "").strip()
    secret = os.environ.get("SUPABASE_SECRET_KEY", "").strip()
    if not url or not secret:
        return UnconfiguredSupabase()
    return SupabaseTrace(url, secret)
