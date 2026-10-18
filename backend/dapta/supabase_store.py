from __future__ import annotations

import os
from typing import Any, Mapping

import httpx

from .jobs import JobStore


SENSITIVE_KEYS = {
    "api_key", "authorization", "document", "document_text", "password",
    "prompt", "request_body", "response_body", "secret", "token",
}


class SupabaseStoreError(RuntimeError):
    pass


class SupabaseFactoryStore:
    def __init__(
        self,
        url: str,
        secret_key: str,
        *,
        client: httpx.Client | None = None,
    ):
        if not url.startswith("https://") or ".supabase.co" not in url:
            raise ValueError("SUPABASE_URL must be an HTTPS Supabase project URL")
        if not secret_key:
            raise ValueError("a Supabase server secret is required")
        self.url = url.rstrip("/")
        self.secret_key = secret_key
        self.client = client or httpx.Client(timeout=30.0)

    @classmethod
    def from_env(cls) -> "SupabaseFactoryStore | None":
        url = os.environ.get("SUPABASE_URL", "").strip()
        secret = (
            os.environ.get("SUPABASE_SECRET_KEY", "").strip()
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        )
        if url and secret:
            return cls(url, secret)

        management_token = os.environ.get("SUPABASE_API_KEY", "").strip()
        if not management_token:
            return None
        project_ref = os.environ.get("SUPABASE_PROJECT_REF", "").strip()
        project_name = os.environ.get("SUPABASE_PROJECT_NAME", "dapta").strip()
        client = httpx.Client(timeout=30.0)
        if not project_ref:
            response = client.get(
                "https://api.supabase.com/v1/projects",
                headers={"Authorization": f"Bearer {management_token}"},
            )
            _require_success(response, "discover projects")
            projects = response.json()
            matches = [project for project in projects if project.get("name") == project_name]
            if len(matches) != 1:
                raise SupabaseStoreError(
                    "SUPABASE_PROJECT_REF is required when the management account does not "
                    f"contain exactly one project named {project_name!r}"
                )
            project_ref = str(matches[0].get("ref") or matches[0].get("id") or "")
        response = client.get(
            f"https://api.supabase.com/v1/projects/{project_ref}/api-keys",
            headers={"Authorization": f"Bearer {management_token}"},
        )
        _require_success(response, "resolve project secret")
        keys = response.json()
        selected = _server_key(keys)
        secret = str((selected or {}).get("api_key") or "")
        if not secret:
            raise SupabaseStoreError("the Supabase project has no server secret key")
        return cls(f"https://{project_ref}.supabase.co", secret)

    def sync_job(self, job: Mapping[str, Any], store: JobStore) -> None:
        publish = job.get("publish_result") or {}
        self._upsert(
            "dapta_factory_runs",
            "job_id",
            [{
                "job_id": job["job_id"],
                "workflow_hash": job["workflow_hash"],
                "skill_id": job.get("skill_id"),
                "status": job["status"],
                "dataset_round": int(job.get("dataset_round") or 1),
                "failure_code": job.get("failure_code"),
                "failure_message": _safe(job.get("failure_message")),
                "artifact_uri": publish.get("artifact_uri"),
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
            }],
        )
        events = [
            {
                "event_id": event["event_id"],
                "job_id": job["job_id"],
                "sequence": event["sequence"],
                "from_status": event.get("from_status"),
                "to_status": event["to_status"],
                "occurred_at": event["occurred_at"],
                "metadata": _safe(event.get("metadata") or {}),
            }
            for event in job.get("events") or []
        ]
        if events:
            self._upsert("dapta_factory_events", "event_id", events)

        artifacts = []
        for artifact in job.get("artifacts") or []:
            name = str(artifact["name"])
            try:
                payload = store.get_artifact(str(job["job_id"]), name)
            except Exception:
                continue
            artifacts.append({
                "job_id": job["job_id"],
                "name": name,
                "sha256": artifact["sha256"],
                "payload": _safe(payload),
                "created_at": artifact["created_at"],
            })
        if artifacts:
            self._upsert("dapta_factory_artifacts", "job_id,name", artifacts)

    def _upsert(self, table: str, conflict: str, rows: list[dict[str, Any]]) -> None:
        headers = {
            "apikey": self.secret_key,
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }
        if self.secret_key.startswith("eyJ"):
            headers["Authorization"] = f"Bearer {self.secret_key}"
        response = self.client.post(
            f"{self.url}/rest/v1/{table}",
            params={"on_conflict": conflict},
            headers=headers,
            json=rows,
        )
        _require_success(response, f"upsert {table}")


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[redacted]" if str(key).lower() in SENSITIVE_KEYS else _safe(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_safe(item) for item in value]
    if isinstance(value, str):
        return value[:100_000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:10_000]


def _server_key(keys: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    selected = next(
        (
            item for item in keys
            if item.get("type") == "secret" and _revealed_key(item.get("api_key"))
        ),
        None,
    )
    if selected is not None:
        return selected
    return next((item for item in keys if item.get("name") == "service_role"), None)


def _revealed_key(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sb_secret_") and value.isascii()


def _require_success(response: httpx.Response, action: str) -> None:
    if response.is_success:
        return
    detail = response.text.strip().replace("\n", " ")[:300]
    raise SupabaseStoreError(
        f"Supabase could not {action} (HTTP {response.status_code}): {detail}"
    )
