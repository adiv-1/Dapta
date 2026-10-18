from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


class DevinError(RuntimeError):
    pass


@dataclass(frozen=True)
class DevinSession:
    session_id: str
    url: str | None
    status: str
    pull_request_url: str | None = None
    acus_consumed: float | None = None


class DevinClient:
    def __init__(self, *, api_key: str, organization_id: str, base_url: str = "https://api.devin.ai"):
        if not api_key or not organization_id:
            raise ValueError("Devin API key and organization ID are required")
        self.api_key = api_key
        self.organization_id = organization_id
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "DevinClient":
        organization_id = os.environ.get("DEVIN_ORGANIZATION_ID") or os.environ.get("DEVIN_ORG_ID")
        return cls(
            api_key=os.environ["DEVIN_API_KEY"],
            organization_id=organization_id or "",
            base_url=os.environ.get("DEVIN_BASE_URL", "https://api.devin.ai"),
        )

    def create_integration_session(
        self,
        *,
        job_id: str,
        workflow: str,
        skill_contract: Mapping[str, Any],
        repository_url: str,
    ) -> DevinSession:
        prompt = (
            "Implement one scoped Dapta skill integration in the supplied repository. "
            "Create SKILL.md, a deterministic dataset adapter, validator, and tests from "
            "the contract below. Do not edit frozen test labels or quality thresholds. "
            "Run the tests and open a reviewable pull request.\n\n"
            f"Job: {job_id}\nWorkflow: {workflow}\nRepository: {repository_url}\n"
            f"Contract: {json.dumps(dict(skill_contract), ensure_ascii=False)}"
        )
        payload = self._request(
            "POST",
            f"/v3/organizations/{self.organization_id}/sessions",
            {"prompt": prompt, "tags": ["dapta", f"job:{job_id}"]},
        )
        return _session(payload)

    def get_session(self, session_id: str) -> DevinSession:
        payload = self._request(
            "GET", f"/v3/organizations/{self.organization_id}/sessions/{session_id}"
        )
        return _session(payload)

    def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise DevinError(f"Devin request failed with HTTP {error.code}: {body[:300]}") from error
        except (urllib.error.URLError, json.JSONDecodeError) as error:
            raise DevinError(f"Devin request failed: {error}") from error


def _session(payload: Mapping[str, Any]) -> DevinSession:
    session_id = payload.get("session_id") or payload.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise DevinError("Devin response did not include a session ID")
    pull_request = payload.get("pull_request") or {}
    pull_request_url = pull_request.get("url") if isinstance(pull_request, Mapping) else None
    return DevinSession(
        session_id=session_id,
        url=payload.get("url"),
        status=str(payload.get("status_enum") or payload.get("status") or "created"),
        pull_request_url=pull_request_url,
        acus_consumed=payload.get("acus_consumed"),
    )
