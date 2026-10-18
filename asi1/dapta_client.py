from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class DaptaClientError(RuntimeError):
    """A content-free error safe to return through the public agent."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class JsonTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None = None,
        timeout: float,
    ) -> Any: ...


class UrllibJsonTransport:
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None = None,
        timeout: float,
    ) -> Any:
        data = None
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", **dict(headers)},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            # The response may contain prompts or generated content. Never include it in a
            # message that can leave through Agentverse.
            raise DaptaClientError(
                f"DAPTA returned HTTP {error.code}", status=error.code
            ) from error
        except urllib.error.URLError as error:
            raise DaptaClientError("DAPTA is currently unreachable") from error
        except (TimeoutError, json.JSONDecodeError) as error:
            raise DaptaClientError("DAPTA returned an invalid or timed-out response") from error


@dataclass(frozen=True)
class DaptaClient:
    controller_url: str
    controller_token: str
    qwen_url: str
    qwen_token: str
    transport: JsonTransport = UrllibJsonTransport()
    timeout: float = 180.0

    def health(self) -> dict[str, Any]:
        result = self._controller("GET", "/health")
        if not isinstance(result, Mapping):
            raise DaptaClientError("DAPTA returned an invalid health response")
        return dict(result)

    def list_adapters(self) -> list[dict[str, Any]]:
        result = self._tool("list_adapters", {})
        if not isinstance(result, list):
            raise DaptaClientError("DAPTA returned an invalid adapter list")
        return [dict(item) for item in result if isinstance(item, Mapping)]

    def route_workflow(self, workflow: str) -> dict[str, Any]:
        result = self._controller("POST", "/v1/route", {"workflow": workflow})
        if not isinstance(result, Mapping):
            raise DaptaClientError("DAPTA returned an invalid routing decision")
        return dict(result)

    def create_adapter(
        self,
        *,
        workflow: str,
        idempotency_key: str,
        skill_id: str | None = None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "workflow": workflow,
            "idempotency_key": idempotency_key,
        }
        if skill_id:
            arguments["skill_id"] = skill_id
        result = self._tool(
            "create_adapter", arguments, idempotency_key=idempotency_key
        )
        if not isinstance(result, Mapping):
            raise DaptaClientError("DAPTA returned an invalid adapter job")
        return dict(result)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        result = self._tool("get_job_status", {"job_id": job_id})
        if not isinstance(result, Mapping):
            raise DaptaClientError("DAPTA returned an invalid job status")
        return dict(result)

    def execute_adapter(
        self,
        *,
        run_id: str,
        workflow: str,
        document_text: str,
        adapter_id: str,
    ) -> dict[str, Any]:
        result = self.transport.request(
            "POST",
            f"{self.qwen_url.rstrip('/')}/generate",
            headers={"Authorization": f"Bearer {self.qwen_token}"},
            payload={
                "run_id": run_id,
                "workflow": workflow,
                "document_text": document_text,
                "adapter_id": adapter_id,
            },
            timeout=self.timeout,
        )
        if not isinstance(result, Mapping):
            raise DaptaClientError("The LoRA worker returned an invalid result")
        return dict(result)

    def _tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> Any:
        payload: dict[str, Any] = {"arguments": dict(arguments)}
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        return self._controller("POST", f"/v1/agent/tools/{name}", payload)

    def _controller(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        return self.transport.request(
            method,
            f"{self.controller_url.rstrip('/')}{path}",
            headers={"X-Dapta-Token": self.controller_token},
            payload=payload,
            timeout=self.timeout,
        )
