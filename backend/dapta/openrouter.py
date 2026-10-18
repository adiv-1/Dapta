from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol

from .ledger import UsageLedger, openrouter_usage
from .observability import ai_agent, ai_chat, log_event, mark_failed, record_ai_usage


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status
        # Provider error text can include request-specific details. Keep it for the local UI/ledger,
        # but replace it with a content-free message if an outer stage captures this exception.
        self.dapta_sentry_message = (
            f"OpenRouter request failed with HTTP {status}" if status is not None
            else "OpenRouter request failed"
        )


class JsonTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class UrllibJsonTransport:
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=dict(headers), method=method)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise OpenRouterError(_safe_error(body), status=error.code) from error
        except urllib.error.URLError as error:
            raise OpenRouterError(f"OpenRouter network error: {error.reason}") from error
        except json.JSONDecodeError as error:
            raise OpenRouterError("OpenRouter returned invalid JSON") from error


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        ledger: UsageLedger,
        transport: JsonTransport | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        app_url: str = "https://github.com/adiv-1/Dapta",
        app_name: str = "Dapta",
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        self.api_key = api_key
        self.ledger = ledger
        self.transport = transport or UrllibJsonTransport()
        self.base_url = base_url.rstrip("/")
        self.app_url = app_url
        self.app_name = app_name
        self.sleep = sleep

    def complete_json(
        self,
        *,
        run_id: str,
        stage: str,
        agent_role: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float = 0,
        benchmark_case_id: str | None = None,
        condition: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        with ai_chat(model, system="openrouter", agent=agent_role, **{
            "dapta.run_id": run_id, "dapta.stage": stage, "dapta.condition": condition or agent_role,
            "gen_ai.request.max_tokens": max_tokens, "gen_ai.request.temperature": temperature,
            "dapta.reasoning_effort": reasoning_effort,
            # One completion request per call; generation lookups are bookkeeping, not retries.
            "dapta.retry_count": 0,
        }) as telemetry:
            call_id = str(uuid.uuid4())
            started_at = _now()
            started_clock = time.monotonic()
            response: dict[str, Any] | None = None
            try:
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "usage": {"include": True},
                    "user": run_id,
                    "trace": {
                        "trace_id": run_id,
                        "trace_name": "dapta.adapter_build",
                        "span_name": f"gen_ai.{agent_role}",
                        "generation_name": agent_role,
                    },
                }
                # Anthropic models currently do not advertise response_format support through
                # OpenRouter. Their JSON is still required by the prompt and parsed defensively.
                if not model.startswith("anthropic/"):
                    payload["response_format"] = {"type": "json_object"}
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                if reasoning_effort:
                    payload["reasoning"] = {"effort": reasoning_effort}
                response = self.transport.request(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    payload=payload,
                )
                completed_at = _now()
                duration_ms = round((time.monotonic() - started_clock) * 1000)
                generation, reconciled = self._generation_metadata(response.get("id"))
                self.ledger.record_openrouter_call(
                    call_id=call_id,
                    run_id=run_id,
                    stage=stage,
                    agent_role=agent_role,
                    benchmark_case_id=benchmark_case_id,
                    condition=condition or agent_role,
                    requested_model=model,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                    response=response,
                    generation=generation,
                    metadata={
                        "generation_reconciled": reconciled,
                        "reasoning_effort": reasoning_effort,
                    },
                )
                usage = openrouter_usage(response, generation)
                record_ai_usage(telemetry, usage)
                if telemetry is not None:
                    telemetry.set_data("dapta.generation_reconciled", reconciled)
                    if usage["finish_reason"]:
                        telemetry.set_data("gen_ai.response.finish_reasons", str(usage["finish_reason"]))
                result = _response_json(response)
                log_event("model_call_completed", run_id=run_id, stage=stage, model=model, agent=agent_role,
                          duration_ms=duration_ms, input_tokens=usage["input_tokens"],
                          output_tokens=usage["output_tokens"], cost_source=usage["cost_source"], retry_count=0)
                return result
            except Exception as error:
                completed_at = _now()
                duration_ms = round((time.monotonic() - started_clock) * 1000)
                if response is None:
                    self.ledger.record_model_call(
                        call_id=call_id,
                        run_id=run_id,
                        stage=stage,
                        agent_role=agent_role,
                        benchmark_case_id=benchmark_case_id,
                        condition=condition or agent_role,
                        provider="openrouter",
                        requested_model=model,
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_ms=duration_ms,
                        status="failed",
                        error_code=type(error).__name__,
                        metadata={"generation_reconciled": False},
                    )
                category = mark_failed(telemetry, error, retry_count=0, duration_ms=duration_ms)
                log_event("model_call_failed", level="warning", run_id=run_id, stage=stage, model=model,
                          agent=agent_role, duration_ms=duration_ms, error_category=category, retry_count=0)
                raise

    def _generation_metadata(self, generation_id: Any) -> tuple[dict[str, Any], bool]:
        if not isinstance(generation_id, str) or not generation_id:
            return {}, False
        url = f"{self.base_url}/generation?{urllib.parse.urlencode({'id': generation_id})}"
        for attempt in range(3):
            try:
                return self.transport.request("GET", url, headers=self._headers()), True
            except OpenRouterError as error:
                if error.status != 404 or attempt == 2:
                    return {}, False
                self.sleep(0.25 * (attempt + 1))
        return {}, False

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.app_url,
            "X-Title": self.app_name,
            "X-OpenRouter-Metadata": "enabled",
        }


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    model: str
    stage: str
    max_tokens: int | None
    system_prompt: str


class FactoryAgents:
    def __init__(
        self,
        client: OpenRouterClient,
        *,
        factory_model: str,
        critic_model: str,
    ):
        self.client = client
        self.skill_architect = AgentDefinition(
            name="skill_architect",
            model=factory_model,
            stage="skill_architect",
            max_tokens=12000,
            system_prompt=(
                "You are Dapta's Skill Architect. Convert a stable repetitive workflow into "
                "a strict JSON contract. Return exactly three top-level objects: skill_spec, "
                "output_schema, and evaluation_plan. Define deterministic checks and explicit "
                "exclusions. skill_spec must include skill_id, name, description, instructions, "
                "and trainability. output_schema must be valid JSON Schema. evaluation_plan must "
                "define deterministic pass thresholds. Keep the final JSON under 3000 tokens. "
                "Do not invent training examples, expose chain-of-thought, or evaluate a model."
            ),
        )
        self.data_builder = AgentDefinition(
            name="data_builder",
            model=factory_model,
            stage="data_builder",
            max_tokens=None,
            system_prompt=(
                "You are Dapta's Data Builder. Follow the approved skill contract exactly. "
                "Return four top-level arrays: train, dev, test, and ood. Every item must be "
                "{id, input, output}; input is the complete user material and output conforms to "
                "output_schema. Return at least 16 train, 4 dev, 4 test, and 2 OOD examples. "
                "Keep every case traceable to deterministic truth or a cited public source. "
                "Never inspect model outputs, approve TEST, or alter evaluation rules. On repair "
                "rounds, use only supplied DEV feedback and never change frozen TEST."
            ),
        )
        self.independent_evaluator = AgentDefinition(
            name="independent_evaluator",
            model=critic_model,
            stage="model_evaluate",
            max_tokens=4000,
            system_prompt=(
                "You are Dapta's Independent Evaluator. Review only qualities that deterministic "
                "code cannot score. Return strict JSON with qualitative_findings and failures. "
                "Never change labels, thresholds, or objective metrics."
            ),
        )

    @classmethod
    def from_env(cls, ledger: UsageLedger) -> "FactoryAgents":
        client = OpenRouterClient(
            api_key=os.environ["OPENROUTER_API_KEY"],
            ledger=ledger,
            app_url=os.environ.get("OPENROUTER_APP_URL", "https://github.com/adiv-1/Dapta"),
        )
        return cls(
            client,
            factory_model=os.environ["OPENROUTER_FACTORY_MODEL"],
            critic_model=os.environ["OPENROUTER_CRITIC_MODEL"],
        )

    def architect(self, *, run_id: str, workflow: str) -> Mapping[str, Any]:
        return self._run(
            self.skill_architect,
            run_id=run_id,
            payload={"workflow": workflow},
        )

    def build_dataset(
        self,
        *,
        run_id: str,
        workflow: str,
        skill_contract: Mapping[str, Any],
        feedback: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._run(
            self.data_builder,
            run_id=run_id,
            payload={
                "workflow": workflow,
                "skill_contract": dict(skill_contract),
                "dev_feedback": dict(feedback or {}),
            },
        )

    def evaluate(
        self, *, run_id: str, deterministic_report: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return self._run(
            self.independent_evaluator,
            run_id=run_id,
            payload={"deterministic_report": dict(deterministic_report)},
        )

    def _run(
        self, definition: AgentDefinition, *, run_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        reasoning_effort = None
        if definition.name == "data_builder":
            reasoning_effort = os.environ.get("OPENROUTER_DATA_BUILDER_REASONING_EFFORT")
        with ai_agent(definition.name, system="openrouter", model=definition.model, **{
            "dapta.run_id": run_id, "dapta.stage": definition.stage,
        }):
            return self.client.complete_json(
                run_id=run_id,
                stage=definition.stage,
                agent_role=definition.name,
                model=definition.model,
                messages=[
                    {"role": "system", "content": definition.system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                    },
                ],
                max_tokens=definition.max_tokens,
                reasoning_effort=reasoning_effort,
            )


def _response_json(response: Mapping[str, Any]) -> dict[str, Any]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise OpenRouterError("OpenRouter response did not contain assistant content") from error
    if isinstance(content, Mapping) and "type" not in content:
        return dict(content)
    if content is None:
        usage = response.get("usage")
        reasoning_tokens = 0
        completion_tokens = 0
        if isinstance(usage, Mapping):
            completion_tokens = int(usage.get("completion_tokens") or 0)
            details = usage.get("completion_tokens_details")
            if isinstance(details, Mapping):
                reasoning_tokens = int(details.get("reasoning_tokens") or 0)
            reasoning_tokens = int(usage.get("reasoning_tokens") or reasoning_tokens)
        if completion_tokens and reasoning_tokens >= completion_tokens:
            raise OpenRouterError(
                "OpenRouter returned no assistant content because reasoning exhausted the completion budget"
            )
    content = _content_text(content).strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        # Some providers wrap otherwise valid JSON in a Markdown fence or one short sentence even
        # when explicitly asked for raw JSON. Accept one complete object, but never repair values.
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        start = content.find("{")
        try:
            value, _ = json.JSONDecoder().raw_decode(content[start:]) if start >= 0 else (None, 0)
        except json.JSONDecodeError as error:
            raise OpenRouterError("OpenRouter agent did not return valid JSON") from error
    if not isinstance(value, dict):
        raise OpenRouterError("OpenRouter agent response must be a JSON object")
    return value


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        for key in ("text", "content", "output_text"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                for key in ("text", "content", "output_text"):
                    value = item.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
        if parts:
            return "".join(parts)
    raise OpenRouterError("OpenRouter assistant content was not text")


def _safe_error(body: str) -> str:
    try:
        payload = json.loads(body)
        message = payload.get("error", {}).get("message") or payload.get("error")
        if isinstance(message, str):
            return f"OpenRouter request failed: {message[:300]}"
    except json.JSONDecodeError:
        pass
    return "OpenRouter request failed"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
