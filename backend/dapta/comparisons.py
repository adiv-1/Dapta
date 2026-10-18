from __future__ import annotations

import contextvars
import json
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from jsonschema import Draft202012Validator

from .config import Settings
from .ledger import UsageLedger
from .observability import (
    ai_agent, ai_chat, capture_exception, continued_transaction, error_category, log_event, mark_failed,
    record_ai_usage, span, trace_headers,
)
from .openrouter import OpenRouterClient


TERMINAL_RUN_STATES = {"completed", "failed"}


class ComparisonNotFound(LookupError):
    pass


class ComparisonStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create(
        self,
        *,
        workflow: str,
        adapter_id: str,
        idempotency_key: str,
        attachment_name: str | None,
    ) -> dict[str, Any]:
        now = _now()
        run_id = str(uuid.uuid4())
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO comparison_runs(
                        run_id, idempotency_key, workflow, adapter_id, attachment_name,
                        status, results_json, summary_json, error_message, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'queued', '{}', '{}', NULL, ?, ?)
                    """,
                    (run_id, idempotency_key, workflow, adapter_id, attachment_name, now, now),
                )
        except sqlite3.IntegrityError:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT run_id, workflow, adapter_id FROM comparison_runs WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
            if row is None or row["workflow"] != workflow or row["adapter_id"] != adapter_id:
                raise ValueError("idempotency key was already used for another comparison")
            return self.get(row["run_id"])
        self.add_event(run_id, "queued", {"adapter_id": adapter_id})
        return self.get(run_id)

    def get(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM comparison_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            events = connection.execute(
                "SELECT sequence, status, occurred_at, metadata_json FROM comparison_events "
                "WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        if row is None:
            raise ComparisonNotFound(run_id)
        result = dict(row)
        result["results"] = json.loads(result.pop("results_json"))
        result["summary"] = json.loads(result.pop("summary_json"))
        result["events"] = [
            {
                "sequence": event["sequence"],
                "status": event["status"],
                "occurred_at": event["occurred_at"],
                "metadata": json.loads(event["metadata_json"]),
            }
            for event in events
        ]
        return result

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            ids = [
                row["run_id"]
                for row in connection.execute(
                    "SELECT run_id FROM comparison_runs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            ]
        return [self.get(run_id) for run_id in ids]

    def recover_interrupted(self) -> int:
        """Fail runs left queued/running by a previous controller process.

        Comparisons run in a thread of the controller; after a restart nothing can finish them, and
        the frontend would poll a "running" comparison forever. Call once at controller startup.
        """
        with self._connect() as connection:
            ids = [row["run_id"] for row in connection.execute(
                "SELECT run_id FROM comparison_runs WHERE status NOT IN ('completed', 'failed')"
            ).fetchall()]
        for run_id in ids:
            self.update(run_id, status="failed",
                        error_message="Interrupted: the controller restarted before this comparison finished.")
            log_event("comparison_interrupted", level="warning", run_id=run_id,
                      error_category="controller_restart")
        return len(ids)

    def put_benchmark(self, benchmark_id: str, payload: Mapping[str, Any]) -> None:
        now = _now()
        encoded = json.dumps(dict(payload), sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO benchmark_snapshots(benchmark_id, payload_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(benchmark_id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (benchmark_id, encoded, now),
            )

    def list_benchmarks(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT benchmark_id, payload_json, created_at FROM benchmark_snapshots "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "benchmark_id": row["benchmark_id"],
                "created_at": row["created_at"],
                **json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def update(
        self,
        run_id: str,
        *,
        status: str,
        results: Mapping[str, Any] | None = None,
        summary: Mapping[str, Any] | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        now = _now()
        with self._connect() as connection:
            current = connection.execute(
                "SELECT results_json, summary_json FROM comparison_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if current is None:
                raise ComparisonNotFound(run_id)
            connection.execute(
                """
                UPDATE comparison_runs
                SET status = ?, results_json = ?, summary_json = ?, error_message = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    json.dumps(results) if results is not None else current["results_json"],
                    json.dumps(summary) if summary is not None else current["summary_json"],
                    error_message,
                    now,
                    run_id,
                ),
            )
        self.add_event(run_id, status)
        return self.get(run_id)

    def add_event(
        self, run_id: str, status: str, metadata: Mapping[str, Any] | None = None
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO comparison_events(run_id, status, occurred_at, metadata_json) VALUES (?, ?, ?, ?)",
                (run_id, status, _now(), json.dumps(dict(metadata or {}))),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS comparison_runs (
                    run_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    workflow TEXT NOT NULL,
                    adapter_id TEXT NOT NULL,
                    attachment_name TEXT,
                    status TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS comparison_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES comparison_runs(run_id),
                    status TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS benchmark_snapshots (
                    benchmark_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class QwenGateway:
    def __init__(self, *, url: str, token: str | None, timeout: int = 180):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def generate(
        self, *, run_id: str, workflow: str, document_text: str, adapter_id: str
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", **trace_headers()}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            f"{self.url}/generate",
            data=json.dumps(
                {
                    "run_id": run_id,
                    "workflow": workflow,
                    "document_text": document_text,
                    "adapter_id": adapter_id,
                }
            ).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Qwen inference failed: {error}") from error


class ComparisonService:
    def __init__(
        self,
        *,
        store: ComparisonStore,
        ledger: UsageLedger,
        settings: Settings,
        openrouter: OpenRouterClient | None = None,
        qwen: QwenGateway | None = None,
    ):
        self.store = store
        self.ledger = ledger
        self.settings = settings
        self.openrouter = openrouter
        self.qwen = qwen

    def start(
        self,
        *,
        workflow: str,
        document_text: str,
        adapter: Mapping[str, Any],
        idempotency_key: str,
        attachment_name: str | None = None,
        background: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        run = self.store.create(
            workflow=workflow,
            adapter_id=str(adapter["adapter_id"]),
            idempotency_key=idempotency_key,
            attachment_name=attachment_name,
        )
        try:
            self.ledger.create_run(
                run_id=run["run_id"],
                run_type="comparison",
                adapter_id=str(adapter["adapter_id"]),
                skill_id=str(adapter.get("skill_id") or ""),
                metadata={"document_sha256": _sha256(document_text), "attachment_name": attachment_name},
            )
        except Exception:
            pass
        callback = background or _start_thread
        callback(self.execute, run["run_id"], workflow, document_text, dict(adapter), trace_headers())
        log_event("comparison_started", run_id=run["run_id"], adapter=str(adapter["adapter_id"]),
                  adapter_version=adapter.get("version"))
        return run

    def execute(
        self,
        run_id: str,
        workflow: str,
        document_text: str,
        adapter: Mapping[str, Any],
        trace: Mapping[str, str] | None = None,
    ) -> None:
        self.store.update(run_id, status="running")
        try:
            # Runs in its own thread: continue the request's trace instead of starting an orphan span.
            with continued_transaction(trace, op="dapta.comparison", name="Dapta model comparison",
                                       **{"gen_ai.conversation.id": run_id, "dapta.run_id": run_id,
                                          "dapta.adapter_id": adapter.get("adapter_id"),
                                          "dapta.adapter_version": adapter.get("version")}) as comparison:
                started = time.perf_counter()
                conditions = self._run_conditions(run_id, workflow, document_text, adapter)
                summary = _summarize(conditions)
                completed = [name for name, value in conditions.items() if value.get("status") == "completed"]
                if comparison is not None:
                    comparison.set_data("dapta.conditions.completed", len(completed))
                    comparison.set_data("dapta.conditions.failed", len(conditions) - len(completed))
                    comparison.set_data("dapta.best_overall", summary.get("best_overall"))
                with span("db.sql.write", "Persist comparison results", **{"dapta.run_id": run_id}):
                    self.store.update(run_id, status="completed", results=conditions, summary=summary)
                    self.ledger.record_run_event(run_id=run_id, stage="compare", status="completed")
                log_event("comparison_completed", run_id=run_id, duration_ms=round((time.perf_counter() - started) * 1000),
                          conditions_completed=len(completed), conditions_failed=len(conditions) - len(completed),
                          best_overall=summary.get("best_overall"))
        except Exception as error:
            capture_exception(error, run_id=run_id, stage="compare")
            log_event("comparison_failed", level="error", run_id=run_id, error_category=error_category(error))
            self.store.update(run_id, status="failed", error_message=str(error)[:500])
            self.ledger.record_run_event(
                run_id=run_id,
                stage="compare",
                status="failed",
                metadata={"error_type": type(error).__name__},
            )

    def _run_conditions(
        self, run_id: str, workflow: str, document_text: str, adapter: Mapping[str, Any]
    ) -> dict[str, Any]:
        metadata = dict(adapter.get("metadata") or {})
        full_skill = str(metadata.get("skill_prompt") or workflow)
        calls = {
            "qwen_lora": lambda: self._run_qwen(run_id, workflow, document_text, adapter),
            "frontier_a": lambda: self._run_frontier(
                run_id, "frontier_a", self.settings.frontier_model_a, full_skill,
                document_text, self.settings.frontier_reasoning_a,
            ),
            "frontier_b": lambda: self._run_frontier(
                run_id, "frontier_b", self.settings.frontier_model_b, full_skill,
                document_text, self.settings.frontier_reasoning_b,
            ),
        }
        results: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(contextvars.copy_context().run, call): name for name, call in calls.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as error:
                    # One condition failing is a result, not a crash: the run still completes,
                    # but Sentry gets the failure with its condition, model and category.
                    capture_exception(error, run_id=run_id, condition=name, model=_model_for(name, self.settings),
                                      stage="compare")
                    log_event("condition_failed", level="warning", run_id=run_id, condition=name,
                              model=_model_for(name, self.settings), error_category=error_category(error))
                    results[name] = {
                        "status": "failed",
                        "model": _model_for(name, self.settings),
                        "error": str(error)[:300],
                        "output": {},
                        "metrics": {"latency_ms": None, "cost_usd": None, "accuracy": 0},
                    }
                self.store.add_event(run_id, f"{name}_completed", {"status": results[name]["status"]})
        self._score_results(document_text, results, metadata)
        return results

    def _run_qwen(
        self, run_id: str, workflow: str, document_text: str, adapter: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self.qwen is None:
            raise RuntimeError("DAPTA_QWEN_URL is not configured")
        started_at = _now()
        started = time.perf_counter()
        adapter_fields = {"dapta.run_id": run_id, "dapta.condition": "qwen_lora",
                          "dapta.adapter_id": str(adapter["adapter_id"]), "dapta.adapter_version": adapter.get("version")}
        with ai_agent("qwen_lora", system="dapta-qwen", model=self.settings.qwen_model, **adapter_fields) as agent, \
                ai_chat(self.settings.qwen_model, system="dapta-qwen", agent="qwen_lora", **adapter_fields) as call:
            try:
                response = self.qwen.generate(
                    run_id=run_id,
                    workflow=workflow,
                    document_text=document_text,
                    adapter_id=str(adapter["adapter_id"]),
                )
            except Exception as error:
                mark_failed(call, error, retry_count=0)
                mark_failed(agent, error)
                raise
            duration_ms = round((time.perf_counter() - started) * 1000)
            # Worker-reported counts only; a missing count stays missing in Sentry.
            record_ai_usage(call, {
                "input_tokens": response.get("input_tokens"), "output_tokens": response.get("output_tokens"),
                "total_tokens": (response["input_tokens"] + response["output_tokens"])
                if isinstance(response.get("input_tokens"), int) and isinstance(response.get("output_tokens"), int) else None,
                "cost_usd": Decimal(self.settings.hostinger_hourly_cost_usd) * Decimal(duration_ms) / Decimal(3_600_000),
                "cost_source": "calculated", "response_model": self.settings.qwen_model,
            })
            if call is not None:
                call.set_data("dapta.retry_count", 0)
                if response.get("backend"):
                    call.set_data("dapta.inference_backend", response["backend"])
                for key in ("latency_ms", "ttft_ms"):
                    if response.get(key) is not None:
                        call.set_data(f"dapta.worker.{key}", response[key])
        duration_ms = round((time.perf_counter() - started) * 1000)
        completed_at = _now()
        gpu_cost = Decimal(self.settings.hostinger_hourly_cost_usd) * Decimal(duration_ms) / Decimal(3_600_000)
        self.ledger.record_model_call(
            run_id=run_id,
            stage="compare",
            provider="hostinger",
            requested_model=self.settings.qwen_model,
            resolved_model=self.settings.qwen_model,
            condition="qwen_lora",
            agent_role="adapter_runtime",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            ttft_ms=response.get("ttft_ms"),
            prompt_tokens=int(response.get("input_tokens") or 0),
            completion_tokens=int(response.get("output_tokens") or 0),
            cost_usd=gpu_cost,
            cost_source="calculated",
            metadata={"adapter_id": adapter["adapter_id"], "gpu_type": "A100-80GB",
                      "inference_backend": response.get("backend")},
        )
        output = _object(response.get("output"))
        return {
            "status": "completed",
            "model": self.settings.qwen_model,
            "label": "Qwen LoRA",
            "output": output,
            "metrics": {
                "latency_ms": duration_ms,
                "ttft_ms": response.get("ttft_ms"),
                "input_tokens": int(response.get("input_tokens") or 0),
                "output_tokens": int(response.get("output_tokens") or 0),
                "cost_usd": str(gpu_cost.quantize(Decimal("0.000000001"))),
            },
        }

    def _run_frontier(
        self, run_id: str, condition: str, model: str, skill: str, document_text: str,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        if self.openrouter is None:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        started = time.perf_counter()
        with ai_agent(condition, system="openrouter", model=model, **{
            "dapta.run_id": run_id, "dapta.condition": condition,
        }) as agent:
            try:
                output = self.openrouter.complete_json(
                    run_id=run_id,
                    stage="compare",
                    agent_role=condition,
                    condition=condition,
                    model=model,
                    messages=[
                        {"role": "system", "content": skill + "\nReturn valid JSON only."},
                        {"role": "user", "content": document_text},
                    ],
                    max_tokens=3000,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as error:
                mark_failed(agent, error)
                raise
        duration_ms = round((time.perf_counter() - started) * 1000)
        summary = self.ledger.run_summary(run_id)
        condition_usage = next(
            (item for item in summary["conditions"] if item["condition"] == condition), {}
        )
        return {
            "status": "completed",
            "model": model,
            "label": model.split("/")[-1],
            "output": output,
            "metrics": {
                "latency_ms": duration_ms,
                "ttft_ms": None,
                "input_tokens": condition_usage.get("prompt_tokens", 0),
                "output_tokens": condition_usage.get("completion_tokens", 0),
                "cost_usd": condition_usage.get("cost_usd", "0"),
            },
        }

    def _score_results(
        self, document_text: str, results: dict[str, Any], metadata: Mapping[str, Any]
    ) -> None:
        required_fields = list(metadata.get("required_fields") or [])
        output_schema = metadata.get("output_schema")
        validator = (
            Draft202012Validator(output_schema)
            if isinstance(output_schema, Mapping) and output_schema
            else None
        )
        for result in results.values():
            output = result.get("output")
            valid = isinstance(output, dict) and bool(output)
            if valid and validator is not None:
                valid = not any(validator.iter_errors(output))
            present = 0
            grounded = 0
            fields = required_fields or (list(output) if isinstance(output, dict) else [])
            for field in fields:
                value = output.get(field) if isinstance(output, dict) else None
                if value not in (None, "", [], {}):
                    present += 1
                    if _is_grounded(field, value, document_text):
                        grounded += 1
            completeness = present / len(fields) if fields else float(valid)
            grounding = grounded / present if present else 0.0
            score = 100 * (0.4 * float(valid) + 0.3 * completeness + 0.3 * grounding)
            result.setdefault("metrics", {}).update(
                {
                    "schema_valid": valid,
                    "completeness": round(completeness, 4),
                    "grounding": round(grounding, 4),
                    "accuracy": round(score, 2),
                    "accuracy_label": "grounded evaluator score",
                }
            )


def _is_grounded(field: str, value: Any, document_text: str) -> bool:
    """Check normalized extraction values against prose without changing model output."""
    if isinstance(value, list):
        return bool(value) and all(_is_grounded(field, item, document_text) for item in value)
    if not isinstance(value, (str, int, float)):
        return False

    document = _searchable_text(document_text)
    candidate = _searchable_text(str(value).replace("_", " ").replace(":", " "))
    if candidate and candidate in document:
        return True

    if field == "effective_date" and isinstance(value, str):
        try:
            parsed = datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            pass
        else:
            variants = (
                f"{parsed.strftime('%B')} {parsed.day} {parsed.year}",
                f"{parsed.strftime('%b')} {parsed.day} {parsed.year}",
                f"{parsed.day} {parsed.strftime('%B')} {parsed.year}",
            )
            return any(_searchable_text(item) in document for item in variants)

    if field == "term" and isinstance(value, str):
        match = re.fullmatch(r"(\d+)_(days?|months?|years?)", value.lower())
        if match:
            number, unit = match.groups()
            words = {
                "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
                "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
                "12": "twelve", "18": "eighteen", "24": "twenty four", "36": "thirty six",
            }
            forms = [number]
            if number in words:
                forms.append(words[number])
            alternatives = "|".join(re.escape(item) for item in forms)
            expression = rf"\b(?:{alternatives})(?:\s+\d+)?\s+{re.escape(unit)}\b"
            return re.search(expression, document) is not None
    return False


def _searchable_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _summarize(results: Mapping[str, Any]) -> dict[str, Any]:
    completed = {
        name: value for name, value in results.items() if value.get("status") == "completed"
    }
    if not completed:
        return {"fastest": None, "cheapest": None, "most_accurate": None, "best_overall": None}
    fastest = min(completed, key=lambda name: completed[name]["metrics"]["latency_ms"])
    cheapest = min(
        completed,
        key=lambda name: Decimal(str(completed[name]["metrics"].get("cost_usd") or "999999")),
    )
    accurate = max(completed, key=lambda name: completed[name]["metrics"]["accuracy"])
    normalized: dict[str, float] = {}
    max_latency = max(value["metrics"]["latency_ms"] for value in completed.values()) or 1
    max_cost = max(
        Decimal(str(value["metrics"].get("cost_usd") or "0")) for value in completed.values()
    ) or Decimal("1")
    for name, value in completed.items():
        metrics = value["metrics"]
        normalized[name] = (
            0.6 * metrics["accuracy"] / 100
            + 0.2 * (1 - metrics["latency_ms"] / max_latency)
            + 0.2 * (1 - float(Decimal(str(metrics.get("cost_usd") or "0")) / max_cost))
        )
    return {
        "fastest": fastest,
        "cheapest": cheapest,
        "most_accurate": accurate,
        "best_overall": max(normalized, key=normalized.get),
    }


def _model_for(name: str, settings: Settings) -> str:
    return {
        "qwen_lora": settings.qwen_model,
        "frontier_a": settings.frontier_model_a,
        "frontier_b": settings.frontier_model_b,
    }[name]


def _object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {"raw_output": value}
    return {"raw_output": value}


def _sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _start_thread(function: Callable[..., Any], *args: Any) -> None:
    threading.Thread(target=function, args=args, daemon=True).start()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
