from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterator, Mapping


SCHEMA_VERSION = 1
NANOS_PER_USD = Decimal("1000000000")
MODEL_COST_SOURCES = {"provider_reported", "calculated", "estimated", "none"}
COMPUTE_COST_SOURCES = {"provider_reported", "calculated", "estimated"}
COMPUTE_EVENTS = {
    "provision_requested",
    "created",
    "ready",
    "workload_started",
    "workload_finished",
    "destroy_requested",
    "destroyed",
    "failed",
}
FORBIDDEN_METADATA_KEYS = {
    "api_key",
    "authorization",
    "completion_text",
    "content",
    "document",
    "image_base64",
    "messages",
    "password",
    "prompt",
    "receipt",
    "request_body",
    "response_body",
    "response_text",
    "secret",
    "token",
}


class LedgerConflict(ValueError):
    pass


class UsageLedger:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_run(
        self,
        *,
        run_id: str,
        run_type: str,
        job_id: str | None = None,
        adapter_id: str | None = None,
        skill_id: str | None = None,
        benchmark_id: str | None = None,
        started_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        created_at = _timestamp(started_at)
        values = {
            "run_id": _required(run_id, "run_id"),
            "run_type": _required(run_type, "run_type"),
            "job_id": job_id,
            "adapter_id": adapter_id,
            "skill_id": skill_id,
            "benchmark_id": benchmark_id,
            "started_at": created_at,
            "metadata_json": _metadata_json(metadata),
            "created_at": created_at,
        }
        self._insert_idempotent("runs", "run_id", values)
        return run_id

    def record_run_event(
        self,
        *,
        run_id: str,
        stage: str,
        status: str,
        event_id: str | None = None,
        occurred_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        event_id = event_id or str(uuid.uuid4())
        values = {
            "event_id": event_id,
            "run_id": _required(run_id, "run_id"),
            "stage": _required(stage, "stage"),
            "status": _required(status, "status"),
            "occurred_at": _timestamp(occurred_at),
            "metadata_json": _metadata_json(metadata),
        }
        self._insert_idempotent("run_events", "event_id", values)
        return event_id

    def record_model_call(
        self,
        *,
        run_id: str,
        stage: str,
        provider: str,
        requested_model: str,
        started_at: str,
        completed_at: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
        total_tokens: int | None = None,
        cost_usd: Decimal | str | int | None = None,
        cost_source: str = "none",
        call_id: str | None = None,
        agent_role: str | None = None,
        benchmark_case_id: str | None = None,
        condition: str | None = None,
        resolved_model: str | None = None,
        provider_name: str | None = None,
        provider_request_id: str | None = None,
        provider_generation_id: str | None = None,
        duration_ms: int | None = None,
        ttft_ms: int | None = None,
        input_images: int = 0,
        input_audio: int = 0,
        output_images: int = 0,
        upstream_cost_usd: Decimal | str | int | None = None,
        status: str = "completed",
        finish_reason: str | None = None,
        error_code: str | None = None,
        usage: Mapping[str, Any] | None = None,
        pricing: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if cost_source not in MODEL_COST_SOURCES:
            raise ValueError(f"invalid model cost source: {cost_source}")
        token_values = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
            "input_images": input_images,
            "input_audio": input_audio,
            "output_images": output_images,
        }
        for name, value in token_values.items():
            _nonnegative_integer(value, name)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        _nonnegative_integer(total_tokens, "total_tokens")
        if duration_ms is not None:
            _nonnegative_integer(duration_ms, "duration_ms")
        if ttft_ms is not None:
            _nonnegative_integer(ttft_ms, "ttft_ms")

        cost_text, cost_nanos = _money(cost_usd)
        upstream_text, upstream_nanos = _money(upstream_cost_usd)
        if cost_source != "none" and cost_text is None:
            raise ValueError("cost_usd is required when cost_source is not 'none'")
        if cost_source == "none" and cost_text not in (None, "0"):
            raise ValueError("non-zero cost requires a cost source")

        call_id = call_id or str(uuid.uuid4())
        started = _timestamp(started_at)
        completed = _timestamp(completed_at)
        if _parse_timestamp(completed) < _parse_timestamp(started):
            raise ValueError("completed_at precedes started_at")
        if duration_ms is None:
            duration_ms = round(
                (_parse_timestamp(completed) - _parse_timestamp(started)).total_seconds() * 1000
            )

        values = {
            "call_id": call_id,
            "run_id": _required(run_id, "run_id"),
            "stage": _required(stage, "stage"),
            "agent_role": agent_role,
            "benchmark_case_id": benchmark_case_id,
            "condition_name": condition,
            "provider": _required(provider, "provider"),
            "requested_model": _required(requested_model, "requested_model"),
            "resolved_model": resolved_model,
            "provider_name": provider_name,
            "provider_request_id": provider_request_id,
            "provider_generation_id": provider_generation_id,
            "started_at": started,
            "completed_at": completed,
            "duration_ms": duration_ms,
            "ttft_ms": ttft_ms,
            **token_values,
            "total_tokens": total_tokens,
            "cost_usd": cost_text,
            "cost_nanos": cost_nanos,
            "cost_source": cost_source,
            "upstream_cost_usd": upstream_text,
            "upstream_cost_nanos": upstream_nanos,
            "status": _required(status, "status"),
            "finish_reason": finish_reason,
            "error_code": error_code,
            "usage_json": _metadata_json(usage),
            "pricing_json": _json(pricing),
            "metadata_json": _metadata_json(metadata),
            "created_at": completed,
        }
        self._insert_idempotent("model_calls", "call_id", values)
        return call_id

    def record_openrouter_call(
        self,
        *,
        run_id: str,
        stage: str,
        requested_model: str,
        started_at: str,
        completed_at: str,
        response: Mapping[str, Any],
        generation: Mapping[str, Any] | None = None,
        call_id: str | None = None,
        agent_role: str | None = None,
        benchmark_case_id: str | None = None,
        condition: str | None = None,
        duration_ms: int | None = None,
        ttft_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        generation_data = dict(generation or {})
        if "data" in generation_data and isinstance(generation_data["data"], Mapping):
            generation_data = dict(generation_data["data"])
        usage = dict(response.get("usage") or {})
        prompt_details = dict(usage.get("prompt_tokens_details") or {})
        completion_details = dict(usage.get("completion_tokens_details") or {})

        prompt_tokens = _first_integer(
            generation_data.get("native_tokens_prompt"),
            usage.get("prompt_tokens"),
            usage.get("input_tokens"),
        )
        completion_tokens = _first_integer(
            generation_data.get("native_tokens_completion"),
            usage.get("completion_tokens"),
            usage.get("output_tokens"),
        )
        reasoning_tokens = _first_integer(
            generation_data.get("native_tokens_reasoning"),
            completion_details.get("reasoning_tokens"),
        )
        cached_tokens = _first_integer(
            generation_data.get("native_tokens_cached"),
            prompt_details.get("cached_tokens"),
            dict(usage.get("input_tokens_details") or {}).get("cached_tokens"),
        )
        cache_write_tokens = _first_integer(prompt_details.get("cache_write_tokens"))
        total_tokens = _first_integer(usage.get("total_tokens"), prompt_tokens + completion_tokens)
        cost = generation_data.get("total_cost")
        if cost is None:
            cost = usage.get("cost")
        cost_source = "provider_reported" if cost is not None else "none"

        safe_generation_usage = {
            key: generation_data.get(key)
            for key in (
                "cache_discount",
                "cancelled",
                "generation_time",
                "latency",
                "native_finish_reason",
                "native_tokens_cached",
                "native_tokens_completion",
                "native_tokens_completion_images",
                "native_tokens_prompt",
                "native_tokens_reasoning",
                "num_fetches",
                "num_input_audio_prompt",
                "num_media_completion",
                "num_media_prompt",
                "num_search_results",
                "service_tier",
                "streamed",
                "total_cost",
                "upstream_inference_cost",
            )
            if generation_data.get(key) is not None
        }
        safe_usage = {"response_usage": usage, "generation_usage": safe_generation_usage}

        return self.record_model_call(
            run_id=run_id,
            stage=stage,
            provider="openrouter",
            requested_model=requested_model,
            resolved_model=generation_data.get("model") or response.get("model"),
            provider_name=generation_data.get("provider_name"),
            provider_request_id=generation_data.get("request_id"),
            provider_generation_id=generation_data.get("id") or response.get("id"),
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            ttft_ms=ttft_ms
            if ttft_ms is not None
            else _optional_nonnegative_integer(generation_data.get("latency")),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            total_tokens=total_tokens,
            input_images=_first_integer(generation_data.get("num_media_prompt")),
            input_audio=_first_integer(generation_data.get("num_input_audio_prompt")),
            output_images=_first_integer(generation_data.get("num_media_completion")),
            cost_usd=cost,
            cost_source=cost_source,
            upstream_cost_usd=generation_data.get("upstream_inference_cost"),
            status="cancelled" if generation_data.get("cancelled") else response.get("status", "completed"),
            finish_reason=generation_data.get("finish_reason")
            or generation_data.get("native_finish_reason"),
            usage=safe_usage,
            metadata=metadata,
            call_id=call_id,
            agent_role=agent_role,
            benchmark_case_id=benchmark_case_id,
            condition=condition,
        )

    def register_compute(
        self,
        *,
        run_id: str,
        allocation_id: str,
        provider: str,
        instance_id: str,
        plan_id: str,
        region: str,
        gpu_type: str,
        gpu_count: int,
        hourly_cost_usd: Decimal | str | int,
        pricing_source: str,
        registered_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        _nonnegative_integer(gpu_count, "gpu_count")
        if gpu_count == 0:
            raise ValueError("gpu_count must be positive")
        hourly_text, hourly_nanos = _money(hourly_cost_usd)
        values = {
            "allocation_id": _required(allocation_id, "allocation_id"),
            "run_id": _required(run_id, "run_id"),
            "provider": _required(provider, "provider"),
            "instance_id": _required(instance_id, "instance_id"),
            "plan_id": _required(plan_id, "plan_id"),
            "region": _required(region, "region"),
            "gpu_type": _required(gpu_type, "gpu_type"),
            "gpu_count": gpu_count,
            "hourly_cost_usd": hourly_text,
            "hourly_cost_nanos": hourly_nanos,
            "pricing_source": _required(pricing_source, "pricing_source"),
            "metadata_json": _metadata_json(metadata),
            "created_at": _timestamp(registered_at),
        }
        self._insert_idempotent("compute_allocations", "allocation_id", values)
        return allocation_id

    def record_compute_event(
        self,
        *,
        allocation_id: str,
        event_type: str,
        occurred_at: str | None = None,
        event_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if event_type not in COMPUTE_EVENTS:
            raise ValueError(f"invalid compute event: {event_type}")
        event_id = event_id or str(uuid.uuid4())
        values = {
            "event_id": event_id,
            "allocation_id": _required(allocation_id, "allocation_id"),
            "event_type": event_type,
            "occurred_at": _timestamp(occurred_at),
            "metadata_json": _metadata_json(metadata),
        }
        self._insert_idempotent("compute_events", "event_id", values)
        return event_id

    def record_compute_cost(
        self,
        *,
        allocation_id: str,
        amount_usd: Decimal | str | int,
        source: str,
        period_start: str,
        period_end: str,
        cost_id: str | None = None,
        recorded_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if source not in COMPUTE_COST_SOURCES:
            raise ValueError(f"invalid compute cost source: {source}")
        amount_text, amount_nanos = _money(amount_usd)
        start = _timestamp(period_start)
        end = _timestamp(period_end)
        if _parse_timestamp(end) < _parse_timestamp(start):
            raise ValueError("period_end precedes period_start")
        cost_id = cost_id or str(uuid.uuid4())
        values = {
            "cost_id": cost_id,
            "allocation_id": _required(allocation_id, "allocation_id"),
            "amount_usd": amount_text,
            "amount_nanos": amount_nanos,
            "source": source,
            "period_start": start,
            "period_end": end,
            "recorded_at": _timestamp(recorded_at),
            "metadata_json": _metadata_json(metadata),
        }
        self._insert_idempotent("compute_costs", "cost_id", values)
        return cost_id

    def estimate_destroyed_compute_cost(self, allocation_id: str) -> str:
        with self._connect() as connection:
            allocation = connection.execute(
                "SELECT * FROM compute_allocations WHERE allocation_id = ?", (allocation_id,)
            ).fetchone()
            if allocation is None:
                raise ValueError(f"unknown allocation: {allocation_id}")
            events = connection.execute(
                "SELECT event_type, occurred_at FROM compute_events WHERE allocation_id = ?",
                (allocation_id,),
            ).fetchall()
        by_type: dict[str, list[str]] = {}
        for event in events:
            by_type.setdefault(event["event_type"], []).append(event["occurred_at"])
        starts = by_type.get("created") or by_type.get("provision_requested")
        ends = by_type.get("destroyed")
        if not starts or not ends:
            raise ValueError("a created/provision_requested event and destroyed event are required")
        start = min(starts)
        end = max(ends)
        seconds = Decimal(str((_parse_timestamp(end) - _parse_timestamp(start)).total_seconds()))
        hourly = Decimal(allocation["hourly_cost_usd"])
        amount = hourly * seconds / Decimal(3600)
        cost_id = f"calculated:{allocation_id}:{end}"
        return self.record_compute_cost(
            allocation_id=allocation_id,
            amount_usd=amount,
            source="calculated",
            period_start=start,
            period_end=end,
            cost_id=cost_id,
            metadata={"billable_seconds": int(seconds)},
        )

    def run_summary(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            run = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                raise ValueError(f"unknown run: {run_id}")
            calls = connection.execute(
                "SELECT * FROM model_calls WHERE run_id = ? ORDER BY started_at, call_id", (run_id,)
            ).fetchall()
            compute_rows = connection.execute(
                """
                SELECT cc.*, ca.run_id
                FROM compute_costs cc
                JOIN compute_allocations ca USING (allocation_id)
                WHERE ca.run_id = ?
                ORDER BY cc.recorded_at, cc.cost_id
                """,
                (run_id,),
            ).fetchall()

        model_cost = sum((Decimal(row["cost_usd"] or "0") for row in calls), Decimal(0))
        selected_compute = _select_compute_costs(compute_rows)
        compute_cost = sum((Decimal(row["amount_usd"]) for row in selected_compute), Decimal(0))
        by_condition: dict[str, dict[str, Any]] = {}
        for row in calls:
            name = row["condition_name"] or "unassigned"
            group = by_condition.setdefault(
                name,
                {
                    "request_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "cached_tokens": 0,
                    "total_tokens": 0,
                    "cost": Decimal(0),
                    "duration_ms": 0,
                },
            )
            group["request_count"] += 1
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "cached_tokens",
                "total_tokens",
                "duration_ms",
            ):
                group[field] += row[field] or 0
            group["cost"] += Decimal(row["cost_usd"] or "0")

        serialized_conditions = []
        for name, values in sorted(by_condition.items()):
            values = dict(values)
            values["condition"] = name
            values["cost_usd"] = _decimal_text(values.pop("cost"))
            serialized_conditions.append(values)

        total_cost = model_cost + compute_cost
        return {
            "run_id": run_id,
            "run_type": run["run_type"],
            "job_id": run["job_id"],
            "adapter_id": run["adapter_id"],
            "skill_id": run["skill_id"],
            "benchmark_id": run["benchmark_id"],
            "model_request_count": len(calls),
            "prompt_tokens": sum(row["prompt_tokens"] for row in calls),
            "completion_tokens": sum(row["completion_tokens"] for row in calls),
            "reasoning_tokens": sum(row["reasoning_tokens"] for row in calls),
            "cached_tokens": sum(row["cached_tokens"] for row in calls),
            "total_tokens": sum(row["total_tokens"] for row in calls),
            "model_cost_usd": _decimal_text(model_cost),
            "compute_cost_usd": _decimal_text(compute_cost),
            "total_cost_usd": _decimal_text(total_cost),
            "total_cost_nanos": _money(total_cost)[1],
            "conditions": serialized_conditions,
        }

    def benchmark_summary(self, benchmark_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            run_ids = [
                row["run_id"]
                for row in connection.execute(
                    "SELECT run_id FROM runs WHERE benchmark_id = ? ORDER BY started_at, run_id",
                    (benchmark_id,),
                ).fetchall()
            ]
        summaries = [self.run_summary(run_id) for run_id in run_ids]
        total = sum((Decimal(item["total_cost_usd"]) for item in summaries), Decimal(0))
        return {
            "benchmark_id": benchmark_id,
            "run_count": len(summaries),
            "total_cost_usd": _decimal_text(total),
            "runs": summaries,
        }

    def list_benchmarks(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            benchmark_ids = [
                row["benchmark_id"]
                for row in connection.execute(
                    """
                    SELECT DISTINCT benchmark_id
                    FROM runs
                    WHERE benchmark_id IS NOT NULL
                    ORDER BY benchmark_id
                    """
                ).fetchall()
            ]
        return [self.benchmark_summary(benchmark_id) for benchmark_id in benchmark_ids]

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    job_id TEXT,
                    adapter_id TEXT,
                    skill_id TEXT,
                    benchmark_id TEXT,
                    started_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_calls (
                    call_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    stage TEXT NOT NULL,
                    agent_role TEXT,
                    benchmark_case_id TEXT,
                    condition_name TEXT,
                    provider TEXT NOT NULL,
                    requested_model TEXT NOT NULL,
                    resolved_model TEXT,
                    provider_name TEXT,
                    provider_request_id TEXT,
                    provider_generation_id TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
                    ttft_ms INTEGER CHECK (ttft_ms >= 0),
                    prompt_tokens INTEGER NOT NULL CHECK (prompt_tokens >= 0),
                    completion_tokens INTEGER NOT NULL CHECK (completion_tokens >= 0),
                    reasoning_tokens INTEGER NOT NULL CHECK (reasoning_tokens >= 0),
                    cached_tokens INTEGER NOT NULL CHECK (cached_tokens >= 0),
                    cache_write_tokens INTEGER NOT NULL CHECK (cache_write_tokens >= 0),
                    total_tokens INTEGER NOT NULL CHECK (total_tokens >= 0),
                    input_images INTEGER NOT NULL CHECK (input_images >= 0),
                    input_audio INTEGER NOT NULL CHECK (input_audio >= 0),
                    output_images INTEGER NOT NULL CHECK (output_images >= 0),
                    cost_usd TEXT,
                    cost_nanos INTEGER,
                    cost_source TEXT NOT NULL,
                    upstream_cost_usd TEXT,
                    upstream_cost_nanos INTEGER,
                    status TEXT NOT NULL,
                    finish_reason TEXT,
                    error_code TEXT,
                    usage_json TEXT NOT NULL,
                    pricing_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS model_calls_provider_generation
                ON model_calls(provider, provider_generation_id)
                WHERE provider_generation_id IS NOT NULL;

                CREATE INDEX IF NOT EXISTS model_calls_run_condition
                ON model_calls(run_id, condition_name, benchmark_case_id);

                CREATE TABLE IF NOT EXISTS compute_allocations (
                    allocation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    provider TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    region TEXT NOT NULL,
                    gpu_type TEXT NOT NULL,
                    gpu_count INTEGER NOT NULL CHECK (gpu_count > 0),
                    hourly_cost_usd TEXT NOT NULL,
                    hourly_cost_nanos INTEGER NOT NULL,
                    pricing_source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(provider, instance_id)
                );

                CREATE TABLE IF NOT EXISTS compute_events (
                    event_id TEXT PRIMARY KEY,
                    allocation_id TEXT NOT NULL REFERENCES compute_allocations(allocation_id),
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS compute_events_allocation_time
                ON compute_events(allocation_id, occurred_at);

                CREATE TABLE IF NOT EXISTS compute_costs (
                    cost_id TEXT PRIMARY KEY,
                    allocation_id TEXT NOT NULL REFERENCES compute_allocations(allocation_id),
                    amount_usd TEXT NOT NULL,
                    amount_nanos INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS compute_costs_allocation
                ON compute_costs(allocation_id, source, recorded_at);
                """
            )
            row = connection.execute("SELECT version FROM schema_info").fetchone()
            if row is None:
                connection.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            elif row["version"] != SCHEMA_VERSION:
                raise RuntimeError(
                    f"unsupported ledger schema {row['version']}; expected {SCHEMA_VERSION}"
                )

    def _insert_idempotent(self, table: str, key: str, values: Mapping[str, Any]) -> None:
        columns = tuple(values)
        placeholders = ", ".join("?" for _ in columns)
        statement = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        try:
            with self._connect() as connection:
                connection.execute(statement, tuple(values[column] for column in columns))
        except sqlite3.IntegrityError as error:
            with self._connect() as connection:
                existing = connection.execute(
                    f"SELECT {', '.join(columns)} FROM {table} WHERE {key} = ?", (values[key],)
                ).fetchone()
            if existing is not None and all(existing[column] == values[column] for column in columns):
                return
            raise LedgerConflict(f"conflicting or duplicate {table} record: {values[key]}") from error

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _select_compute_costs(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    priority = {"estimated": 0, "calculated": 1, "provider_reported": 2}
    selected: dict[str, sqlite3.Row] = {}
    for row in rows:
        current = selected.get(row["allocation_id"])
        if current is None:
            selected[row["allocation_id"]] = row
            continue
        candidate_rank = (priority[row["source"]], row["recorded_at"], row["cost_id"])
        current_rank = (priority[current["source"]], current["recorded_at"], current["cost_id"])
        if candidate_rank > current_rank:
            selected[row["allocation_id"]] = row
    return list(selected.values())


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _nonnegative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _money(value: Decimal | str | int | None) -> tuple[str | None, int | None]:
    if value is None:
        return None, None
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"invalid USD amount: {value}") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError("USD amount must be finite and non-negative")
    nanos = int((amount * NANOS_PER_USD).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return _decimal_text(amount), nanos


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _timestamp(value: str | None) -> str:
    if value is None:
        moment = datetime.now(timezone.utc)
    else:
        moment = _parse_timestamp(value)
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid timestamp: {value}") from error
    if moment.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return moment.astimezone(timezone.utc)


def _first_integer(*values: Any) -> int:
    for value in values:
        if value is not None:
            parsed = int(value)
            _nonnegative_integer(parsed, "usage value")
            return parsed
    return 0


def _optional_nonnegative_integer(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    _nonnegative_integer(parsed, "usage value")
    return parsed


def _metadata_json(value: Mapping[str, Any] | None) -> str:
    metadata = dict(value or {})
    _reject_sensitive_metadata(metadata)
    return _json(metadata)


def _reject_sensitive_metadata(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_METADATA_KEYS or normalized.endswith("_content"):
                raise ValueError(f"sensitive metadata field is not allowed: {key}")
            _reject_sensitive_metadata(child)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_metadata(child)


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), allow_nan=False)
