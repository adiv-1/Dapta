from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from .ledger import UsageLedger


JOB_STAGES = (
    "queued",
    "architecting",
    "integrating",
    "building_dataset",
    "validating",
    "provisioning_gpu",
    "training",
    "evaluating",
    "publishing",
    "ready",
    "failed",
)

ALLOWED_TRANSITIONS = {
    "queued": {"architecting", "failed"},
    "architecting": {"integrating", "failed"},
    "integrating": {"building_dataset", "failed"},
    "building_dataset": {"validating", "failed"},
    "validating": {"provisioning_gpu", "failed"},
    "provisioning_gpu": {"training", "failed"},
    "training": {"evaluating", "failed"},
    "evaluating": {"publishing", "failed"},
    "publishing": {"ready", "failed"},
    "ready": set(),
    "failed": set(),
}


class JobNotFound(LookupError):
    pass


class JobConflict(ValueError):
    pass


class InvalidTransition(ValueError):
    pass


class FactoryAgentGateway(Protocol):
    def architect(self, *, run_id: str, workflow: str) -> Mapping[str, Any]: ...

    def build_dataset(
        self, *, run_id: str, workflow: str, skill_contract: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def evaluate(
        self, *, run_id: str, deterministic_report: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class JobStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_job(
        self,
        *,
        workflow: str,
        idempotency_key: str,
        skill_id: str | None = None,
        job_id: str | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        workflow = _required(workflow, "workflow")
        idempotency_key = _required(idempotency_key, "idempotency_key")
        skill_id = skill_id.strip() if skill_id else None
        job_id = job_id or str(uuid.uuid4())
        now = _timestamp(created_at)
        workflow_hash = hashlib.sha256(workflow.encode("utf-8")).hexdigest()

        try:
            with self._connect(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO jobs (
                        job_id, idempotency_key, workflow, workflow_hash, skill_id,
                        status, failure_code, failure_message, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'queued', NULL, NULL, ?, ?)
                    """,
                    (job_id, idempotency_key, workflow, workflow_hash, skill_id, now, now),
                )
                connection.execute(
                    """
                    INSERT INTO job_events (
                        event_id, job_id, from_status, to_status, occurred_at, metadata_json
                    ) VALUES (?, ?, NULL, 'queued', ?, '{}')
                    """,
                    (str(uuid.uuid4()), job_id, now),
                )
        except sqlite3.IntegrityError as error:
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
            if existing is None:
                raise JobConflict(f"job ID already exists: {job_id}") from error
            if existing["workflow_hash"] != workflow_hash or existing["skill_id"] != skill_id:
                raise JobConflict("idempotency key was already used for a different job") from error
            return self.get_job(existing["job_id"])

        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(job_id)
            events = connection.execute(
                "SELECT * FROM job_events WHERE job_id = ? ORDER BY sequence", (job_id,)
            ).fetchall()
            artifacts = connection.execute(
                """
                SELECT name, sha256, created_at
                FROM job_artifacts
                WHERE job_id = ?
                ORDER BY name
                """,
                (job_id,),
            ).fetchall()
            attempts = connection.execute(
                """
                SELECT stage, status, claimed_at, completed_at
                FROM job_stage_attempts
                WHERE job_id = ?
                ORDER BY claimed_at, stage
                """,
                (job_id,),
            ).fetchall()
        job = dict(row)
        job["events"] = [_event_dict(item) for item in events]
        job["artifacts"] = [dict(item) for item in artifacts]
        job["stage_attempts"] = [dict(item) for item in attempts]
        return job

    def list_jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, job_id LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def transition(
        self,
        job_id: str,
        to_status: str,
        *,
        occurred_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if to_status not in JOB_STAGES:
            raise ValueError(f"unknown job stage: {to_status}")
        now = _timestamp(occurred_at)
        metadata_json = _json(metadata)
        with self._connect(immediate=True) as connection:
            current = connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if current is None:
                raise JobNotFound(job_id)
            from_status = current["status"]
            if from_status == to_status:
                return self.get_job(job_id)
            if to_status not in ALLOWED_TRANSITIONS[from_status]:
                raise InvalidTransition(f"cannot move job from {from_status} to {to_status}")
            connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (to_status, now, job_id),
            )
            connection.execute(
                """
                INSERT INTO job_events (
                    event_id, job_id, from_status, to_status, occurred_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), job_id, from_status, to_status, now, metadata_json),
            )
        return self.get_job(job_id)

    def fail(
        self,
        job_id: str,
        *,
        code: str,
        message: str,
        occurred_at: str | None = None,
    ) -> dict[str, Any]:
        code = _required(code, "code")
        message = _required(message, "message")[:1000]
        now = _timestamp(occurred_at)
        with self._connect(immediate=True) as connection:
            current = connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if current is None:
                raise JobNotFound(job_id)
            from_status = current["status"]
            if from_status == "failed":
                return self.get_job(job_id)
            if "failed" not in ALLOWED_TRANSITIONS[from_status]:
                raise InvalidTransition(f"cannot fail a job in {from_status}")
            connection.execute(
                """
                UPDATE jobs
                SET status = 'failed', failure_code = ?, failure_message = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (code, message, now, job_id),
            )
            connection.execute(
                """
                INSERT INTO job_events (
                    event_id, job_id, from_status, to_status, occurred_at, metadata_json
                ) VALUES (?, ?, ?, 'failed', ?, ?)
                """,
                (str(uuid.uuid4()), job_id, from_status, now, _json({"code": code})),
            )
        return self.get_job(job_id)

    def put_artifact(
        self,
        job_id: str,
        name: str,
        value: Any,
        *,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        name = _required(name, "name")
        canonical = _json(value)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        now = _timestamp(created_at)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO job_artifacts(job_id, name, content_json, sha256, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (job_id, name, canonical, digest, now),
                )
        except sqlite3.IntegrityError as error:
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT sha256, created_at FROM job_artifacts WHERE job_id = ? AND name = ?",
                    (job_id, name),
                ).fetchone()
            if existing is not None and existing["sha256"] == digest:
                return {"name": name, "sha256": digest, "created_at": existing["created_at"]}
            raise JobConflict(f"artifact already exists with different content: {name}") from error
        return {"name": name, "sha256": digest, "created_at": now}

    def claim_stage(
        self, job_id: str, stage: str, *, claimed_at: str | None = None
    ) -> None:
        stage = _required(stage, "stage")
        now = _timestamp(claimed_at)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO job_stage_attempts(
                        job_id, stage, status, claimed_at, completed_at
                    ) VALUES (?, ?, 'running', ?, NULL)
                    """,
                    (job_id, stage, now),
                )
        except sqlite3.IntegrityError as error:
            with self._connect() as connection:
                job = connection.execute(
                    "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
            if job is None:
                raise JobNotFound(job_id) from error
            raise JobConflict(f"stage was already claimed: {stage}") from error

    def complete_stage(
        self,
        job_id: str,
        stage: str,
        *,
        status: str,
        completed_at: str | None = None,
    ) -> None:
        if status not in {"completed", "failed"}:
            raise ValueError("stage attempt status must be completed or failed")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE job_stage_attempts
                SET status = ?, completed_at = ?
                WHERE job_id = ? AND stage = ? AND status = 'running'
                """,
                (status, _timestamp(completed_at), job_id, stage),
            )
        if cursor.rowcount != 1:
            raise JobConflict(f"stage is not running: {stage}")

    def get_artifact(self, job_id: str, name: str) -> Any:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content_json FROM job_artifacts WHERE job_id = ? AND name = ?",
                (job_id, name),
            ).fetchone()
        if row is None:
            raise JobNotFound(f"{job_id}/{name}")
        return json.loads(row["content_json"])

    def register_adapter(
        self,
        *,
        adapter_id: str,
        skill_id: str,
        version: str,
        base_model: str,
        artifact_uri: str,
        metadata: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        now = _timestamp(created_at)
        values = (
            _required(adapter_id, "adapter_id"),
            _required(skill_id, "skill_id"),
            _required(version, "version"),
            _required(base_model, "base_model"),
            _required(artifact_uri, "artifact_uri"),
            _json(metadata),
            now,
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO adapters(
                        adapter_id, skill_id, version, base_model, artifact_uri,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
        except sqlite3.IntegrityError as error:
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT * FROM adapters WHERE adapter_id = ?", (adapter_id,)
                ).fetchone()
            if existing is not None and tuple(existing) == values:
                return _adapter_dict(existing)
            raise JobConflict(f"adapter already exists: {adapter_id}") from error
        return self.get_adapter(adapter_id)

    def get_adapter(self, adapter_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM adapters WHERE adapter_id = ?", (adapter_id,)
            ).fetchone()
        if row is None:
            raise JobNotFound(adapter_id)
        return _adapter_dict(row)

    def list_adapters(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM adapters ORDER BY created_at DESC, adapter_id"
            ).fetchall()
        return [_adapter_dict(row) for row in rows]

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS job_schema_info (
                    version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    workflow TEXT NOT NULL,
                    workflow_hash TEXT NOT NULL,
                    skill_id TEXT,
                    status TEXT NOT NULL,
                    failure_code TEXT,
                    failure_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS jobs_status_updated
                ON jobs(status, updated_at);

                CREATE TABLE IF NOT EXISTS job_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_artifacts (
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    name TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, name)
                );

                CREATE TABLE IF NOT EXISTS job_stage_attempts (
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    claimed_at TEXT NOT NULL,
                    completed_at TEXT,
                    PRIMARY KEY(job_id, stage)
                );

                CREATE TABLE IF NOT EXISTS adapters (
                    adapter_id TEXT PRIMARY KEY,
                    skill_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    artifact_uri TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(skill_id, version)
                );
                """
            )
            row = connection.execute("SELECT version FROM job_schema_info").fetchone()
            if row is None:
                connection.execute("INSERT INTO job_schema_info(version) VALUES (1)")
            elif row["version"] != 1:
                raise RuntimeError(f"unsupported job schema: {row['version']}")

    @contextmanager
    def _connect(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class AdapterJobOrchestrator:
    def __init__(
        self,
        store: JobStore,
        ledger: UsageLedger,
        agents: FactoryAgentGateway | None = None,
    ):
        self.store = store
        self.ledger = ledger
        self.agents = agents

    def create_job(
        self, *, workflow: str, idempotency_key: str, skill_id: str | None = None
    ) -> dict[str, Any]:
        job = self.store.create_job(
            workflow=workflow,
            idempotency_key=idempotency_key,
            skill_id=skill_id,
        )
        self.ledger.create_run(
            run_id=job["job_id"],
            run_type="adapter_build",
            job_id=job["job_id"],
            skill_id=job["skill_id"],
            started_at=job["created_at"],
            metadata={"workflow_hash": job["workflow_hash"]},
        )
        return job

    def architect(self, job_id: str) -> dict[str, Any]:
        job = self.store.transition(job_id, "architecting")
        self.store.claim_stage(job_id, "skill_architect")
        self.ledger.record_run_event(
            run_id=job_id, stage="skill_architect", status="started"
        )
        try:
            result = dict(self._agents().architect(run_id=job_id, workflow=job["workflow"]))
            required = {"skill_spec", "output_schema", "evaluation_plan"}
            missing = required - result.keys()
            if missing:
                raise ValueError(f"Skill Architect omitted: {sorted(missing)}")
            for name in sorted(required):
                self.store.put_artifact(job_id, f"{name}.json", result[name])
            self.store.complete_stage(job_id, "skill_architect", status="completed")
            job = self.store.transition(job_id, "integrating")
            self.ledger.record_run_event(
                run_id=job_id, stage="skill_architect", status="completed"
            )
            return job
        except Exception as error:
            _finish_failed_attempt(self.store, job_id, "skill_architect")
            self.store.fail(job_id, code=type(error).__name__, message=str(error))
            self.ledger.record_run_event(
                run_id=job_id,
                stage="skill_architect",
                status="failed",
                metadata={"error_type": type(error).__name__},
            )
            raise

    def complete_integration(
        self, job_id: str, *, commit_sha: str, test_summary: Mapping[str, Any]
    ) -> dict[str, Any]:
        self.store.put_artifact(
            job_id,
            "integration.json",
            {"commit_sha": _required(commit_sha, "commit_sha"), "tests": dict(test_summary)},
        )
        return self.store.transition(job_id, "building_dataset")

    def build_dataset(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job["status"] != "building_dataset":
            raise InvalidTransition("job must be in building_dataset")
        self.store.claim_stage(job_id, "data_builder")
        skill_contract = {
            name: self.store.get_artifact(job_id, f"{name}.json")
            for name in ("skill_spec", "output_schema", "evaluation_plan")
        }
        self.ledger.record_run_event(
            run_id=job_id, stage="data_builder", status="started"
        )
        try:
            result = dict(
                self._agents().build_dataset(
                    run_id=job_id,
                    workflow=job["workflow"],
                    skill_contract=skill_contract,
                )
            )
            required = {"train", "dev", "test", "ood"}
            missing = required - result.keys()
            if missing:
                raise ValueError(f"Data Builder omitted: {sorted(missing)}")
            self.store.put_artifact(job_id, "dataset_candidates.json", result)
            self.store.complete_stage(job_id, "data_builder", status="completed")
            job = self.store.transition(job_id, "validating")
            self.ledger.record_run_event(
                run_id=job_id, stage="data_builder", status="completed"
            )
            return job
        except Exception as error:
            _finish_failed_attempt(self.store, job_id, "data_builder")
            self.store.fail(job_id, code=type(error).__name__, message=str(error))
            self.ledger.record_run_event(
                run_id=job_id,
                stage="data_builder",
                status="failed",
                metadata={"error_type": type(error).__name__},
            )
            raise

    def evaluate(
        self, job_id: str, *, deterministic_report: Mapping[str, Any]
    ) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job["status"] != "evaluating":
            raise InvalidTransition("job must be in evaluating")
        self.store.claim_stage(job_id, "independent_evaluator")
        self.ledger.record_run_event(
            run_id=job_id, stage="independent_evaluator", status="started"
        )
        try:
            result = dict(
                self._agents().evaluate(
                    run_id=job_id,
                    deterministic_report=dict(deterministic_report),
                )
            )
            required = {"qualitative_findings", "failures"}
            missing = required - result.keys()
            if missing:
                raise ValueError(f"Independent Evaluator omitted: {sorted(missing)}")
            self.store.put_artifact(job_id, "qualitative_evaluation.json", result)
            self.store.complete_stage(job_id, "independent_evaluator", status="completed")
            job = self.store.transition(job_id, "publishing")
            self.ledger.record_run_event(
                run_id=job_id, stage="independent_evaluator", status="completed"
            )
            return job
        except Exception as error:
            _finish_failed_attempt(self.store, job_id, "independent_evaluator")
            self.store.fail(job_id, code=type(error).__name__, message=str(error))
            self.ledger.record_run_event(
                run_id=job_id,
                stage="independent_evaluator",
                status="failed",
                metadata={"error_type": type(error).__name__},
            )
            raise

    def _agents(self) -> FactoryAgentGateway:
        if self.agents is None:
            raise RuntimeError("factory agents are not configured in this process")
        return self.agents


def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["metadata"] = json.loads(result.pop("metadata_json"))
    return result


def _finish_failed_attempt(store: JobStore, job_id: str, stage: str) -> None:
    try:
        store.complete_stage(job_id, stage, status="failed")
    except JobConflict:
        pass


def _adapter_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["metadata"] = json.loads(result.pop("metadata_json"))
    return result


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _timestamp(value: str | None = None) -> str:
    if value is None:
        moment = datetime.now(timezone.utc)
    else:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if moment.tzinfo is None:
            raise ValueError("timestamps must include a timezone")
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), allow_nan=False)
