from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol

from .ledger import UsageLedger
from .cancellation import BuildCancelled, BuildCleanupError, cancellation_scope, check_cancelled
from .observability import FactoryStage, emit_job_root, error_category, log_event, span


SCHEMA_VERSION = 2
JOB_STAGES = (
    "queued", "architecting", "integrating", "building_dataset", "validating",
    "awaiting_compute", "provisioning_gpu", "training", "evaluating", "publishing",
    "ready", "failed", "cancelling", "cancelled",
)
ALLOWED_TRANSITIONS = {
    "queued": {"architecting", "failed"},
    "architecting": {"integrating", "failed"},
    "integrating": {"building_dataset", "failed"},
    "building_dataset": {"validating", "failed"},
    "validating": {"awaiting_compute", "failed"},
    "awaiting_compute": {"provisioning_gpu", "failed"},
    "provisioning_gpu": {"training", "failed"},
    "training": {"evaluating", "failed"},
    "evaluating": {"publishing", "building_dataset", "failed"},
    "publishing": {"ready", "failed"},
    "ready": set(),
    "failed": set(),
    "cancelling": {"cancelled"},
    "cancelled": set(),
}
for _stage_name in JOB_STAGES:
    if _stage_name not in {"ready", "failed", "cancelling", "cancelled"}:
        ALLOWED_TRANSITIONS[_stage_name].add("cancelling")


class JobNotFound(LookupError):
    pass


class JobConflict(ValueError):
    pass


class EvaluationOutputTruncated(RuntimeError):
    """The benchmark hit its generation ceiling and therefore is not scoreable."""


class InvalidTransition(ValueError):
    pass


class JobBlocked(RuntimeError):
    """The job is waiting for configuration or an external service, not failed."""


class FactoryAgentGateway(Protocol):
    def architect(self, *, run_id: str, workflow: str) -> Mapping[str, Any]: ...

    def build_dataset(
        self, *, run_id: str, workflow: str, skill_contract: Mapping[str, Any],
        feedback: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...

    def evaluate(
        self, *, run_id: str, deterministic_report: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class SkillIntegrator(Protocol):
    def integrate(
        self, *, job: Mapping[str, Any], skill_contract: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class DatasetSource(Protocol):
    max_rounds: int

    def build(
        self, *, job: Mapping[str, Any], round: int,
        skill_contract: Mapping[str, Any], integration: Mapping[str, Any],
        feedback: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]: ...

    def validate(
        self, *, job: Mapping[str, Any], round: int, dataset: Mapping[str, Any],
        integration: Mapping[str, Any], skill_contract: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class TrainingBackend(Protocol):
    def preflight(
        self, *, job: Mapping[str, Any], round: int, dataset: Mapping[str, Any],
        integration: Mapping[str, Any],
    ) -> None: ...

    def train(
        self, *, job: Mapping[str, Any], round: int, dataset: Mapping[str, Any],
        integration: Mapping[str, Any],
        on_progress: Callable[[Mapping[str, Any]], None],
    ) -> Mapping[str, Any]: ...


class AdapterEvaluator(Protocol):
    def evaluate(
        self, *, job: Mapping[str, Any], round: int, training: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class AdapterPublisher(Protocol):
    def promote(
        self, *, job: Mapping[str, Any], training: Mapping[str, Any],
        evaluation: Mapping[str, Any], skill_contract: Mapping[str, Any],
    ) -> str: ...


class JobStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_job(
        self, *, workflow: str, idempotency_key: str, skill_id: str | None = None,
        job_id: str | None = None, created_at: str | None = None,
        route_decision: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        workflow = _required(workflow, "workflow")
        idempotency_key = _required(idempotency_key, "idempotency_key")
        skill_id = skill_id.strip() if skill_id else None
        job_id = job_id or str(uuid.uuid4())
        now = _timestamp(created_at)
        workflow_hash = hashlib.sha256(workflow.encode()).hexdigest()
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
                    ) VALUES (?, ?, NULL, 'queued', ?, ?)
                    """,
                    (str(uuid.uuid4()), job_id, now, _json({"stage": "jev_router"})),
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
        if route_decision:
            self.put_artifact(job_id, "route_decision.json", dict(route_decision))
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
                "SELECT name, sha256, created_at FROM job_artifacts WHERE job_id = ? ORDER BY name",
                (job_id,),
            ).fetchall()
            attempts = connection.execute(
                """
                SELECT stage, round, status, claimed_at, completed_at
                FROM job_stage_attempts WHERE job_id = ? ORDER BY claimed_at, stage, round
                """,
                (job_id,),
            ).fetchall()
            live = connection.execute(
                """
                SELECT name, content_json FROM job_artifacts
                WHERE job_id = ? AND name IN (
                    'route_decision.json', 'progress.json', 'validation.json',
                    'deterministic_evaluation.json', 'publish_result.json',
                    'devin_session.json', 'devin_result.json', 'integration.json',
                    'skill_spec.json', 'output_schema.json', 'evaluation_plan.json',
                    'cancellation.json'
                )
                """,
                (job_id,),
            ).fetchall()
        job = dict(row)
        job["events"] = [_event_dict(item) for item in events]
        job["dataset_round"] = 1 + sum(
            item["from_status"] == "evaluating" and item["to_status"] == "building_dataset"
            for item in job["events"]
        )
        job["artifacts"] = [dict(item) for item in artifacts]
        job["stage_attempts"] = [dict(item) for item in attempts]
        for item in live:
            key = item["name"].removesuffix(".json").replace("route_decision", "route")
            job[key] = json.loads(item["content_json"])
        training_name = _round_name("training.json", job["dataset_round"])
        if any(item["name"] == training_name for item in job["artifacts"]):
            job["training"] = _training_summary(self.get_artifact(job_id, training_name))
        job["agent_outputs"] = _agent_outputs(job)
        return job

    def list_jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connect() as connection:
            ids = [row["job_id"] for row in connection.execute(
                "SELECT job_id FROM jobs ORDER BY created_at DESC, job_id LIMIT ?", (limit,)
            ).fetchall()]
        return [self.get_job(job_id) for job_id in ids]

    def transition(
        self, job_id: str, to_status: str, *, occurred_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._connect(immediate=True) as connection:
            previous = _apply_transition(connection, job_id, to_status, _timestamp(occurred_at), _json(metadata))
        _log_transition(job_id, previous, to_status)
        return self.get_job(job_id)

    def record_progress(self, job_id: str, progress: Mapping[str, Any]) -> None:
        if self.get_job(job_id)["status"] in {"cancelled", "failed", "ready"}:
            return
        now = _timestamp()
        self.set_artifact(job_id, "progress.json", _safe_progress(progress), created_at=now)
        with self._connect(immediate=True) as connection:
            current = connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if current is None:
                raise JobNotFound(job_id)
            connection.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?", (now, job_id))
            connection.execute(
                """
                INSERT INTO job_events(
                    event_id, job_id, from_status, to_status, occurred_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(uuid.uuid4()), job_id, current["status"], current["status"], now,
                 _json({"progress": _safe_progress(progress)})),
            )

    def fail(
        self, job_id: str, *, code: str, message: str, occurred_at: str | None = None,
    ) -> dict[str, Any]:
        now = _timestamp(occurred_at)
        already_failed = False
        with self._connect(immediate=True) as connection:
            current = connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if current is None:
                raise JobNotFound(job_id)
            from_status = current["status"]
            if from_status in {"failed", "cancelling", "cancelled"}:
                already_failed = True
            elif "failed" not in ALLOWED_TRANSITIONS[from_status]:
                raise InvalidTransition(f"cannot fail a job in {from_status}")
            if not already_failed:
                connection.execute(
                    """
                    UPDATE jobs SET status='failed', failure_code=?, failure_message=?, updated_at=?
                    WHERE job_id=?
                    """,
                    (_required(code, "code"), _required(message, "message")[:1000], now, job_id),
                )
                connection.execute(
                    """
                    INSERT INTO job_events(event_id, job_id, from_status, to_status, occurred_at, metadata_json)
                    VALUES (?, ?, ?, 'failed', ?, ?)
                    """,
                    (str(uuid.uuid4()), job_id, from_status, now, _json({"code": code})),
                )
        if not already_failed:
            _log_transition(job_id, from_status, "failed")
        return self.get_job(job_id)

    def check_cancelled(self, job_id: str) -> None:
        if self.get_job(job_id)["status"] in {"cancelling", "cancelled"}:
            raise BuildCancelled("Build stopped at user request")

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        now = _timestamp()
        with self._connect(immediate=True) as connection:
            row = connection.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(job_id)
            if row["status"] not in {"ready", "failed", "cancelling", "cancelled"}:
                _apply_transition(connection, job_id, "cancelling", now, _json({"requested_by": "user"}))
        log_event("job_cancel_requested", job_id=job_id)
        return self.get_job(job_id)

    def finish_cancel(self, job_id: str) -> dict[str, Any]:
        with self._connect(immediate=True) as connection:
            row = connection.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise JobNotFound(job_id)
            active = connection.execute("SELECT 1 FROM job_stage_attempts WHERE job_id=? AND status='running'", (job_id,)).fetchone()
            if row["status"] == "cancelling" and active is None:
                _apply_transition(connection, job_id, "cancelled", _timestamp(), _json({"cleanup_confirmed": True}))
        return self.get_job(job_id)

    def retry_failed(self, job_id: str) -> dict[str, Any]:
        now = _timestamp()
        with self._connect(immediate=True) as connection:
            current = connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if current is None:
                raise JobNotFound(job_id)
            if current["status"] != "failed":
                raise InvalidTransition("only a failed job can be retried")
            failed_event = connection.execute(
                """
                SELECT from_status FROM job_events
                WHERE job_id=? AND to_status='failed'
                ORDER BY sequence DESC LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if failed_event is None or failed_event["from_status"] not in JOB_STAGES:
                raise JobConflict("failed job has no resumable stage")
            attempt = connection.execute(
                """
                SELECT stage, round FROM job_stage_attempts
                WHERE job_id=? AND status='failed'
                ORDER BY claimed_at DESC LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if attempt is not None:
                connection.execute(
                    """
                    UPDATE job_stage_attempts
                    SET status='interrupted', completed_at=?
                    WHERE job_id=? AND stage=? AND round=? AND status='failed'
                    """,
                    (now, job_id, attempt["stage"], attempt["round"]),
                )
            resume_status = failed_event["from_status"]
            if resume_status == "training":
                # A failed worker no longer owns the training claim. Resume at
                # the claimable state so the factory runner can start a fresh
                # adapter_training attempt instead of treating it as active.
                resume_status = "provisioning_gpu"
            connection.execute(
                """
                UPDATE jobs SET status=?, failure_code=NULL, failure_message=NULL, updated_at=?
                WHERE job_id=?
                """,
                (resume_status, now, job_id),
            )
            connection.execute(
                """
                INSERT INTO job_events(
                    event_id, job_id, from_status, to_status, occurred_at, metadata_json
                ) VALUES (?, ?, 'failed', ?, ?, ?)
                """,
                (str(uuid.uuid4()), job_id, resume_status, now, _json({"retry": True})),
            )
        return self.get_job(job_id)

    def repair_truncated_evaluation(self, job_id: str) -> dict[str, Any]:
        """Resume at benchmark evaluation without repeating data generation or training."""
        now = _timestamp()
        with self._connect(immediate=True) as connection:
            current = connection.execute(
                "SELECT status, failure_code, failure_message "
                "FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if current is None:
                raise JobNotFound(job_id)
            is_truncation = current["failure_code"] == "EvaluationOutputTruncated" or (
                current["failure_code"] in {"ValueError", "RuntimeError"}
                and "evaluation_output_truncated" in (current["failure_message"] or "")
            )
            if current["status"] != "failed" or not is_truncation:
                raise InvalidTransition("only a truncated evaluation can use benchmark-only repair")
            attempt = connection.execute(
                """
                SELECT round FROM job_stage_attempts
                WHERE job_id=? AND stage='independent_evaluator'
                  AND status IN ('failed', 'completed')
                ORDER BY round DESC, claimed_at DESC LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if attempt is None:
                raise JobConflict("truncated evaluation has no resumable evaluator attempt")
            # Re-open only the evaluator claim. This is the same recovery model
            # used by retry_failed(), and preserves the original event history
            # while allowing claim_stage() to reacquire the existing round.
            connection.execute(
                """
                UPDATE job_stage_attempts
                SET status='interrupted', completed_at=?
                WHERE job_id=? AND stage='independent_evaluator' AND round=?
                  AND status IN ('failed', 'completed')
                """,
                (now, job_id, attempt["round"]),
            )
            connection.execute(
                """
                UPDATE jobs SET status='evaluating', failure_code=NULL, failure_message=NULL,
                                updated_at=? WHERE job_id=?
                """,
                (now, job_id),
            )
            connection.execute(
                """
                INSERT INTO job_events(event_id, job_id, from_status, to_status, occurred_at, metadata_json)
                VALUES (?, ?, 'failed', 'evaluating', ?, ?)
                """,
                (str(uuid.uuid4()), job_id, now, _json({
                    "benchmark_only_repair": True,
                    "training_skipped": True,
                })),
            )
        return self.get_job(job_id)

    def put_artifact(
        self, job_id: str, name: str, value: Any, *, created_at: str | None = None,
    ) -> dict[str, Any]:
        name = _required(name, "name")
        encoded = _json(value)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        now = _timestamp(created_at)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO job_artifacts(job_id, name, content_json, sha256, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (job_id, name, encoded, digest, now),
                )
        except sqlite3.IntegrityError as error:
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT sha256, created_at FROM job_artifacts WHERE job_id=? AND name=?",
                    (job_id, name),
                ).fetchone()
            if existing is not None and existing["sha256"] == digest:
                return {"name": name, "sha256": digest, "created_at": existing["created_at"]}
            raise JobConflict(f"artifact already exists with different content: {name}") from error
        return {"name": name, "sha256": digest, "created_at": now}

    def set_artifact(
        self, job_id: str, name: str, value: Any, *, created_at: str | None = None,
    ) -> dict[str, Any]:
        name = _required(name, "name")
        encoded = _json(value)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        now = _timestamp(created_at)
        with self._connect(immediate=True) as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone() is None:
                raise JobNotFound(job_id)
            connection.execute(
                """
                INSERT INTO job_artifacts(job_id, name, content_json, sha256, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id, name) DO UPDATE SET
                    content_json=excluded.content_json, sha256=excluded.sha256,
                    created_at=excluded.created_at
                """,
                (job_id, name, encoded, digest, now),
            )
        return {"name": name, "sha256": digest, "created_at": now}

    def set_skill_id(self, job_id: str, skill_id: str) -> None:
        value = _required(skill_id, "skill_id")
        with self._connect(immediate=True) as connection:
            row = connection.execute(
                "SELECT skill_id FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise JobNotFound(job_id)
            if row["skill_id"] not in (None, value):
                raise JobConflict("Skill Architect changed an existing skill ID")
            connection.execute(
                "UPDATE jobs SET skill_id=?, updated_at=? WHERE job_id=?",
                (value, _timestamp(), job_id),
            )

    def get_artifact(self, job_id: str, name: str) -> Any:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT content_json FROM job_artifacts WHERE job_id=? AND name=?", (job_id, name)
            ).fetchone()
        if row is None:
            raise JobNotFound(f"{job_id}/{name}")
        return json.loads(row["content_json"])

    def claim_stage(
        self, job_id: str, stage: str, *, round: int = 1,
        enter_status: str | None = None, claimed_at: str | None = None,
    ) -> None:
        now = _timestamp(claimed_at)
        try:
            with self._connect(immediate=True) as connection:
                current = connection.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                if current is None:
                    raise JobNotFound(job_id)
                if current["status"] in {"ready", "failed", "cancelling", "cancelled"}:
                    raise BuildCancelled("Job no longer accepts stage claims")
                connection.execute(
                    """
                    INSERT INTO job_stage_attempts(job_id, stage, round, status, claimed_at, completed_at)
                    VALUES (?, ?, ?, 'running', ?, NULL)
                    """,
                    (job_id, _required(stage, "stage"), _round(round), now),
                )
                previous = (_apply_transition(connection, job_id, enter_status, now, _json({"round": round}))
                            if enter_status else None)
        except sqlite3.IntegrityError as error:
            with self._connect(immediate=True) as connection:
                attempt = connection.execute(
                    """
                    SELECT status FROM job_stage_attempts
                    WHERE job_id=? AND stage=? AND round=?
                    """,
                    (job_id, stage, _round(round)),
                ).fetchone()
                if attempt is None or attempt["status"] != "interrupted":
                    raise JobConflict(
                        f"stage was already claimed: {stage} round {round}"
                    ) from error
                current = connection.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                if current is None or current["status"] in {"ready", "failed", "cancelling", "cancelled"}:
                    raise BuildCancelled("Job no longer accepts stage claims")
                connection.execute(
                    """
                    UPDATE job_stage_attempts
                    SET status='running', claimed_at=?, completed_at=NULL
                    WHERE job_id=? AND stage=? AND round=? AND status='interrupted'
                    """,
                    (now, job_id, stage, _round(round)),
                )
                previous = (_apply_transition(connection, job_id, enter_status, now, _json({"round": round}))
                            if enter_status else None)
        _log_transition(job_id, previous, enter_status)

    def recover_interrupted_stages(self) -> int:
        """Release stage claims abandoned when the single controller was replaced."""
        now = _timestamp()
        with self._connect(immediate=True) as connection:
            rows = connection.execute(
                """
                SELECT a.job_id, a.stage, a.round, j.status
                FROM job_stage_attempts a
                JOIN jobs j ON j.job_id = a.job_id
                WHERE a.status='running' AND j.status NOT IN ('ready', 'failed', 'cancelled')
                """
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE job_stage_attempts SET status='interrupted', completed_at=?
                    WHERE job_id=? AND stage=? AND round=? AND status='running'
                    """,
                    (now, row["job_id"], row["stage"], row["round"]),
                )
                connection.execute(
                    """
                    INSERT INTO job_events(
                        event_id, job_id, from_status, to_status, occurred_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()), row["job_id"], row["status"], row["status"], now,
                        _json({
                            "stage": row["stage"], "round": row["round"],
                            "recovered_after_controller_restart": True,
                        }),
                    ),
                )
        return len(rows)

    def complete_stage(
        self, job_id: str, stage: str, *, status: str, round: int = 1,
        next_status: str | None = None, metadata: Mapping[str, Any] | None = None,
        completed_at: str | None = None,
    ) -> None:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("stage attempt status must be completed or failed")
        now = _timestamp(completed_at)
        with self._connect(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE job_stage_attempts SET status=?, completed_at=?
                WHERE job_id=? AND stage=? AND round=? AND status='running'
                """,
                (status, now, job_id, stage, _round(round)),
            )
            if cursor.rowcount != 1:
                raise JobConflict(f"stage is not running: {stage}")
            previous = (_apply_transition(connection, job_id, next_status, now, _json(metadata))
                        if next_status else None)
        _log_transition(job_id, previous, next_status)

    def register_adapter(
        self, *, adapter_id: str, skill_id: str, version: str, base_model: str,
        artifact_uri: str, metadata: Mapping[str, Any] | None = None,
        created_at: str | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        now = _timestamp(created_at)
        values = (
            _required(adapter_id, "adapter_id"), _required(skill_id, "skill_id"),
            _required(version, "version"), _required(base_model, "base_model"),
            _required(artifact_uri, "artifact_uri"), _json(metadata), now,
        )
        try:
            with self._connect(immediate=True) as connection:
                _guard_publication(connection, job_id)
                connection.execute(
                    """
                    INSERT INTO adapters(adapter_id, skill_id, version, base_model, artifact_uri,
                                         metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
        except sqlite3.IntegrityError as error:
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT * FROM adapters WHERE adapter_id=?", (adapter_id,)
                ).fetchone()
            if existing is not None and tuple(existing) == values:
                return _adapter_dict(existing)
            raise JobConflict(f"adapter already exists: {adapter_id}") from error
        return self.get_adapter(adapter_id)

    def update_adapter(
        self, *, adapter_id: str, artifact_uri: str, metadata: Mapping[str, Any], job_id: str | None = None,
    ) -> dict[str, Any]:
        with self._connect(immediate=True) as connection:
            _guard_publication(connection, job_id)
            cursor = connection.execute(
                "UPDATE adapters SET artifact_uri=?, metadata_json=? WHERE adapter_id=?",
                (_required(artifact_uri, "artifact_uri"), _json(metadata), adapter_id),
            )
            if cursor.rowcount != 1:
                raise JobNotFound(adapter_id)
        return self.get_adapter(adapter_id)

    def get_adapter(self, adapter_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM adapters WHERE adapter_id=?", (adapter_id,)).fetchone()
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
                CREATE TABLE IF NOT EXISTS job_schema_info (version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE,
                    workflow TEXT NOT NULL, workflow_hash TEXT NOT NULL, skill_id TEXT,
                    status TEXT NOT NULL, failure_code TEXT, failure_message TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_status_updated ON jobs(status, updated_at);
                CREATE TABLE IF NOT EXISTS job_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id), from_status TEXT,
                    to_status TEXT NOT NULL, occurred_at TEXT NOT NULL, metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_artifacts (
                    job_id TEXT NOT NULL REFERENCES jobs(job_id), name TEXT NOT NULL,
                    content_json TEXT NOT NULL, sha256 TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, name)
                );
                CREATE TABLE IF NOT EXISTS job_stage_attempts (
                    job_id TEXT NOT NULL REFERENCES jobs(job_id), stage TEXT NOT NULL,
                    round INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL,
                    claimed_at TEXT NOT NULL, completed_at TEXT,
                    PRIMARY KEY(job_id, stage, round)
                );
                CREATE TABLE IF NOT EXISTS adapters (
                    adapter_id TEXT PRIMARY KEY, skill_id TEXT NOT NULL, version TEXT NOT NULL,
                    base_model TEXT NOT NULL, artifact_uri TEXT NOT NULL,
                    metadata_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(skill_id, version)
                );
                """
            )
            if not connection.in_transaction:
                connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT version FROM job_schema_info").fetchone()
            if row is None:
                connection.execute("INSERT INTO job_schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            elif row["version"] == 1:
                _migrate_stage_attempts(connection)
                connection.execute("UPDATE job_schema_info SET version=?", (SCHEMA_VERSION,))
            elif row["version"] != SCHEMA_VERSION:
                raise RuntimeError(f"unsupported job schema: {row['version']}")

    @contextmanager
    def _connect(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
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
        self, store: JobStore, ledger: UsageLedger, agents: FactoryAgentGateway | None = None,
        *, integrator: SkillIntegrator | None = None, dataset_source: DatasetSource | None = None,
        trainer: TrainingBackend | None = None, evaluator: AdapterEvaluator | None = None,
        publisher: AdapterPublisher | None = None, max_dataset_rounds: int = 3,
    ):
        self.store = store
        self.ledger = ledger
        self.agents = agents
        self.integrator = integrator
        self.dataset_source = dataset_source
        self.trainer = trainer
        self.evaluator = evaluator
        self.publisher = publisher
        self.max_dataset_rounds = _round(max_dataset_rounds)

    def create_job(
        self, *, workflow: str, idempotency_key: str, skill_id: str | None = None,
        route_decision: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        job = self.store.create_job(
            workflow=workflow, idempotency_key=idempotency_key, skill_id=skill_id,
            route_decision=route_decision,
        )
        self.ledger.create_run(
            run_id=job["job_id"], run_type="adapter_build", job_id=job["job_id"],
            skill_id=job["skill_id"], started_at=job["created_at"],
            metadata={"workflow_hash": job["workflow_hash"]},
        )
        if len(job["events"]) == 1:  # newly created, not an idempotent replay
            emit_job_root(job["job_id"], skill_id=job["skill_id"], route_decision=(route_decision or {}).get("decision"))
            log_event("job_created", job_id=job["job_id"], skill_id=job["skill_id"], next_stage=job["status"])
        return job

    def advance(self, job_id: str) -> dict[str, Any]:
        status = self.store.get_job(job_id)["status"]
        if status in {"ready", "failed", "cancelling", "cancelled"}:
            raise JobBlocked(f"job is {status}")
        if status == "training":
            raise JobConflict("training is already owned by a worker")
        return {
            "queued": self.architect, "architecting": self.architect,
            "integrating": self.integrate, "building_dataset": self.build_dataset,
            "validating": self.validate_dataset, "awaiting_compute": self.authorize_compute,
            "provisioning_gpu": self.train, "evaluating": self.evaluate,
            "publishing": self.publish,
        }[status](job_id)

    def architect(self, job_id: str) -> dict[str, Any]:
        job = self._require(job_id, {"queued", "architecting"})
        agents = self._agents()
        def work() -> str:
            result = dict(agents.architect(run_id=job_id, workflow=job["workflow"]))
            required = {"skill_spec", "output_schema", "evaluation_plan"}
            if required - result.keys():
                raise ValueError(f"Skill Architect omitted: {sorted(required - result.keys())}")
            if not all(isinstance(result[name], Mapping) for name in required):
                raise ValueError("Skill Architect outputs must be JSON objects")
            if result["output_schema"].get("type") != "object":
                raise ValueError("Skill Architect output_schema must describe a JSON object")
            try:
                import jsonschema

                jsonschema.Draft202012Validator.check_schema(result["output_schema"])
            except Exception as error:
                raise ValueError(f"Skill Architect produced invalid JSON Schema: {error}") from error
            trainability = result["skill_spec"].get("trainability")
            if trainability is False or str(trainability).lower() in {"false", "unsupported"}:
                raise ValueError("Skill Architect marked this workflow as not trainable")
            for name in required:
                self.store.put_artifact(job_id, f"{name}.json", result[name])
            self.store.set_skill_id(
                job_id, job["skill_id"] or _skill_id(result["skill_spec"], job["workflow_hash"])
            )
            return "integrating"
        return self._stage(job_id, "skill_architect", 1, work, enter_status="architecting")

    def integrate(self, job_id: str) -> dict[str, Any]:
        job = self._require(job_id, {"integrating"})
        if self.integrator is None:
            raise JobBlocked("integrating: Devin is not configured")
        def work() -> str:
            result = dict(self.integrator.integrate(job=job, skill_contract=self._contract(job_id)))
            if not result.get("session_id"):
                raise ValueError("Devin integration did not produce session evidence")
            self.store.put_artifact(job_id, "integration.json", result)
            return "building_dataset"
        return self._stage(job_id, "skill_integration", 1, work)

    def complete_integration(
        self, job_id: str, *, commit_sha: str, test_summary: Mapping[str, Any]
    ) -> dict[str, Any]:
        self._require(job_id, {"integrating"})
        self.store.put_artifact(job_id, "integration.json", {
            "commit_sha": _required(commit_sha, "commit_sha"), "tests": dict(test_summary),
        })
        return self.store.transition(job_id, "building_dataset", metadata={"manual": True})

    def build_dataset(self, job_id: str) -> dict[str, Any]:
        job = self._require(job_id, {"building_dataset"})
        round = job["dataset_round"]
        if self.dataset_source is None:
            raise JobBlocked("building_dataset: dataset source is not configured")
        def work() -> str:
            dataset = dict(self.dataset_source.build(
                job=job, round=round, skill_contract=self._contract(job_id),
                integration=self.store.get_artifact(job_id, "integration.json"),
                feedback=self._feedback(job_id, round),
            ))
            if round > 1:
                dataset["test"] = self.store.get_artifact(job_id, "dataset.json")["test"]
            self.store.put_artifact(job_id, _round_name("dataset.json", round), dataset)
            return "validating"
        return self._stage(job_id, "data_builder", round, work)

    def validate_dataset(self, job_id: str) -> dict[str, Any]:
        job = self._require(job_id, {"validating"})
        round = job["dataset_round"]
        if self.dataset_source is None:
            raise JobBlocked("validating: deterministic validator is not configured")
        def work() -> str:
            report = dict(self.dataset_source.validate(
                job=job, round=round,
                dataset=self.store.get_artifact(job_id, _round_name("dataset.json", round)),
                integration=self.store.get_artifact(job_id, "integration.json"),
                skill_contract=self._contract(job_id),
            ))
            self.store.put_artifact(job_id, _round_name("validation.json", round), report)
            self.store.set_artifact(job_id, "validation.json", report)
            if report.get("passed") is not True:
                raise ValueError("dataset validation did not pass")
            return "awaiting_compute"
        return self._stage(job_id, "dataset_validator", round, work)

    def authorize_compute(self, job_id: str) -> dict[str, Any]:
        job = self._require(job_id, {"awaiting_compute"})
        round = job["dataset_round"]
        if self.trainer is None:
            raise JobBlocked("awaiting_compute: training backend is not configured")
        with FactoryStage(job_id, "compute_authorization", round=round, op="dapta.factory.stage",
                          name="factory compute_authorization") as telemetry:
            try:
                with span("compute.preflight", "Authorize adapter compute", job_id=job_id, round=round):
                    self.trainer.preflight(
                        job=job, round=round,
                        dataset=self.store.get_artifact(job_id, _round_name("dataset.json", round)),
                        integration=self.store.get_artifact(job_id, "integration.json"),
                    )
            except JobBlocked as blocked:
                log_event("stage_blocked", level="warning", **telemetry.fields, reason=str(blocked)[:200])
                raise
            return self.store.transition(job_id, "provisioning_gpu", metadata={
                "round": round, "budget_checked": True,
            })

    def train(self, job_id: str) -> dict[str, Any]:
        job = self._require(job_id, {"provisioning_gpu"})
        round = job["dataset_round"]
        if self.trainer is None:
            raise JobBlocked("provisioning_gpu: training backend is not configured")
        def work() -> str:
            result = dict(self.trainer.train(
                job=job, round=round,
                dataset=self.store.get_artifact(job_id, _round_name("dataset.json", round)),
                integration=self.store.get_artifact(job_id, "integration.json"),
                on_progress=lambda item: self.store.record_progress(job_id, item),
            ))
            required = {"version", "adapter_dir", "base_model", "evaluation"}
            if required - result.keys():
                raise ValueError(f"training backend omitted: {sorted(required - result.keys())}")
            self.store.put_artifact(job_id, _round_name("training.json", round), result)
            return "evaluating"
        return self._stage(job_id, "adapter_training", round, work, enter_status="training")

    def evaluate(self, job_id: str) -> dict[str, Any]:
        job = self._require(job_id, {"evaluating"})
        round = job["dataset_round"]
        if self.evaluator is None:
            raise JobBlocked("evaluating: deterministic evaluator is not configured")
        def work() -> str:
            training, training_round = self._latest_round_artifact(job_id, "training.json", round)
            training = dict(training)
            training["_source_round"] = training_round
            report = dict(self.evaluator.evaluate(
                job=job, round=round,
                training=training,
            ))
            if not isinstance(report.get("passed"), bool):
                raise ValueError("deterministic evaluation must include boolean passed")
            self.store.put_artifact(job_id, _round_name("deterministic_evaluation.json", round), report)
            self.store.set_artifact(job_id, "deterministic_evaluation.json", report)
            self.store.put_artifact(
                job_id, _round_name("qualitative_evaluation.json", round),
                self._qualitative(job_id, report),
            )
            if report["passed"]:
                return "publishing"
            if report.get("repairable") is False:
                reason = str(report.get("failure_reason") or "evaluation configuration failed")
                if reason == "evaluation_output_truncated":
                    raise EvaluationOutputTruncated(reason)
                raise ValueError(reason)
            if round < self._round_limit():
                return "building_dataset"
            raise ValueError(f"adapter failed evaluation after {round} data rounds")
        return self._stage(job_id, "independent_evaluator", round, work)

    def publish(self, job_id: str) -> dict[str, Any]:
        job = self._require(job_id, {"publishing"})
        round = job["dataset_round"]
        if self.publisher is None:
            raise JobBlocked("publishing: Hugging Face publisher is not configured")
        training, _ = self._latest_round_artifact(job_id, "training.json", round)
        evaluation = self.store.get_artifact(job_id, _round_name("deterministic_evaluation.json", round))
        contract = self._contract(job_id)
        contract["integration"] = self.store.get_artifact(job_id, "integration.json")
        def work() -> str:
            uri = self.publisher.promote(
                job=job, training=training, evaluation=evaluation, skill_contract=contract,
            )
            check_cancelled()
            skill_id = job["skill_id"] or _skill_id(contract["skill_spec"], job["workflow_hash"])
            adapter_id = training.get("adapter_id") or f"generated/{skill_id}/{training['version']}"
            metadata = {
                "name": contract["skill_spec"].get("name", skill_id),
                "description": contract["skill_spec"].get("description", job["workflow"][:240]),
                "job_id": job_id, "dataset_round": round,
                "benchmark_accuracy": evaluation.get("conditions", {}).get(
                    "lora_short_prompt", {}).get("field_f1", 0),
                "benchmark_status": "verified" if evaluation.get("passed") else "needs_repair",
                "evaluation_verified": bool(evaluation.get("passed")),
                "training_cost_usd": training.get("training_cost_usd", "0"),
            }
            try:
                adapter = self.store.register_adapter(
                    adapter_id=adapter_id, skill_id=skill_id, version=training["version"],
                    base_model=training["base_model"], artifact_uri=uri, metadata=metadata, job_id=job_id,
                )
            except JobConflict:
                existing = self.store.get_adapter(adapter_id)
                if (
                    existing["skill_id"] != skill_id
                    or existing["version"] != training["version"]
                    or existing["base_model"] != training["base_model"]
                ):
                    raise
                adapter = self.store.update_adapter(
                    adapter_id=adapter_id, artifact_uri=uri, metadata=metadata, job_id=job_id,
                )
            self.store.set_artifact(job_id, "publish_result.json", {
                "adapter": adapter, "artifact_uri": uri,
            })
            return "ready"
        return self._stage(job_id, "registry_publish", round, work)

    def _stage(
        self, job_id: str, stage: str, round: int, work: Callable[[], str],
        *, enter_status: str | None = None,
    ) -> dict[str, Any]:
        operation = {
            "skill_architect": "gen_ai.skill_architect",
            "skill_integration": "agent.devin.integrate",
            "data_builder": "gen_ai.data_builder",
            "dataset_validator": "dataset.validate",
            "adapter_training": "model.train_qlora",
            "independent_evaluator": "model.evaluate",
            "registry_publish": "registry.publish",
        }.get(stage, f"dapta.{stage}")
        with FactoryStage(job_id, stage, round=round, op="dapta.factory.stage", name=f"factory {stage}") as telemetry:
            with span("db.sql.transaction", "claim stage", **telemetry.fields):
                self.store.claim_stage(job_id, stage, round=round, enter_status=enter_status)
            self.ledger.record_run_event(
                run_id=job_id, stage=stage, status="started", metadata={"round": round},
            )
            log_event("stage_started", **telemetry.fields)
            started = time.monotonic()
            try:
                with cancellation_scope(lambda: self.store.check_cancelled(job_id)):
                    check_cancelled()
                    if stage in {"skill_integration", "adapter_training"}:
                        next_status = work()
                    else:
                        with span(operation, stage.replace("_", " ").title(), job_id=job_id, round=round):
                            next_status = work()
                    check_cancelled()
                with span("db.sql.transaction", "complete stage", **telemetry.fields):
                    self.store.complete_stage(
                        job_id, stage, status="completed", round=round, next_status=next_status,
                        metadata={"round": round},
                    )
                self.ledger.record_run_event(
                    run_id=job_id, stage=stage, status="completed",
                    metadata={"round": round, "next_status": next_status},
                )
                log_event("stage_completed", **telemetry.fields, next_stage=next_status,
                          duration_ms=round_ms(started))
                return self.store.get_job(job_id)
            except Exception as error:
                if self.store.get_job(job_id)["status"] in {"cancelling", "cancelled"}:
                    try:
                        self.store.complete_stage(job_id, stage, status="cancelled", round=round)
                    except JobConflict:
                        pass
                    if isinstance(error, BuildCleanupError):
                        self.store.set_artifact(job_id, "cancellation.json", {"cleanup_error": "Inference reload failed. Retry Stop build to restore service.", "inference_restore_required": True})
                    self.ledger.record_run_event(run_id=job_id, stage=stage, status="cancelled", metadata={"round": round})
                    log_event("stage_cancelled", **telemetry.fields)
                    raise BuildCancelled("Build stopped at user request") from None
                _finish_attempt(self.store, job_id, stage, round)
                if isinstance(error, JobBlocked):
                    log_event("stage_blocked", level="warning", **telemetry.fields, reason=str(error)[:200])
                else:
                    self.store.fail(job_id, code=type(error).__name__, message=str(error) or repr(error))
                    self.ledger.record_run_event(
                        run_id=job_id, stage=stage, status="failed",
                        metadata={"round": round, "error_type": type(error).__name__},
                    )
                    log_event("stage_failed", level="error", **telemetry.fields, duration_ms=round_ms(started),
                              error_category=error_category(error), error_type=type(error).__name__)
                raise

    def _qualitative(self, job_id: str, report: Mapping[str, Any]) -> dict[str, Any]:
        try:
            result = dict(self._agents().evaluate(run_id=job_id, deterministic_report=report))
            if {"qualitative_findings", "failures"} - result.keys():
                raise ValueError("independent evaluator returned an incomplete review")
            return result
        except Exception as error:
            return {"degraded": True, "error_type": type(error).__name__}

    def _contract(self, job_id: str) -> dict[str, Any]:
        return {name: self.store.get_artifact(job_id, f"{name}.json") for name in (
            "skill_spec", "output_schema", "evaluation_plan",
        )}

    def _latest_round_artifact(
        self, job_id: str, name: str, round: int
    ) -> tuple[dict[str, Any], int]:
        for candidate in range(round, 0, -1):
            try:
                return self.store.get_artifact(job_id, _round_name(name, candidate)), candidate
            except JobNotFound:
                continue
        raise JobNotFound(f"{job_id}:{name}")

    def _feedback(self, job_id: str, round: int) -> dict[str, Any] | None:
        if round == 1:
            return None
        previous = self.store.get_artifact(
            job_id, _round_name("deterministic_evaluation.json", round - 1),
        )
        return {"round": round - 1, "dev_feedback": previous.get("dev_feedback", {})}

    def _round_limit(self) -> int:
        source = getattr(self.dataset_source, "max_rounds", self.max_dataset_rounds)
        return min(self.max_dataset_rounds, source)

    def _require(self, job_id: str, allowed: set[str]) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job["status"] not in allowed:
            raise InvalidTransition(
                f"job must be in {' or '.join(sorted(allowed))}, not {job['status']}",
            )
        return job

    def _agents(self) -> FactoryAgentGateway:
        if self.agents is None:
            raise JobBlocked("factory agents are not configured")
        return self.agents


def _guard_publication(connection: sqlite3.Connection, job_id: str | None) -> None:
    if job_id is None:
        return
    row = connection.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None or row["status"] in {"cancelling", "cancelled"}:
        raise BuildCancelled("Cancelled build cannot publish a registry entry")


def _apply_transition(
    connection: sqlite3.Connection, job_id: str, to_status: str, now: str, metadata: str,
) -> str | None:
    """Move the job; returns the previous status, or None when it was already there."""
    if to_status not in JOB_STAGES:
        raise ValueError(f"unknown job stage: {to_status}")
    current = connection.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if current is None:
        raise JobNotFound(job_id)
    from_status = current["status"]
    if from_status == to_status:
        return None
    if to_status not in ALLOWED_TRANSITIONS[from_status]:
        raise InvalidTransition(f"cannot move job from {from_status} to {to_status}")
    connection.execute("UPDATE jobs SET status=?, updated_at=? WHERE job_id=?", (to_status, now, job_id))
    connection.execute(
        """
        INSERT INTO job_events(event_id, job_id, from_status, to_status, occurred_at, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (str(uuid.uuid4()), job_id, from_status, to_status, now, metadata),
    )
    return from_status


def _log_transition(job_id: str, previous: str | None, next_stage: str | None) -> None:
    if previous and next_stage:
        log_event("stage_transitioned", job_id=job_id, previous_stage=previous, next_stage=next_stage)


def _migrate_stage_attempts(connection: sqlite3.Connection) -> None:
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(job_stage_attempts)")}
    if "round" in columns:
        return
    connection.execute("ALTER TABLE job_stage_attempts RENAME TO job_stage_attempts_v1")
    connection.execute(
        """
        CREATE TABLE job_stage_attempts (
            job_id TEXT NOT NULL REFERENCES jobs(job_id), stage TEXT NOT NULL,
            round INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL,
            claimed_at TEXT NOT NULL, completed_at TEXT,
            PRIMARY KEY(job_id, stage, round)
        )
        """,
    )
    connection.execute(
        """
        INSERT INTO job_stage_attempts(job_id, stage, round, status, claimed_at, completed_at)
        SELECT job_id, stage, 1, status, claimed_at, completed_at FROM job_stage_attempts_v1
        """,
    )
    connection.execute("DROP TABLE job_stage_attempts_v1")


def round_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def _finish_attempt(store: JobStore, job_id: str, stage: str, round: int) -> None:
    try:
        store.complete_stage(job_id, stage, status="failed", round=round)
    except JobConflict:
        pass


def _training_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"status": "saved"}
    metrics = value.get("metrics") if isinstance(value.get("metrics"), Mapping) else {}
    return {
        key: item for key, item in {
            "adapter_id": value.get("adapter_id"),
            "base_model": value.get("base_model"),
            "version": value.get("version"),
            "training_seconds": value.get("training_seconds") or metrics.get("training_seconds"),
            "training_cost_usd": value.get("training_cost_usd"),
            "peak_gpu_memory_gb": value.get("peak_gpu_memory_gb") or metrics.get("peak_gpu_memory_gb"),
            "mean_gpu_utilization_percent": value.get("mean_gpu_utilization_percent")
            or metrics.get("mean_gpu_utilization_percent"),
        }.items() if item is not None
    }


def _agent_outputs(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

    def add(stage: str, agent: str, summary: str, output: Any) -> None:
        if output is None:
            return
        outputs.append({
            "stage": stage,
            "agent": agent,
            "summary": summary,
            "output": output,
            "source": "Supabase mirror",
        })

    add("queued", "JEV Router", "Routing decision", job.get("route"))
    architect = {
        key: job[key] for key in ("skill_spec", "output_schema", "evaluation_plan")
        if key in job
    }
    add("architecting", "Kimi K3 — Skill Architect", "Skill contract", architect or None)
    devin = job.get("devin_result") or job.get("integration") or job.get("devin_session")
    add("integrating", "Devin — Skill Integration", "Implementation evidence", devin)
    validation = job.get("validation")
    if isinstance(validation, Mapping):
        add("building_dataset", "Kimi K3 — Data Builder", "Generated split counts", {
            "round": validation.get("round"),
            "counts": validation.get("counts"),
        })
    add("validating", "Dataset Validator", "Deterministic validation", validation)
    progress = job.get("progress")
    if isinstance(progress, Mapping) and progress.get("stage") in {
        "awaiting_compute", "provisioning_gpu", "training",
    }:
        add("training", "RunPod Training Worker", "Live compute telemetry", progress)
    add("training", "RunPod Training Worker", "Saved training metrics", job.get("training"))
    add(
        "evaluating", "Independent Evaluator", "Base vs Skill vs LoRA",
        job.get("deterministic_evaluation"),
    )
    add("publishing", "Hugging Face Registry", "Published adapter", job.get("publish_result"))
    return outputs


def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["metadata"] = json.loads(result.pop("metadata_json"))
    return result


def _adapter_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["metadata"] = json.loads(result.pop("metadata_json"))
    return result


def _round_name(name: str, round: int) -> str:
    return name if round == 1 else f"round-{round}/{name}"


def _skill_id(spec: Mapping[str, Any], fallback: str) -> str:
    value = str(spec.get("skill_id") or spec.get("name") or fallback[:12]).lower()
    normalized = "-".join(
        "".join(character if character.isalnum() else " " for character in value).split()
    )[:80]
    return normalized or fallback[:12]


def _safe_progress(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "stage", "message", "current", "total", "percent", "eta_seconds", "loss",
        "gpu_utilization_percent", "gpu_memory_used_gb", "cost_usd",
    }
    return {key: item for key, item in value.items() if key in allowed}


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _round(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("round must be a positive integer")
    return value


def _timestamp(value: str | None = None) -> str:
    moment = datetime.now(timezone.utc) if value is None else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), allow_nan=False)
