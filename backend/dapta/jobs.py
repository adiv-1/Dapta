from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol

from .ledger import UsageLedger


SCHEMA_VERSION = 2
DEFAULT_MAX_DATASET_ROUNDS = 3
SPEND_CONFIRMATION = "PROVISION"

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
    "evaluating": {"publishing", "building_dataset", "failed"},
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


class JobBlocked(RuntimeError):
    """The job cannot advance without outside input; it is not a failure."""


class FactoryAgentGateway(Protocol):
    def architect(self, *, run_id: str, workflow: str) -> Mapping[str, Any]: ...

    # Callers pass feedback only after round 1, so older gateways keep working.
    def build_dataset(
        self,
        *,
        run_id: str,
        workflow: str,
        skill_contract: Mapping[str, Any],
        feedback: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...

    def evaluate(
        self, *, run_id: str, deterministic_report: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class ContractStore(Protocol):
    def get(self, skill_id: str) -> Mapping[str, Any] | None: ...


class SkillIntegrator(Protocol):
    def integrate(
        self, *, job: Mapping[str, Any], skill_contract: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class DatasetSource(Protocol):
    # Sources that cannot vary between rounds (pinned public data) set this to 1.
    max_rounds: int

    def build(
        self,
        *,
        job: Mapping[str, Any],
        round: int,
        skill_contract: Mapping[str, Any],
        integration: Mapping[str, Any],
        feedback: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]: ...

    def validate(
        self,
        *,
        job: Mapping[str, Any],
        round: int,
        dataset: Mapping[str, Any],
        integration: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class TrainingBackend(Protocol):
    # Checks that run before any spend; raising here blocks the job instead of failing it.
    def preflight(
        self,
        *,
        job: Mapping[str, Any],
        round: int,
        dataset: Mapping[str, Any],
        integration: Mapping[str, Any],
    ) -> None: ...

    def train(
        self,
        *,
        job: Mapping[str, Any],
        round: int,
        dataset: Mapping[str, Any],
        integration: Mapping[str, Any],
        on_progress: Callable[[str], None],
    ) -> Mapping[str, Any]: ...


class AdapterPublisher(Protocol):
    # Moves a candidate adapter to its published location and returns the published URI.
    def promote(self, *, job: Mapping[str, Any], training: Mapping[str, Any]) -> str: ...


class AdapterEvaluator(Protocol):
    def evaluate(
        self, *, job: Mapping[str, Any], round: int, training: Mapping[str, Any]
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
                SELECT stage, round, status, claimed_at, completed_at
                FROM job_stage_attempts
                WHERE job_id = ?
                ORDER BY claimed_at, stage, round
                """,
                (job_id,),
            ).fetchall()
        job = dict(row)
        job["events"] = [_event_dict(item) for item in events]
        job["dataset_round"] = 1 + sum(
            1
            for item in job["events"]
            if item["from_status"] == "evaluating" and item["to_status"] == "building_dataset"
        )
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
        with self._connect(immediate=True) as connection:
            _apply_transition(
                connection, job_id, to_status, _timestamp(occurred_at), _json(metadata)
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
        self,
        job_id: str,
        stage: str,
        *,
        round: int = 1,
        enter_status: str | None = None,
        claimed_at: str | None = None,
    ) -> None:
        """Claim a stage; with enter_status the job moves there in the same transaction."""
        stage = _required(stage, "stage")
        round = _round(round)
        now = _timestamp(claimed_at)
        try:
            with self._connect(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO job_stage_attempts(
                        job_id, stage, round, status, claimed_at, completed_at
                    ) VALUES (?, ?, ?, 'running', ?, NULL)
                    """,
                    (job_id, stage, round, now),
                )
                if enter_status is not None:
                    _apply_transition(
                        connection, job_id, enter_status, now, _json({"round": round})
                    )
        except sqlite3.IntegrityError as error:
            with self._connect() as connection:
                job = connection.execute(
                    "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
            if job is None:
                raise JobNotFound(job_id) from error
            raise JobConflict(f"stage was already claimed: {stage} round {round}") from error

    def complete_stage(
        self,
        job_id: str,
        stage: str,
        *,
        status: str,
        round: int = 1,
        next_status: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        completed_at: str | None = None,
    ) -> None:
        """Close a stage attempt; with next_status the job moves on in the same transaction."""
        if status not in {"completed", "failed"}:
            raise ValueError("stage attempt status must be completed or failed")
        now = _timestamp(completed_at)
        with self._connect(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE job_stage_attempts
                SET status = ?, completed_at = ?
                WHERE job_id = ? AND stage = ? AND round = ? AND status = 'running'
                """,
                (status, now, job_id, stage, _round(round)),
            )
            if cursor.rowcount != 1:
                raise JobConflict(f"stage is not running: {stage}")
            if next_status is not None:
                _apply_transition(connection, job_id, next_status, now, _json(metadata))

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
                    round INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL,
                    claimed_at TEXT NOT NULL,
                    completed_at TEXT,
                    PRIMARY KEY(job_id, stage, round)
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
            # Python < 3.12 commits before executescript; reopen so the migration is atomic.
            if not connection.in_transaction:
                connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT version FROM job_schema_info").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO job_schema_info(version) VALUES (?)", (SCHEMA_VERSION,)
                )
            elif row["version"] == 1:
                _migrate_stage_attempts_to_rounds(connection)
                connection.execute(
                    "UPDATE job_schema_info SET version = ?", (SCHEMA_VERSION,)
                )
            elif row["version"] != SCHEMA_VERSION:
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


class DatasetRejected(ValueError):
    pass


class EvaluationFailed(ValueError):
    pass


class AdapterJobOrchestrator:
    """Drives one job through the state machine, one stage per advance() call.

    Each stage claims (job, stage, round) before doing work, so a crashed or
    concurrent caller can never repeat a paid stage. Missing configuration or a
    pending human decision raises JobBlocked and leaves the job untouched.
    """

    def __init__(
        self,
        store: JobStore,
        ledger: UsageLedger,
        agents: FactoryAgentGateway | None = None,
        *,
        contracts: ContractStore | None = None,
        integrator: SkillIntegrator | None = None,
        dataset_source: DatasetSource | None = None,
        trainer: TrainingBackend | None = None,
        evaluator: AdapterEvaluator | None = None,
        publisher: AdapterPublisher | None = None,
        max_dataset_rounds: int = DEFAULT_MAX_DATASET_ROUNDS,
    ):
        self.store = store
        self.ledger = ledger
        self.agents = agents
        self.contracts = contracts
        self.integrator = integrator
        self.dataset_source = dataset_source
        self.trainer = trainer
        self.evaluator = evaluator
        self.publisher = publisher
        self.max_dataset_rounds = _round(max_dataset_rounds)

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

    def advance(self, job_id: str) -> dict[str, Any]:
        status = self.store.get_job(job_id)["status"]
        if status in {"ready", "failed"}:
            raise JobBlocked(f"job is {status}")
        if status == "training":
            raise JobConflict(
                "training is running or was interrupted; check the Vultr allocation before resuming"
            )
        steps = {
            "queued": self.architect,
            "architecting": self.architect,
            "integrating": self.integrate,
            "building_dataset": self.build_dataset,
            "validating": self.validate_dataset,
            "provisioning_gpu": self.train,
            "evaluating": self.evaluate,
            "publishing": self.publish,
        }
        return steps[status](job_id)

    def run(self, job_id: str, *, max_steps: int = 50) -> tuple[dict[str, Any], str]:
        """Advance until the job finishes or blocks; returns the job and the reason."""
        for _ in range(max_steps):
            try:
                self.advance(job_id)
            except JobBlocked as blocked:
                return self.store.get_job(job_id), str(blocked)
        raise RuntimeError(f"job did not settle within {max_steps} steps")

    def architect(self, job_id: str) -> dict[str, Any]:
        job = self._require_status(job_id, {"queued", "architecting"})
        contract = None
        if self.contracts is not None and job["skill_id"]:
            try:
                contract = self.contracts.get(job["skill_id"])
            except Exception as error:
                self.store.fail(job_id, code=type(error).__name__, message=str(error) or repr(error))
                raise
        if contract is None:
            self._agents()

        def work() -> str:
            if contract is not None:
                result = dict(contract)
            else:
                result = dict(self._agents().architect(run_id=job_id, workflow=job["workflow"]))
            required = {"skill_spec", "output_schema", "evaluation_plan"}
            missing = required - result.keys()
            if missing:
                raise ValueError(f"Skill Architect omitted: {sorted(missing)}")
            for name in sorted(required):
                self.store.put_artifact(job_id, f"{name}.json", result[name])
            return "integrating"

        return self._stage(
            job_id, "skill_architect", round=1, work=work, enter_status="architecting"
        )

    def integrate(self, job_id: str) -> dict[str, Any]:
        job = self._require_status(job_id, {"integrating"})
        waiting = "waiting for complete_integration with a full commit SHA"
        if self.integrator is None:
            raise JobBlocked(f"integrating: {waiting}")
        if any(
            item["stage"] == "skill_integration" and item["status"] == "failed"
            for item in job["stage_attempts"]
        ):
            raise JobBlocked(f"integrating: the integrator failed; {waiting}")
        contract = self._contract(job_id)

        # Not routed through _stage: an integrator failure must leave the manual fallback open.
        self.store.claim_stage(job_id, "skill_integration")
        self.ledger.record_run_event(
            run_id=job_id, stage="skill_integration", status="started", metadata={"round": 1}
        )
        try:
            result = dict(self.integrator.integrate(job=job, skill_contract=contract))
            integration = _integration_record(
                commit_sha=result.get("commit_sha", ""),
                test_summary=result.get("tests", {}),
                approved_by=result.get("approved_by"),
            )
        except Exception as error:
            _finish_failed_attempt(self.store, job_id, "skill_integration")
            self.store.put_artifact(
                job_id,
                "integration_error.json",
                {"error_type": type(error).__name__, "message": str(error)[:1000]},
            )
            self.ledger.record_run_event(
                run_id=job_id,
                stage="skill_integration",
                status="failed",
                metadata={"round": 1, "error_type": type(error).__name__},
            )
            raise JobBlocked(
                f"integrating: the integrator failed ({type(error).__name__}); {waiting}"
            ) from error
        self.store.put_artifact(job_id, "integration.json", integration)
        self.store.complete_stage(
            job_id,
            "skill_integration",
            status="completed",
            next_status="building_dataset",
            metadata={"round": 1},
        )
        self.ledger.record_run_event(
            run_id=job_id, stage="skill_integration", status="completed", metadata={"round": 1}
        )
        return self.store.get_job(job_id)

    def complete_integration(
        self,
        job_id: str,
        *,
        commit_sha: str,
        test_summary: Mapping[str, Any],
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        self._require_status(job_id, {"integrating"})
        self.store.put_artifact(
            job_id,
            "integration.json",
            _integration_record(
                commit_sha=commit_sha, test_summary=test_summary, approved_by=approved_by
            ),
        )
        return self.store.transition(job_id, "building_dataset", metadata={"manual": True})

    def build_dataset(self, job_id: str) -> dict[str, Any]:
        job = self._require_status(job_id, {"building_dataset"})
        round = job["dataset_round"]
        contract = self._contract(job_id)
        integration = self.store.get_artifact(job_id, "integration.json")
        feedback = self._dev_feedback(job_id, round)
        if self.dataset_source is None:
            self._agents()

        def work() -> str:
            if self.dataset_source is not None:
                dataset = dict(
                    self.dataset_source.build(
                        job=job,
                        round=round,
                        skill_contract=contract,
                        integration=integration,
                        feedback=feedback,
                    )
                )
                self.store.put_artifact(job_id, _round_name("dataset.json", round), dataset)
                return "validating"

            options = {"feedback": feedback} if feedback is not None else {}
            result = dict(
                self._agents().build_dataset(
                    run_id=job_id, workflow=job["workflow"], skill_contract=contract, **options
                )
            )
            required = {"train", "dev", "test", "ood"}
            missing = required - result.keys()
            if missing:
                raise ValueError(f"Data Builder omitted: {sorted(missing)}")
            if round > 1:
                # TEST is frozen in round 1; later rounds may only change TRAIN/DEV/OOD.
                result["test"] = self.store.get_artifact(job_id, "dataset_candidates.json")["test"]
            self.store.put_artifact(job_id, _round_name("dataset_candidates.json", round), result)
            return "validating"

        return self._stage(job_id, "data_builder", round=round, work=work)

    def validate_dataset(self, job_id: str) -> dict[str, Any]:
        job = self._require_status(job_id, {"validating"})
        round = job["dataset_round"]
        if self.dataset_source is None:
            raise JobBlocked(
                "validating: no deterministic validator is configured for agent-generated candidates"
            )
        dataset = self.store.get_artifact(job_id, _round_name("dataset.json", round))
        integration = self.store.get_artifact(job_id, "integration.json")

        def work() -> str:
            report = dict(
                self.dataset_source.validate(
                    job=job, round=round, dataset=dataset, integration=integration
                )
            )
            self.store.put_artifact(job_id, _round_name("validation.json", round), report)
            if report.get("passed") is not True:
                raise DatasetRejected("dataset validation did not pass")
            return "provisioning_gpu"

        return self._stage(job_id, "dataset_validator", round=round, work=work)

    def approve_spend(
        self, job_id: str, *, confirmation: str, approved_by: str
    ) -> dict[str, Any]:
        job = self._require_status(job_id, {"provisioning_gpu"})
        if confirmation != SPEND_CONFIRMATION:
            raise ValueError(f"spend approval requires confirmation {SPEND_CONFIRMATION}")
        round = job["dataset_round"]
        return self.store.put_artifact(
            job_id,
            _round_name("spend_approval.json", round),
            {
                "approved_by": _required(approved_by, "approved_by"),
                "confirmation": confirmation,
                "round": round,
            },
        )

    def train(self, job_id: str) -> dict[str, Any]:
        job = self._require_status(job_id, {"provisioning_gpu"})
        round = job["dataset_round"]
        if self.trainer is None:
            raise JobBlocked("provisioning_gpu: no training backend is configured")
        if self._optional_artifact(job_id, _round_name("spend_approval.json", round)) is None:
            raise JobBlocked(
                f"provisioning_gpu: waiting for approve_spend(confirmation={SPEND_CONFIRMATION!r})"
            )
        dataset = self.store.get_artifact(job_id, _round_name("dataset.json", round))
        integration = self.store.get_artifact(job_id, "integration.json")
        try:
            self.trainer.preflight(job=job, round=round, dataset=dataset, integration=integration)
        except JobBlocked:
            raise
        except Exception as error:
            raise JobBlocked(f"provisioning_gpu: {type(error).__name__}: {error}") from error

        def on_progress(stage: str) -> None:
            if stage == "training":
                self.store.transition(job_id, "training", metadata={"round": round})

        def work() -> str:
            result = dict(
                self.trainer.train(
                    job=job,
                    round=round,
                    dataset=dataset,
                    integration=integration,
                    on_progress=on_progress,
                )
            )
            required = {"version", "artifact_uri", "base_model"}
            missing = required - result.keys()
            if missing:
                raise ValueError(f"training backend omitted: {sorted(missing)}")
            self.store.put_artifact(job_id, _round_name("training.json", round), result)
            on_progress("training")
            return "evaluating"

        return self._stage(job_id, "adapter_training", round=round, work=work)

    def evaluate(
        self, job_id: str, *, deterministic_report: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        job = self._require_status(job_id, {"evaluating"})
        round = job["dataset_round"]
        if deterministic_report is None and self.evaluator is None:
            raise JobBlocked("evaluating: waiting for a deterministic report on frozen TEST")
        self._agents()

        def work() -> str:
            if deterministic_report is not None:
                report = dict(deterministic_report)
            else:
                training = self.store.get_artifact(job_id, _round_name("training.json", round))
                report = dict(self.evaluator.evaluate(job=job, round=round, training=training))
            if not isinstance(report.get("passed"), bool):
                raise ValueError("deterministic report must include a boolean 'passed'")
            self.store.put_artifact(
                job_id, _round_name("deterministic_evaluation.json", round), report
            )
            self.store.put_artifact(
                job_id,
                _round_name("qualitative_evaluation.json", round),
                self._qualitative(job_id, report),
            )
            if report["passed"]:
                return "publishing"
            if round < self._round_limit():
                return "building_dataset"
            raise EvaluationFailed(f"adapter failed evaluation after {round} dataset rounds")

        return self._stage(job_id, "independent_evaluator", round=round, work=work)

    def publish(self, job_id: str) -> dict[str, Any]:
        job = self._require_status(job_id, {"publishing"})
        round = job["dataset_round"]
        training = self.store.get_artifact(job_id, _round_name("training.json", round))
        report = self.store.get_artifact(
            job_id, _round_name("deterministic_evaluation.json", round)
        )
        artifact_uri = training["artifact_uri"]
        if self.publisher is not None:
            # Promotion is idempotent, so a transient failure blocks rather than failing the job.
            try:
                artifact_uri = self.publisher.promote(job=job, training=training)
            except JobBlocked:
                raise
            except Exception as error:
                raise JobBlocked(
                    f"publishing: promotion failed ({type(error).__name__}): {error}"
                ) from error

        def work() -> str:
            skill_id = job["skill_id"] or self.store.get_artifact(job_id, "skill_spec.json").get(
                "skill_id"
            )
            if not skill_id:
                raise ValueError("cannot publish an adapter without a skill_id")
            self.store.register_adapter(
                adapter_id=f"{skill_id}-{training['version']}",
                skill_id=skill_id,
                version=training["version"],
                base_model=training["base_model"],
                artifact_uri=artifact_uri,
                metadata={
                    "candidate_uri": training["artifact_uri"],
                    "job_id": job_id,
                    "dataset_round": round,
                    "metrics": report.get("metrics", {}),
                },
            )
            return "ready"

        return self._stage(job_id, "publisher", round=round, work=work)

    def _stage(
        self,
        job_id: str,
        stage: str,
        *,
        round: int,
        work: Callable[[], str],
        enter_status: str | None = None,
    ) -> dict[str, Any]:
        # A conflicting claim means another caller owns this stage: leave the job alone.
        self.store.claim_stage(job_id, stage, round=round, enter_status=enter_status)
        try:
            self.ledger.record_run_event(
                run_id=job_id, stage=stage, status="started", metadata={"round": round}
            )
            next_status = work()
            self.store.complete_stage(
                job_id,
                stage,
                status="completed",
                round=round,
                next_status=next_status,
                metadata={"round": round},
            )
            job = self.store.get_job(job_id)
            self.ledger.record_run_event(
                run_id=job_id,
                stage=stage,
                status="completed",
                metadata={"round": round, "next_status": next_status},
            )
            return job
        except Exception as error:
            _finish_failed_attempt(self.store, job_id, stage, round)
            self.store.fail(job_id, code=type(error).__name__, message=str(error) or repr(error))
            self.ledger.record_run_event(
                run_id=job_id,
                stage=stage,
                status="failed",
                metadata={"round": round, "error_type": type(error).__name__},
            )
            raise

    def _qualitative(self, job_id: str, report: Mapping[str, Any]) -> dict[str, Any]:
        # The review is advisory; a provider outage must not discard a paid, passing adapter.
        try:
            result = dict(self._agents().evaluate(run_id=job_id, deterministic_report=report))
            required = {"qualitative_findings", "failures"}
            missing = required - result.keys()
            if missing:
                raise ValueError(f"Independent Evaluator omitted: {sorted(missing)}")
            return result
        except Exception as error:
            self.ledger.record_run_event(
                run_id=job_id,
                stage="independent_evaluator",
                status="degraded",
                metadata={"error_type": type(error).__name__},
            )
            return {"error_type": type(error).__name__, "message": str(error)[:1000]}

    def _round_limit(self) -> int:
        source_limit = getattr(self.dataset_source, "max_rounds", None)
        if isinstance(source_limit, int) and source_limit >= 1:
            return min(self.max_dataset_rounds, source_limit)
        return self.max_dataset_rounds

    def _require_status(self, job_id: str, allowed: set[str]) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job["status"] not in allowed:
            raise InvalidTransition(
                f"job must be in {' or '.join(sorted(allowed))}, not {job['status']}"
            )
        return job

    def _contract(self, job_id: str) -> dict[str, Any]:
        return {
            name: self.store.get_artifact(job_id, f"{name}.json")
            for name in ("skill_spec", "output_schema", "evaluation_plan")
        }

    def _dev_feedback(self, job_id: str, round: int) -> dict[str, Any] | None:
        if round == 1:
            return None
        previous = self.store.get_artifact(
            job_id, _round_name("deterministic_evaluation.json", round - 1)
        )
        # Only DEV-derived feedback reaches the Data Builder; TEST results never do.
        return {"round": round - 1, "dev_feedback": previous.get("dev_feedback", {})}

    def _optional_artifact(self, job_id: str, name: str) -> Any:
        try:
            return self.store.get_artifact(job_id, name)
        except JobNotFound:
            return None

    def _agents(self) -> FactoryAgentGateway:
        if self.agents is None:
            raise JobBlocked("factory agents are not configured in this process")
        return self.agents


def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["metadata"] = json.loads(result.pop("metadata_json"))
    return result


def _apply_transition(
    connection: sqlite3.Connection,
    job_id: str,
    to_status: str,
    now: str,
    metadata_json: str,
) -> None:
    if to_status not in JOB_STAGES:
        raise ValueError(f"unknown job stage: {to_status}")
    current = connection.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if current is None:
        raise JobNotFound(job_id)
    from_status = current["status"]
    if from_status == to_status:
        return
    if to_status not in ALLOWED_TRANSITIONS[from_status]:
        raise InvalidTransition(f"cannot move job from {from_status} to {to_status}")
    connection.execute(
        "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?", (to_status, now, job_id)
    )
    connection.execute(
        """
        INSERT INTO job_events (
            event_id, job_id, from_status, to_status, occurred_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (str(uuid.uuid4()), job_id, from_status, to_status, now, metadata_json),
    )


def _migrate_stage_attempts_to_rounds(connection: sqlite3.Connection) -> None:
    columns = {
        item["name"] for item in connection.execute("PRAGMA table_info(job_stage_attempts)")
    }
    if "round" in columns:
        return
    connection.execute("ALTER TABLE job_stage_attempts RENAME TO job_stage_attempts_v1")
    connection.execute(
        """
        CREATE TABLE job_stage_attempts (
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            stage TEXT NOT NULL,
            round INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL,
            claimed_at TEXT NOT NULL,
            completed_at TEXT,
            PRIMARY KEY(job_id, stage, round)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO job_stage_attempts(job_id, stage, round, status, claimed_at, completed_at)
        SELECT job_id, stage, 1, status, claimed_at, completed_at FROM job_stage_attempts_v1
        """
    )
    connection.execute("DROP TABLE job_stage_attempts_v1")


def _finish_failed_attempt(store: JobStore, job_id: str, stage: str, round: int = 1) -> None:
    try:
        store.complete_stage(job_id, stage, status="failed", round=round)
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


def _round(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("round must be a positive integer")
    return value


def _round_name(name: str, round: int) -> str:
    # Round 1 keeps the original artifact names so existing jobs read unchanged.
    return name if round == 1 else f"round-{round}/{name}"


def _integration_record(
    *, commit_sha: str, test_summary: Mapping[str, Any], approved_by: str | None
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "commit_sha": _full_commit_sha(commit_sha),
        "tests": dict(test_summary),
    }
    if approved_by:
        value["approved_by"] = approved_by
    return value


def _full_commit_sha(value: str) -> str:
    value = _required(value, "commit_sha").lower()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("commit_sha must be a full 40-character Git SHA")
    return value


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
