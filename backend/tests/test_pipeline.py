import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext
from pathlib import Path

from dapta.backends import (
    RECEIPT_SKILL_ID,
    ApprovedContracts,
    HuggingFacePromoter,
    RemoteReceiptDataset,
    VultrTrainer,
)
from dapta.jobs import (
    AdapterJobOrchestrator,
    EvaluationFailed,
    JobBlocked,
    JobConflict,
    JobStore,
    TrainingBackend,
)
from dapta.ledger import UsageLedger


REPO = Path(__file__).resolve().parents[2]
SKILLS = REPO / "research/adapter-assets/skills"
COMMIT = "b" * 40
HEAD = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
).stdout.strip()
CONTRACT = {
    "skill_spec": {"skill_id": "expense-report"},
    "output_schema": {"type": "object"},
    "evaluation_plan": {"metric": "exact_match"},
}


class FakeAgents:
    def __init__(self, *, evaluator_error=None):
        self.architect_calls = 0
        self.evaluate_calls = 0
        self.evaluator_error = evaluator_error

    def architect(self, *, run_id, workflow):
        self.architect_calls += 1
        return CONTRACT

    def build_dataset(self, *, run_id, workflow, skill_contract):
        raise AssertionError("the dataset source should be used instead")

    def evaluate(self, *, run_id, deterministic_report):
        self.evaluate_calls += 1
        if self.evaluator_error:
            raise self.evaluator_error
        return {"qualitative_findings": [], "failures": []}


class FakeContracts:
    def get(self, skill_id):
        return CONTRACT if skill_id == "approved-skill" else None


class FakeIntegrator:
    def __init__(self, *, commit_sha=COMMIT, error=None):
        self.commit_sha = commit_sha
        self.error = error

    def integrate(self, *, job, skill_contract):
        if self.error:
            raise self.error
        return {"commit_sha": self.commit_sha, "tests": {"passed": 3}, "approved_by": "pavan"}


class FakeDataset:
    def __init__(self, *, valid=True, max_rounds=None):
        self.valid = valid
        self.feedback = []
        self.commits = []
        if max_rounds is not None:
            self.max_rounds = max_rounds

    def build(self, *, job, round, skill_contract, integration, feedback):
        self.feedback.append(feedback)
        self.commits.append(integration["commit_sha"])
        return {"round": round, "rows": 10 * round}

    def validate(self, *, job, round, dataset, integration):
        return {"passed": self.valid, "round": dataset["round"]}


class FakeTrainer:
    def __init__(self, store, *, fail=False, preflight_errors=()):
        self.store = store
        self.fail = fail
        self.preflight_errors = list(preflight_errors)
        self.statuses = []

    def preflight(self, *, job, round, dataset, integration):
        if self.preflight_errors:
            raise self.preflight_errors.pop(0)

    def train(self, *, job, round, dataset, integration, on_progress):
        self.statuses.append(self.store.get_job(job["job_id"])["status"])
        on_progress("training")
        self.statuses.append(self.store.get_job(job["job_id"])["status"])
        if self.fail:
            raise RuntimeError("worker exploded")
        return {
            "version": f"v1-r{round}",
            "base_model": "Qwen/Qwen3-VL-8B-Instruct",
            "artifact_uri": f"hf://owner/adapters/expense-report/v1-r{round}",
            "commit": integration["commit_sha"],
        }


class FakeEvaluator:
    def __init__(self, results):
        self.results = list(results)

    def evaluate(self, *, job, round, training):
        passed = self.results.pop(0)
        return {
            "passed": passed,
            "metrics": {"field_accuracy": 0.97 if passed else 0.6},
            "test_failures": ["test-case-17"],
            "dev_feedback": {"weak_fields": ["tax"], "round": round},
        }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "dapta.sqlite3"
        self.store = JobStore(self.path)
        self.ledger = UsageLedger(self.path)
        self.agents = FakeAgents()

    def tearDown(self):
        self.temp.cleanup()

    def orchestrator(self, **overrides):
        options = {
            "contracts": FakeContracts(),
            "integrator": FakeIntegrator(),
            "dataset_source": FakeDataset(),
            "trainer": FakeTrainer(self.store),
            "evaluator": FakeEvaluator([True]),
        }
        options.update(overrides)
        return AdapterJobOrchestrator(self.store, self.ledger, self.agents, **options)

    def create(self, orchestrator, skill_id="approved-skill", key="request-001"):
        return orchestrator.create_job(
            workflow="Turn receipt images into an expense report.",
            idempotency_key=key,
            skill_id=skill_id,
        )["job_id"]

    def test_run_reaches_ready_with_a_persisted_spend_gate(self):
        trainer = FakeTrainer(self.store)
        orchestrator = self.orchestrator(trainer=trainer)
        job_id = self.create(orchestrator)

        job, reason = orchestrator.run(job_id)
        self.assertEqual(job["status"], "provisioning_gpu")
        self.assertIn("approve_spend", reason)
        self.assertEqual(trainer.statuses, [])
        self.assertEqual(self.agents.architect_calls, 0, "approved contract skips the Architect")

        with self.assertRaises(ValueError):
            orchestrator.approve_spend(job_id, confirmation="yes", approved_by="pavan")
        orchestrator.approve_spend(job_id, confirmation="PROVISION", approved_by="pavan")
        job, reason = orchestrator.run(job_id)

        self.assertEqual(job["status"], "ready")
        self.assertEqual(reason, "job is ready")
        self.assertEqual(trainer.statuses, ["provisioning_gpu", "training"])
        self.assertEqual(
            [event["to_status"] for event in job["events"]],
            [
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
            ],
        )
        self.assertTrue(all(item["status"] == "completed" for item in job["stage_attempts"]))
        self.assertEqual(self.store.get_artifact(job_id, "integration.json")["commit_sha"], COMMIT)
        [adapter] = self.store.list_adapters()
        self.assertEqual(adapter["adapter_id"], "approved-skill-v1-r1")
        self.assertEqual(adapter["metadata"]["metrics"], {"field_accuracy": 0.97})

    def test_without_integrator_job_waits_for_manual_full_sha(self):
        orchestrator = self.orchestrator(integrator=None)
        job_id = self.create(orchestrator)
        job, reason = orchestrator.run(job_id)
        self.assertEqual(job["status"], "integrating")
        self.assertIn("complete_integration", reason)

        with self.assertRaisesRegex(ValueError, "40-character"):
            orchestrator.complete_integration(job_id, commit_sha="abc123", test_summary={})
        self.assertEqual(self.store.get_job(job_id)["status"], "integrating")
        job = orchestrator.complete_integration(
            job_id, commit_sha=COMMIT.upper(), test_summary={"passed": 1}, approved_by="pavan"
        )
        self.assertEqual(job["status"], "building_dataset")

    def test_unknown_skill_uses_the_architect(self):
        orchestrator = self.orchestrator()
        job_id = self.create(orchestrator, skill_id="new-skill")
        orchestrator.advance(job_id)
        self.assertEqual(self.agents.architect_calls, 1)

    def test_failed_evaluation_loops_back_with_dev_feedback_only(self):
        dataset = FakeDataset()
        orchestrator = self.orchestrator(
            dataset_source=dataset, evaluator=FakeEvaluator([False, True])
        )
        job_id = self.create(orchestrator)
        orchestrator.run(job_id)
        orchestrator.approve_spend(job_id, confirmation="PROVISION", approved_by="pavan")
        job, reason = orchestrator.run(job_id)

        self.assertEqual(job["status"], "provisioning_gpu", "round 2 needs its own spend approval")
        self.assertEqual(job["dataset_round"], 2)
        self.assertEqual(
            dataset.feedback,
            [None, {"round": 1, "dev_feedback": {"weak_fields": ["tax"], "round": 1}}],
        )
        self.assertNotIn("test-case-17", json.dumps(dataset.feedback))

        orchestrator.approve_spend(job_id, confirmation="PROVISION", approved_by="pavan")
        job, _ = orchestrator.run(job_id)
        self.assertEqual(job["status"], "ready")
        rounds = {(item["stage"], item["round"]) for item in job["stage_attempts"]}
        self.assertIn(("adapter_training", 1), rounds)
        self.assertIn(("adapter_training", 2), rounds)
        self.assertEqual(self.store.get_artifact(job_id, "round-2/dataset.json")["rows"], 20)
        self.assertEqual(self.store.list_adapters()[0]["version"], "v1-r2")

    def test_evaluation_fails_job_after_max_rounds(self):
        orchestrator = self.orchestrator(
            evaluator=FakeEvaluator([False, False]), max_dataset_rounds=2
        )
        job_id = self.create(orchestrator)
        for _ in range(2):
            orchestrator.run(job_id)
            orchestrator.approve_spend(job_id, confirmation="PROVISION", approved_by="pavan")
        with self.assertRaises(EvaluationFailed):
            orchestrator.run(job_id)
        job = self.store.get_job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["failure_code"], "EvaluationFailed")

    def test_rejected_dataset_fails_before_spend(self):
        trainer = FakeTrainer(self.store)
        orchestrator = self.orchestrator(dataset_source=FakeDataset(valid=False), trainer=trainer)
        job_id = self.create(orchestrator)
        with self.assertRaisesRegex(ValueError, "validation did not pass"):
            orchestrator.run(job_id)
        self.assertEqual(self.store.get_job(job_id)["failure_code"], "DatasetRejected")
        self.assertEqual(self.store.get_artifact(job_id, "validation.json")["passed"], False)
        self.assertEqual(trainer.statuses, [])

    def test_training_failure_fails_job_and_marks_attempt(self):
        orchestrator = self.orchestrator(trainer=FakeTrainer(self.store, fail=True))
        job_id = self.create(orchestrator)
        orchestrator.run(job_id)
        orchestrator.approve_spend(job_id, confirmation="PROVISION", approved_by="pavan")
        with self.assertRaisesRegex(RuntimeError, "worker exploded"):
            orchestrator.run(job_id)
        job = self.store.get_job(job_id)
        self.assertEqual(job["status"], "failed")
        attempts = {item["stage"]: item["status"] for item in job["stage_attempts"]}
        self.assertEqual(attempts["adapter_training"], "failed")

    def test_claim_conflict_leaves_job_untouched(self):
        orchestrator = self.orchestrator()
        job_id = self.create(orchestrator)
        self.store.claim_stage(job_id, "skill_architect")
        with self.assertRaises(JobConflict):
            orchestrator.advance(job_id)
        self.assertEqual(self.store.get_job(job_id)["status"], "queued")

    def test_interrupted_training_is_not_retried(self):
        orchestrator = self.orchestrator()
        job_id = self.create(orchestrator)
        orchestrator.run(job_id)
        self.store.transition(job_id, "training")
        with self.assertRaisesRegex(JobConflict, "interrupted"):
            orchestrator.advance(job_id)

    def test_missing_backends_block_instead_of_failing(self):
        orchestrator = AdapterJobOrchestrator(self.store, self.ledger)
        job_id = self.create(orchestrator)
        job, reason = orchestrator.run(job_id)
        self.assertEqual(job["status"], "queued")
        self.assertIn("factory agents", reason)
        self.assertIsInstance(JobBlocked(reason), RuntimeError)

    def test_integrator_failure_keeps_manual_fallback_open(self):
        for integrator in (
            FakeIntegrator(error=RuntimeError("devin timed out")),
            FakeIntegrator(commit_sha="abc1234"),
        ):
            orchestrator = self.orchestrator(integrator=integrator)
            job_id = self.create(orchestrator, key=f"request-{id(integrator)}")
            job, reason = orchestrator.run(job_id)
            self.assertEqual(job["status"], "integrating")
            self.assertIn("integrator failed", reason)
            attempts = {item["stage"]: item["status"] for item in job["stage_attempts"]}
            self.assertEqual(attempts["skill_integration"], "failed")
            self.assertIn("message", self.store.get_artifact(job_id, "integration_error.json"))

            _, reason = orchestrator.run(job_id)
            self.assertIn("integrator failed", reason, "a failed integrator is not re-run")
            job = orchestrator.complete_integration(
                job_id, commit_sha="d" * 40, test_summary={}, approved_by="pavan"
            )
            self.assertEqual(job["status"], "building_dataset")

    def test_dataset_source_receives_integration_commit(self):
        dataset = FakeDataset()
        orchestrator = self.orchestrator(dataset_source=dataset)
        orchestrator.run(self.create(orchestrator))
        self.assertEqual(dataset.commits, [COMMIT])

    def test_preflight_errors_block_without_consuming_the_paid_claim(self):
        trainer = FakeTrainer(
            self.store,
            preflight_errors=[ValueError("HF_ADAPTER_REPO is required"), RuntimeError("lock held")],
        )
        orchestrator = self.orchestrator(trainer=trainer)
        job_id = self.create(orchestrator)
        orchestrator.run(job_id)
        orchestrator.approve_spend(job_id, confirmation="PROVISION", approved_by="pavan")

        for expected in ("HF_ADAPTER_REPO", "lock held"):
            job, reason = orchestrator.run(job_id)
            self.assertEqual(job["status"], "provisioning_gpu")
            self.assertIn(expected, reason)
            self.assertNotIn("adapter_training", {item["stage"] for item in job["stage_attempts"]})
        job, _ = orchestrator.run(job_id)
        self.assertEqual(job["status"], "ready")

    def test_qualitative_review_outage_does_not_discard_a_passing_adapter(self):
        self.agents = FakeAgents(evaluator_error=RuntimeError("OpenRouter 503"))
        orchestrator = self.orchestrator()
        job_id = self.create(orchestrator)
        orchestrator.run(job_id)
        orchestrator.approve_spend(job_id, confirmation="PROVISION", approved_by="pavan")
        job, _ = orchestrator.run(job_id)
        self.assertEqual(job["status"], "ready")
        review = self.store.get_artifact(job_id, "qualitative_evaluation.json")
        self.assertEqual(review["error_type"], "RuntimeError")

    def test_single_round_source_fails_as_evaluation_failure(self):
        orchestrator = self.orchestrator(
            dataset_source=FakeDataset(max_rounds=1), evaluator=FakeEvaluator([False])
        )
        job_id = self.create(orchestrator)
        orchestrator.run(job_id)
        orchestrator.approve_spend(job_id, confirmation="PROVISION", approved_by="pavan")
        with self.assertRaises(EvaluationFailed):
            orchestrator.run(job_id)
        self.assertEqual(self.store.get_job(job_id)["failure_code"], "EvaluationFailed")

    def test_completion_and_transition_commit_together(self):
        orchestrator = self.orchestrator()
        job_id = self.create(orchestrator)
        self.store.claim_stage(job_id, "probe", enter_status="architecting")
        self.assertEqual(self.store.get_job(job_id)["status"], "architecting")
        with self.assertRaises(Exception):
            self.store.complete_stage(job_id, "probe", status="completed", next_status="ready")
        job = self.store.get_job(job_id)
        self.assertEqual(job["status"], "architecting")
        self.assertEqual(
            {item["stage"]: item["status"] for item in job["stage_attempts"]}["probe"], "running"
        )

    def test_publish_promotes_candidate_and_retries_after_promotion_errors(self):
        class FlakyPublisher:
            calls = 0

            def promote(self, *, job, training):
                self.calls += 1
                if self.calls == 1:
                    raise ConnectionError("hub unreachable")
                return training["artifact_uri"].replace("/candidates", "")

        publisher = FlakyPublisher()
        trainer = FakeTrainer(self.store)
        orchestrator = self.orchestrator(trainer=trainer, publisher=publisher)
        job_id = self.create(orchestrator)
        orchestrator.run(job_id)
        orchestrator.approve_spend(job_id, confirmation="PROVISION", approved_by="pavan")
        job, reason = orchestrator.run(job_id)
        self.assertEqual(job["status"], "publishing")
        self.assertIn("promotion failed", reason)
        self.assertEqual(self.store.list_adapters(), [])

        job, _ = orchestrator.run(job_id)
        self.assertEqual(job["status"], "ready")
        [adapter] = self.store.list_adapters()
        self.assertEqual(adapter["artifact_uri"], "hf://owner/adapters/expense-report/v1-r1")
        self.assertEqual(
            adapter["metadata"]["candidate_uri"], "hf://owner/adapters/expense-report/v1-r1"
        )

    def test_failed_evaluation_is_never_promoted(self):
        class RecordingPublisher:
            promoted = []

            def promote(self, *, job, training):
                self.promoted.append(training["version"])
                return training["artifact_uri"]

        publisher = RecordingPublisher()
        orchestrator = self.orchestrator(
            evaluator=FakeEvaluator([False]), publisher=publisher, max_dataset_rounds=1
        )
        job_id = self.create(orchestrator)
        orchestrator.run(job_id)
        orchestrator.approve_spend(job_id, confirmation="PROVISION", approved_by="pavan")
        with self.assertRaises(EvaluationFailed):
            orchestrator.run(job_id)
        self.assertEqual(publisher.promoted, [])

    def test_version_one_database_is_migrated(self):
        legacy = Path(self.temp.name) / "legacy.sqlite3"
        with sqlite3.connect(legacy) as connection:
            connection.executescript(
                """
                CREATE TABLE job_schema_info (version INTEGER NOT NULL);
                INSERT INTO job_schema_info VALUES (1);
                CREATE TABLE jobs (
                    job_id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE,
                    workflow TEXT NOT NULL, workflow_hash TEXT NOT NULL, skill_id TEXT,
                    status TEXT NOT NULL, failure_code TEXT, failure_message TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO jobs VALUES ('j1', 'k1', 'w', 'h', NULL, 'integrating',
                    NULL, NULL, '2026-10-17T00:00:00Z', '2026-10-17T00:00:00Z');
                CREATE TABLE job_stage_attempts (
                    job_id TEXT NOT NULL REFERENCES jobs(job_id), stage TEXT NOT NULL,
                    status TEXT NOT NULL, claimed_at TEXT NOT NULL, completed_at TEXT,
                    PRIMARY KEY(job_id, stage)
                );
                INSERT INTO job_stage_attempts VALUES ('j1', 'skill_architect', 'completed',
                    '2026-10-17T00:00:00Z', '2026-10-17T00:01:00Z');
                """
            )
        store = JobStore(legacy)
        job = store.get_job("j1")
        self.assertEqual(job["stage_attempts"][0]["round"], 1)
        store.claim_stage("j1", "data_builder", round=2)
        with sqlite3.connect(legacy) as connection:
            self.assertEqual(
                connection.execute("SELECT version FROM job_schema_info").fetchone()[0], 2
            )


class FakeRunner:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeVultrWorker:
    def __init__(self, on_event, artifacts_dir):
        self.on_event = on_event
        self.artifacts_dir = artifacts_dir
        self.runner = FakeRunner()
        self.kwargs = None
        self.planned = None

    def plan(self, job_id, commit, *, limit):
        self.planned = (job_id, commit, limit)

    def run(self, **kwargs):
        self.kwargs = kwargs
        self.on_event("workload_started")
        path = kwargs["validation_report_path"]
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"passed": True, "counts": {"train": 5}}), encoding="utf-8")
        self.on_event("dataset_validated")
        return {"instance_id": "instance-1", "allocation_id": "vultr:instance-1"}


class ReceiptBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_approved_contract_loads_and_rejects_unsafe_ids(self):
        contracts = ApprovedContracts(SKILLS)
        contract = contracts.get(RECEIPT_SKILL_ID)
        self.assertEqual(set(contract), {"skill_spec", "output_schema", "evaluation_plan"})
        self.assertIsNone(contracts.get("../receipt-to-expense"))
        self.assertIsNone(contracts.get("three-statement-model"))

    def test_unapproved_review_falls_back_to_the_architect(self):
        skill = self.root / RECEIPT_SKILL_ID
        skill.mkdir()
        (skill / "review.json").write_text('{"status": "draft"}', encoding="utf-8")
        self.assertIsNone(ApprovedContracts(self.root).get(RECEIPT_SKILL_ID))

    def test_remote_dataset_reads_the_integration_commit(self):
        source = RemoteReceiptDataset(repo_dir=REPO, limit_per_source_split=8)
        job = {"skill_id": RECEIPT_SKILL_ID}
        integration = {"commit_sha": HEAD}
        contract = ApprovedContracts(SKILLS).get(RECEIPT_SKILL_ID)
        dataset = source.build(
            job=job, round=1, skill_contract=contract, integration=integration, feedback=None
        )
        self.assertEqual(dataset["commit_sha"], HEAD)
        self.assertEqual(set(dataset["sources"]), {"receipt-cord-v2", "receipt-sroie"})
        self.assertTrue(
            source.validate(job=job, round=1, dataset=dataset, integration=integration)["passed"]
        )

        tampered = {**dataset, "sources": {}}
        self.assertFalse(
            source.validate(job=job, round=1, dataset=tampered, integration=integration)["passed"]
        )
        other = {"commit_sha": "0" * 40}
        with self.assertRaisesRegex(ValueError, "not present at commit"):
            source.validate(job=job, round=1, dataset=dataset, integration=other)
        with self.assertRaisesRegex(ValueError, "identical every round"):
            source.build(
                job=job, round=2, skill_contract=contract, integration=integration, feedback={}
            )
        edited = {**contract, "output_schema": {"type": "object"}}
        with self.assertRaisesRegex(ValueError, "differs"):
            source.build(
                job=job, round=1, skill_contract=edited, integration=integration, feedback=None
            )

    def test_vultr_trainer_maps_progress_and_reports_adapter(self):
        workers = []

        def factory(on_event):
            workers.append(FakeVultrWorker(on_event, self.root))
            return workers[-1]

        progress = []
        trainer: TrainingBackend = VultrTrainer(
            factory, hf_repo="owner/adapters", artifacts_dir=self.root, lock=nullcontext
        )
        job = {"job_id": "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0", "skill_id": RECEIPT_SKILL_ID}
        with patch.dict(os.environ, {"HF_TOKEN": "hf-secret"}, clear=True):
            trainer.preflight(
                job=job, round=2, dataset={"limit_per_source_split": 8}, integration={"commit_sha": COMMIT}
            )
            result = trainer.train(
                job=job,
                round=2,
                dataset={"limit_per_source_split": 8},
                integration={"commit_sha": COMMIT},
                on_progress=progress.append,
            )
        self.assertEqual(workers[0].planned, (f"{job['job_id']}-r2", COMMIT, 8))
        self.assertTrue(workers[0].runner.closed, "preflight releases its SSH temp dir")
        self.assertEqual(progress, ["training"])
        self.assertEqual(result["version"], "v1-0f1e2d3c4b5a-r2")
        self.assertEqual(
            result["artifact_uri"],
            "hf://owner/adapters/candidates/receipt-to-expense/v1-0f1e2d3c4b5a-r2",
        )
        self.assertEqual(result["remote_validation"]["counts"], {"train": 5})
        kwargs = workers[1].kwargs
        self.assertEqual(
            (kwargs["run_id"], kwargs["job_id"], kwargs["commit"], kwargs["limit"]),
            (job["job_id"], f"{job['job_id']}-r2", COMMIT, 8),
        )
        self.assertEqual(
            (kwargs["hf_repo"], kwargs["hf_token"]),
            ("owner/adapters", "hf-secret"),
            "the worker uploads to the same repo the trainer reports",
        )

    def test_promoter_copies_candidate_folder_to_published_path(self):
        class FakeHub:
            def __init__(self):
                self.uploads = []

            def snapshot_download(self, *, repo_id, allow_patterns, local_dir):
                folder = Path(local_dir) / "candidates/receipt-to-expense/v1-abc-r1"
                folder.mkdir(parents=True)
                (folder / "adapter_model.safetensors").write_bytes(b"weights")
                self.download = (repo_id, allow_patterns)
                return local_dir

            def upload_folder(self, *, repo_id, folder_path, path_in_repo, commit_message):
                files = sorted(item.name for item in Path(folder_path).iterdir())
                self.uploads.append((repo_id, path_in_repo, files))

        hub = FakeHub()
        uri = HuggingFacePromoter(hub).promote(
            job={"job_id": "job-1"},
            training={"artifact_uri": "hf://owner/adapters/candidates/receipt-to-expense/v1-abc-r1"},
        )
        self.assertEqual(uri, "hf://owner/adapters/receipt-to-expense/v1-abc-r1")
        self.assertEqual(
            hub.download, ("owner/adapters", ["candidates/receipt-to-expense/v1-abc-r1/*"])
        )
        self.assertEqual(
            hub.uploads,
            [("owner/adapters", "receipt-to-expense/v1-abc-r1", ["adapter_model.safetensors"])],
        )
        with self.assertRaisesRegex(ValueError, "not a candidate"):
            HuggingFacePromoter(hub).promote(
                job={"job_id": "job-1"},
                training={"artifact_uri": "hf://owner/adapters/receipt-to-expense/v1"},
            )

    def test_vultr_trainer_versions_never_collide_across_jobs(self):
        def factory(on_event):
            return FakeVultrWorker(on_event, self.root)

        trainer = VultrTrainer(factory, hf_repo="o/r", artifacts_dir=self.root, lock=nullcontext)
        versions = set()
        with patch.dict(os.environ, {"HF_TOKEN": "hf-secret"}, clear=True):
            for job_id in ("11111111-aaaa-bbbb-cccc-000000000000", "22222222-aaaa-bbbb-cccc-000000000000"):
                versions.add(
                    trainer.train(
                        job={"job_id": job_id, "skill_id": RECEIPT_SKILL_ID},
                        round=1,
                        dataset={},
                        integration={"commit_sha": COMMIT},
                        on_progress=lambda _: None,
                    )["version"]
                )
        self.assertEqual(len(versions), 2)

    def test_vultr_trainer_preflight_requires_hf_settings(self):
        trainer = VultrTrainer(lambda _: None, hf_repo="", artifacts_dir=self.root, lock=nullcontext)
        job = {"job_id": "job-1", "skill_id": RECEIPT_SKILL_ID}
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "HF_ADAPTER_REPO"):
                trainer.preflight(job=job, round=1, dataset={}, integration={"commit_sha": COMMIT})
        trainer = VultrTrainer(lambda _: None, hf_repo="o/r", artifacts_dir=self.root, lock=nullcontext)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
                trainer.preflight(job=job, round=1, dataset={}, integration={"commit_sha": COMMIT})


if __name__ == "__main__":
    unittest.main()
