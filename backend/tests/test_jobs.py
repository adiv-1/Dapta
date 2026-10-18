import tempfile
import unittest
from pathlib import Path

from dapta.jobs import AdapterJobOrchestrator, InvalidTransition, JobConflict, JobStore
from dapta.ledger import UsageLedger


class FakeAgents:
    def architect(self, *, run_id, workflow):
        return {
            "skill_spec": {
                "name": "expense-report",
                "description": "Turn receipts into an expense report.",
                "workflow": workflow,
            },
            "output_schema": {"type": "object"},
            "evaluation_plan": {"metric": "field_f1"},
        }

    def build_dataset(self, *, run_id, workflow, skill_contract, feedback=None):
        raise AssertionError("the dataset source owns data generation")

    def evaluate(self, *, run_id, deterministic_report):
        return {"qualitative_findings": [], "failures": []}


class FakeIntegrator:
    def integrate(self, *, job, skill_contract):
        return {
            "session_id": "devin-session-1",
            "status": "completed",
            "pull_request_url": None,
            "evidence_type": "session",
        }


class FakeDatasetSource:
    max_rounds = 2

    def build(self, *, job, round, skill_contract, integration, feedback):
        return {
            "train": [{"id": f"train-{round}", "input": "long training input", "output": {"ok": True}}],
            "dev": [{"id": f"dev-{round}", "input": "long validation input", "output": {"ok": True}}],
            "test": [{"id": f"test-{round}", "input": "frozen test input", "output": {"ok": True}}],
            "ood": [{"id": f"ood-{round}", "input": "long out of distribution input", "output": {"ok": True}}],
        }

    def validate(self, *, job, round, dataset, integration, skill_contract):
        return {
            "passed": True,
            "round": round,
            "counts": {name: len(rows) for name, rows in dataset.items()},
            "frozen_test_sha256": "frozen-test-hash",
        }


class FakeTrainer:
    def preflight(self, *, job, round, dataset, integration):
        self.preflight_round = round

    def train(self, *, job, round, dataset, integration, on_progress):
        on_progress({"stage": "training", "current": 1, "total": 1, "percent": 100})
        return {
            "version": f"v{round}",
            "adapter_dir": f"/tmp/adapter-{round}",
            "base_model": "Qwen/Qwen3-VL-8B-Instruct",
            "training_cost_usd": "0.10",
            "evaluation": {"round": round},
        }


class FakeEvaluator:
    def __init__(self, fail_first=False):
        self.fail_first = fail_first

    def evaluate(self, *, job, round, training):
        passed = not self.fail_first or round > 1
        return {
            "passed": passed,
            "conditions": {
                "base_no_skill": {"field_f1": 0.50},
                "base_with_skill": {"field_f1": 0.55},
                "lora_short_prompt": {"field_f1": 0.70 if passed else 0.54},
            },
            "dev_feedback": {} if passed else {"instruction": "repair DEV failures"},
        }


class FakePublisher:
    def promote(self, *, job, training, evaluation, skill_contract):
        return f"hf://adityave/dapta-lora-adapters/generated/expense-report/{training['version']}"


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "dapta.sqlite3"
        self.store = JobStore(self.path)
        self.ledger = UsageLedger(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def orchestrator(self, *, fail_first=False):
        return AdapterJobOrchestrator(
            self.store,
            self.ledger,
            FakeAgents(),
            integrator=FakeIntegrator(),
            dataset_source=FakeDatasetSource(),
            trainer=FakeTrainer(),
            evaluator=FakeEvaluator(fail_first=fail_first),
            publisher=FakePublisher(),
            max_dataset_rounds=2,
        )

    def test_create_is_idempotent_but_conflicting_reuse_fails(self):
        orchestrator = self.orchestrator()
        route = {"decision": "create_adapter", "confidence": 0.92, "reason": "new workflow"}
        first = orchestrator.create_job(
            workflow="Turn receipt images into an expense report.",
            idempotency_key="request-001",
            skill_id="expense-report",
            route_decision=route,
        )
        second = orchestrator.create_job(
            workflow="Turn receipt images into an expense report.",
            idempotency_key="request-001",
            skill_id="expense-report",
            route_decision=route,
        )
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(second["route"], route)
        self.assertEqual(len(second["events"]), 1)

        with self.assertRaises(JobConflict):
            orchestrator.create_job(
                workflow="Review an NDA.",
                idempotency_key="request-001",
                skill_id="nda-review",
            )

    def test_complete_factory_run_publishes_adapter_and_progress(self):
        orchestrator = self.orchestrator()
        job = orchestrator.create_job(
            workflow="Turn receipt images into an expense report.",
            idempotency_key="request-002",
            skill_id="expense-report",
        )
        seen = [job["status"]]
        while job["status"] not in {"ready", "failed"}:
            job = orchestrator.advance(job["job_id"])
            seen.append(job["status"])

        self.assertEqual(job["status"], "ready")
        self.assertEqual(
            seen,
            [
                "queued", "integrating", "building_dataset", "validating",
                "awaiting_compute", "provisioning_gpu", "evaluating", "publishing", "ready",
            ],
        )
        self.assertEqual(job["progress"]["percent"], 100)
        self.assertTrue(job["deterministic_evaluation"]["passed"])
        self.assertTrue(job["publish_result"]["artifact_uri"].startswith("hf://"))
        self.assertEqual(len(self.store.list_adapters()), 1)
        self.assertTrue(all(attempt["status"] == "completed" for attempt in job["stage_attempts"]))
        self.assertEqual(job["agent_outputs"][0]["agent"], "Kimi K3 — Skill Architect")
        self.assertEqual(job["agent_outputs"][-1]["stage"], "publishing")
        self.assertNotIn("dataset", job)

    def test_failed_evaluation_repairs_dev_but_preserves_frozen_test(self):
        orchestrator = self.orchestrator(fail_first=True)
        job = orchestrator.create_job(
            workflow="Turn receipt images into an expense report.",
            idempotency_key="request-003",
            skill_id="expense-report",
        )
        while job["status"] not in {"ready", "failed"}:
            job = orchestrator.advance(job["job_id"])

        self.assertEqual(job["status"], "ready")
        self.assertEqual(job["dataset_round"], 2)
        first = self.store.get_artifact(job["job_id"], "dataset.json")
        second = self.store.get_artifact(job["job_id"], "round-2/dataset.json")
        self.assertEqual(first["test"], second["test"])
        self.assertNotEqual(first["train"], second["train"])

    def test_stage_order_cannot_be_skipped(self):
        job = self.store.create_job(
            workflow="Turn receipt images into an expense report.",
            idempotency_key="request-004",
        )
        with self.assertRaisesRegex(InvalidTransition, "queued to training"):
            self.store.transition(job["job_id"], "training")

    def test_paid_stage_can_only_be_claimed_once(self):
        job = self.store.create_job(
            workflow="Turn receipt images into an expense report.",
            idempotency_key="request-005",
        )
        self.store.claim_stage(job["job_id"], "skill_architect")
        with self.assertRaisesRegex(JobConflict, "already claimed"):
            self.store.claim_stage(job["job_id"], "skill_architect")

    def test_abandoned_stage_is_recorded_and_can_resume(self):
        job = self.store.create_job(
            workflow="Build a stable repeated reporting specialist.",
            idempotency_key="resume-stage-001",
        )
        self.store.claim_stage(
            job["job_id"], "skill_architect", enter_status="architecting"
        )

        self.assertEqual(self.store.recover_interrupted_stages(), 1)
        recovered = self.store.get_job(job["job_id"])
        self.assertTrue(
            recovered["events"][-1]["metadata"]["recovered_after_controller_restart"]
        )

        self.store.claim_stage(
            job["job_id"], "skill_architect", enter_status="architecting"
        )
        self.store.complete_stage(
            job["job_id"], "skill_architect", status="completed",
            next_status="integrating",
        )
        self.assertEqual(self.store.get_job(job["job_id"])["status"], "integrating")

    def test_adapter_registry_is_versioned(self):
        adapter = self.store.register_adapter(
            adapter_id="expense-report-v1",
            skill_id="expense-report",
            version="v1",
            base_model="Qwen/Qwen3-VL-8B-Instruct",
            artifact_uri="hf://adityave/dapta-lora-adapters/expense-report/v1",
            metadata={"dataset_hash": "abc"},
            created_at="2026-10-17T08:00:00Z",
        )
        self.assertEqual(adapter["metadata"], {"dataset_hash": "abc"})
        self.assertEqual(self.store.list_adapters(), [adapter])

    def test_failed_stage_can_be_retried_from_the_same_stage(self):
        job = self.store.create_job(
            workflow="Build a stable repeated reporting specialist.",
            idempotency_key="retry-stage-001",
        )
        self.store.claim_stage(
            job["job_id"], "skill_architect", enter_status="architecting"
        )
        self.store.complete_stage(
            job["job_id"], "skill_architect", status="failed"
        )
        self.store.fail(job["job_id"], code="ProviderError", message="temporary")

        retried = self.store.retry_failed(job["job_id"])

        self.assertEqual(retried["status"], "architecting")
        self.assertIsNone(retried["failure_code"])
        self.assertEqual(retried["stage_attempts"][0]["status"], "interrupted")
        self.assertTrue(retried["events"][-1]["metadata"]["retry"])

    def test_failed_training_retry_returns_to_claimable_gpu_stage(self):
        job = self.store.create_job(
            workflow="Train a reusable specialist on validated examples.",
            idempotency_key="retry-training-001",
        )
        for status in (
            "architecting",
            "integrating",
            "building_dataset",
            "validating",
            "awaiting_compute",
            "provisioning_gpu",
        ):
            self.store.transition(job["job_id"], status)
        self.store.claim_stage(
            job["job_id"], "adapter_training", enter_status="training"
        )
        self.store.complete_stage(
            job["job_id"], "adapter_training", status="failed"
        )
        self.store.fail(job["job_id"], code="RuntimeError", message="dtype mismatch")

        retried = self.store.retry_failed(job["job_id"])

        self.assertEqual(retried["status"], "provisioning_gpu")
        self.assertIsNone(retried["failure_code"])
        self.assertEqual(retried["stage_attempts"][-1]["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
