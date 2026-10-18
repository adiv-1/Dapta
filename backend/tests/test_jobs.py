import tempfile
import unittest
from pathlib import Path

from dapta.jobs import AdapterJobOrchestrator, InvalidTransition, JobConflict, JobStore
from dapta.ledger import UsageLedger


class FakeAgents:
    def architect(self, *, run_id, workflow):
        return {
            "skill_spec": {"name": "expense-report", "workflow": workflow},
            "output_schema": {"type": "object"},
            "evaluation_plan": {"metric": "exact_match"},
        }

    def build_dataset(self, *, run_id, workflow, skill_contract):
        return {"train": [1], "dev": [2], "test": [3], "ood": [4]}

    def evaluate(self, *, run_id, deterministic_report):
        return {"qualitative_findings": [], "failures": []}


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "dapta.sqlite3"
        self.store = JobStore(self.path)
        self.ledger = UsageLedger(self.path)
        self.orchestrator = AdapterJobOrchestrator(
            self.store, self.ledger, FakeAgents()
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_create_is_idempotent_but_conflicting_reuse_fails(self):
        first = self.orchestrator.create_job(
            workflow="Turn receipt images into an expense report.",
            idempotency_key="request-001",
            skill_id="expense-report",
        )
        second = self.orchestrator.create_job(
            workflow="Turn receipt images into an expense report.",
            idempotency_key="request-001",
            skill_id="expense-report",
        )
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(len(second["events"]), 1)

        with self.assertRaises(JobConflict):
            self.orchestrator.create_job(
                workflow="Review an NDA.",
                idempotency_key="request-001",
                skill_id="nda-review",
            )

    def test_architect_and_data_builder_persist_outputs_and_stages(self):
        job = self.orchestrator.create_job(
            workflow="Turn receipt images into an expense report.",
            idempotency_key="request-002",
            skill_id="expense-report",
        )
        job = self.orchestrator.architect(job["job_id"])
        self.assertEqual(job["status"], "integrating")
        self.assertEqual(job["stage_attempts"][0]["status"], "completed")
        self.assertEqual(
            {item["name"] for item in job["artifacts"]},
            {"skill_spec.json", "output_schema.json", "evaluation_plan.json"},
        )

        job = self.orchestrator.complete_integration(
            job["job_id"], commit_sha="abc123", test_summary={"passed": 12}
        )
        self.assertEqual(job["status"], "building_dataset")
        job = self.orchestrator.build_dataset(job["job_id"])
        self.assertEqual(job["status"], "validating")
        self.assertEqual(
            {item["stage"]: item["status"] for item in job["stage_attempts"]},
            {"skill_architect": "completed", "data_builder": "completed"},
        )
        candidates = self.store.get_artifact(job["job_id"], "dataset_candidates.json")
        self.assertEqual(set(candidates), {"train", "dev", "test", "ood"})

        for stage in ("awaiting_compute", "provisioning_gpu", "training", "evaluating"):
            job = self.store.transition(job["job_id"], stage)
        job = self.orchestrator.evaluate(
            job["job_id"], deterministic_report={"exact_match": 0.95}
        )
        self.assertEqual(job["status"], "publishing")
        self.assertEqual(
            self.store.get_artifact(job["job_id"], "qualitative_evaluation.json"),
            {"qualitative_findings": [], "failures": []},
        )

    def test_stage_order_cannot_be_skipped(self):
        job = self.store.create_job(
            workflow="Turn receipt images into an expense report.",
            idempotency_key="request-003",
        )
        with self.assertRaisesRegex(InvalidTransition, "queued to training"):
            self.store.transition(job["job_id"], "training")

    def test_paid_stage_can_only_be_claimed_once(self):
        job = self.store.create_job(
            workflow="Turn receipt images into an expense report.",
            idempotency_key="request-004",
        )
        self.store.claim_stage(job["job_id"], "skill_architect")
        with self.assertRaisesRegex(JobConflict, "already claimed"):
            self.store.claim_stage(job["job_id"], "skill_architect")

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


if __name__ == "__main__":
    unittest.main()
