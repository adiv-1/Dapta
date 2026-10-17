import os
import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dapta.ledger import UsageLedger
from dapta.vultr import CONFIRMATION, VultrError, VultrWorker, WorkerConfig


COMMIT = "a" * 40


class FakeClient:
    def __init__(self, *, active=None, fail_delete=False):
        self.active = active or []
        self.fail_delete = fail_delete
        self.created = None
        self.deleted = False

    def available_plans(self, region):
        return {"vcg-a40-24c-120g-48vram"}

    def plans(self):
        return [{
            "id": "vcg-a40-24c-120g-48vram",
            "hourly_cost": "1.712",
            "gpu_type": "NVIDIA A40",
        }]

    def instances(self):
        return self.active

    def create_instance(self, payload):
        self.created = dict(payload)
        return {
            "id": "instance-1",
            "label": payload["label"],
            "status": "pending",
            "date_created": "2026-10-17T12:00:00Z",
        }

    def get_instance(self, instance_id):
        if self.deleted:
            return None
        return {
            "id": instance_id,
            "status": "active",
            "power_status": "running",
            "server_status": "ok",
            "main_ip": "8.8.8.8",
        }

    def delete_instance(self, instance_id):
        if self.fail_delete:
            raise RuntimeError("deletion failed")
        self.deleted = True


class FakeRunner:
    def __init__(self, *, fail_train=False):
        self.fail_train = fail_train
        self.calls = []
        self.closed = False

    def wait_ready(self, host, timeout, poll_seconds):
        self.calls.append("wait_ready")

    def verify_commit(self, commit):
        self.calls.append("verify_commit")

    def upload_commit(self, host, commit):
        self.calls.append("upload_commit")

    def upload_environment(self, host, values):
        self.calls.append(("upload_environment", set(values)))

    def prepare(self, host, job_id, *, skill_id, limit, timeout):
        self.calls.append(("prepare", skill_id, limit))

    def read_validation(self, host, job_id, *, timeout):
        self.calls.append("read_validation")
        return {
            "passed": True,
            "validator_version": "receipt-validator-v1",
            "counts": {"train": 10, "dev": 2, "test": 2},
            "hashes": {"splits": {"test": "abc"}},
        }

    def train(self, host, job_id, version, *, skill_id, smoke, timeout):
        self.calls.append(("train", skill_id, smoke))
        if self.fail_train:
            raise RuntimeError("training failed")

    def close(self):
        self.closed = True


class VultrWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.key = root / "id_ed25519"
        self.key.write_text("test", encoding="utf-8")
        self.ledger = UsageLedger(root / "ledger.sqlite3")
        self.ledger.create_run(run_id="run-1", run_type="adapter_training", job_id="job-1")
        self.config = WorkerConfig(
            api_key="secret",
            region="ewr",
            plan_id="vcg-a40-24c-120g-48vram",
            os_id=1743,
            ssh_key_id="ssh-key-1",
            ssh_private_key=self.key,
            hourly_cost_usd=Decimal("1.712"),
            max_job_cost_usd=Decimal("5"),
            max_job_seconds=7200,
            ready_timeout_seconds=60,
            poll_seconds=1,
        )

    def tearDown(self):
        self.temp.cleanup()

    def worker(self, client, runner):
        return VultrWorker(
            self.config,
            client,
            self.ledger,
            runner,
            sleep=lambda _: None,
            monotonic=lambda: 100,
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_configuration_enforces_per_job_cost_limit(self):
        os.environ.update({
            "VULTR_API_KEY": "secret",
            "VULTR_REGION": "ewr",
            "VULTR_GPU_PLAN_ID": "vcg-a40-24c-120g-48vram",
            "VULTR_OS_ID": "1743",
            "VULTR_SSH_KEY_ID": "ssh-key-1",
            "VULTR_SSH_PRIVATE_KEY_PATH": str(self.key),
            "VULTR_HOURLY_COST_USD": "1.712",
            "VULTR_MAX_JOB_COST_USD": "3",
            "VULTR_MAX_JOB_SECONDS": "7200",
        })
        with self.assertRaisesRegex(ValueError, "exceeds"):
            WorkerConfig.from_env()

    def test_plan_is_read_only_and_reports_the_cost_ceiling(self):
        client = FakeClient()
        runner = FakeRunner()
        result = self.worker(client, runner).plan("job-1", COMMIT, limit=32)

        self.assertEqual(result["projected_max_cost_usd"], "3.424")
        self.assertFalse(result["will_provision"])
        self.assertIsNone(client.created)

    def test_paid_run_requires_literal_confirmation_before_api_calls(self):
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "confirm-spend"):
            self.worker(client, FakeRunner()).run(
                run_id="run-1",
                job_id="job-1",
                commit=COMMIT,
                version="v1-smoke",
                limit=32,
                confirmation="",
            )
        self.assertIsNone(client.created)

    @patch.dict(
        os.environ,
        {"HF_TOKEN": "hf-secret", "HF_ADAPTER_REPO": "owner/adapters"},
        clear=True,
    )
    def test_success_records_lifecycle_and_always_deletes(self):
        client = FakeClient()
        runner = FakeRunner()
        result = self.worker(client, runner).run(
            run_id="run-1",
            job_id="job-1",
            commit=COMMIT,
            version="v1-smoke",
            limit=32,
            confirmation=CONFIRMATION,
        )

        self.assertEqual(result["status"], "completed")
        self.assertTrue(client.deleted)
        self.assertTrue(runner.closed)
        self.assertNotIn("block_devices", client.created)
        self.assertEqual(client.created["sshkey_id"], ["ssh-key-1"])
        with sqlite3.connect(self.ledger.path) as connection:
            events = {
                row[0]
                for row in connection.execute(
                    "SELECT event_type FROM compute_events WHERE allocation_id = 'vultr:instance-1'"
                )
            }
            cost_count = connection.execute("SELECT COUNT(*) FROM compute_costs").fetchone()[0]
            run_statuses = [
                row[0]
                for row in connection.execute(
                    "SELECT status FROM run_events "
                    "WHERE run_id = 'run-1' AND stage = 'vultr_worker' ORDER BY occurred_at"
                )
            ]
        self.assertEqual(
            events,
            {
                "provision_requested",
                "created",
                "ready",
                "workload_started",
                "workload_finished",
                "destroy_requested",
                "destroyed",
            },
        )
        self.assertEqual(cost_count, 1)
        self.assertEqual(run_statuses, ["started", "completed"])

    @patch.dict(
        os.environ,
        {"HF_TOKEN": "hf-secret", "HF_ADAPTER_REPO": "owner/adapters"},
        clear=True,
    )
    def test_training_failure_still_deletes_and_records_failure(self):
        client = FakeClient()
        runner = FakeRunner(fail_train=True)
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.worker(client, runner).run(
                run_id="run-1",
                job_id="job-1",
                commit=COMMIT,
                version="v1-smoke",
                limit=32,
                confirmation=CONFIRMATION,
            )
        self.assertTrue(client.deleted)
        with sqlite3.connect(self.ledger.path) as connection:
            failures = connection.execute(
                "SELECT COUNT(*) FROM compute_events WHERE event_type = 'failed'"
            ).fetchone()[0]
            run_failures = connection.execute(
                "SELECT COUNT(*) FROM run_events WHERE status = 'failed'"
            ).fetchone()[0]
        self.assertEqual(failures, 1)
        self.assertEqual(run_failures, 1)

    @patch.dict(
        os.environ,
        {"HF_TOKEN": "hf-secret", "HF_ADAPTER_REPO": "owner/adapters"},
        clear=True,
    )
    def test_deletion_failure_never_marks_run_completed(self):
        client = FakeClient(fail_delete=True)
        with self.assertRaisesRegex(RuntimeError, "deletion failed"):
            self.worker(client, FakeRunner()).run(
                run_id="run-1",
                job_id="job-1",
                commit=COMMIT,
                version="v1-smoke",
                limit=32,
                confirmation=CONFIRMATION,
            )
        with sqlite3.connect(self.ledger.path) as connection:
            statuses = [
                row[0]
                for row in connection.execute(
                    "SELECT status FROM run_events "
                    "WHERE run_id = 'run-1' AND stage = 'vultr_worker' ORDER BY occurred_at"
                )
            ]
        self.assertEqual(statuses, ["started", "failed"])

    def test_existing_managed_worker_blocks_another_instance(self):
        client = FakeClient(active=[{"id": "existing", "tags": ["dapta-managed"]}])
        with self.assertRaisesRegex(VultrError, "already exists"):
            self.worker(client, FakeRunner()).plan("job-1", COMMIT, limit=32)


if __name__ == "__main__":
    unittest.main()
