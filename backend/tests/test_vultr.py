import os
import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dapta.ledger import UsageLedger
from dapta.vultr import (
    CONFIRMATION,
    SshRunner,
    VultrError,
    VultrWorker,
    WorkerConfig,
    destroy_managed,
    managed_status,
)


COMMIT = "a" * 40


class FakeClient:
    def __init__(self, *, active=None):
        self.active = active or []
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

    def prepare(self, host, job_id, *, limit, timeout):
        self.calls.append(("prepare", limit))

    def read_validation(self, host, job_id, *, timeout):
        self.calls.append("read_validation")
        return {
            "passed": True,
            "validator_version": "receipt-validator-v1",
            "counts": {"train": 10, "dev": 2, "test": 2},
            "hashes": {"splits": {"test": "abc"}},
        }

    def train(self, host, job_id, version, *, timeout):
        self.calls.append("train")
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
        self.assertEqual(client.created["block_devices"], [])
        self.assertEqual(client.created["sshkey_id"], ["ssh-key-1"])
        with sqlite3.connect(self.ledger.path) as connection:
            events = {
                row[0]
                for row in connection.execute(
                    "SELECT event_type FROM compute_events WHERE allocation_id = 'vultr:instance-1'"
                )
            }
            cost_count = connection.execute("SELECT COUNT(*) FROM compute_costs").fetchone()[0]
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
        self.assertEqual(failures, 1)

    @patch.dict(
        os.environ,
        {"HF_TOKEN": "hf-secret", "HF_ADAPTER_REPO": "owner/adapters"},
        clear=True,
    )
    def test_progress_callback_errors_never_block_deletion(self):
        client = FakeClient()
        seen = []

        def on_event(event):
            seen.append(event)
            raise RuntimeError("observer broke")

        worker = VultrWorker(
            self.config,
            client,
            self.ledger,
            FakeRunner(),
            sleep=lambda _: None,
            monotonic=lambda: 100,
            on_event=on_event,
        )
        result = worker.run(
            run_id="run-1",
            job_id="job-1",
            commit=COMMIT,
            version="v1-smoke",
            limit=32,
            confirmation=CONFIRMATION,
        )
        self.assertEqual(result["status"], "completed")
        self.assertTrue(client.deleted)
        self.assertLess(seen.index("workload_started"), seen.index("dataset_validated"))
        self.assertLess(seen.index("dataset_validated"), seen.index("workload_finished"))
        self.assertIn("destroyed", seen)

    @patch.dict(
        os.environ,
        {"HF_TOKEN": "hf-secret", "HF_ADAPTER_REPO": "owner/adapters"},
        clear=True,
    )
    def test_unknown_ledger_run_fails_before_provisioning(self):
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "unknown run"):
            self.worker(client, FakeRunner()).run(
                run_id="never-created",
                job_id="job-1",
                commit=COMMIT,
                version="v1-smoke",
                limit=32,
                confirmation=CONFIRMATION,
            )
        self.assertIsNone(client.created)

    @patch.dict(
        os.environ,
        {"HF_TOKEN": "hf-secret", "HF_ADAPTER_REPO": "owner/adapters"},
        clear=True,
    )
    def test_ledger_failure_during_teardown_still_deletes_the_vm(self):
        client = FakeClient()
        original = self.ledger.record_compute_event

        def flaky(**kwargs):
            if kwargs["event_type"] in {"destroy_requested", "destroyed"}:
                raise sqlite3.OperationalError("database is locked")
            return original(**kwargs)

        self.ledger.record_compute_event = flaky
        with self.assertRaisesRegex(RuntimeError, "deleted, but its ledger record is incomplete"):
            self.worker(client, FakeRunner()).run(
                run_id="run-1",
                job_id="job-1",
                commit=COMMIT,
                version="v1-smoke",
                limit=32,
                confirmation=CONFIRMATION,
            )
        self.assertTrue(client.deleted)

        client = FakeClient()
        client.create_instance = lambda payload: {
            "id": "instance-2",
            "label": payload["label"],
            "date_created": "2026-10-17T12:00:00Z",
        }
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.worker(client, FakeRunner(fail_train=True)).run(
                run_id="run-1",
                job_id="job-2",
                commit=COMMIT,
                version="v1-smoke",
                limit=32,
                confirmation=CONFIRMATION,
            )
        self.assertTrue(client.deleted, "the original failure is kept and the VM is still deleted")

    @patch.dict(
        os.environ,
        {"HF_TOKEN": "hf-secret", "HF_ADAPTER_REPO": "owner/adapters"},
        clear=True,
    )
    def test_failed_compute_registration_still_deletes_the_vm(self):
        self.ledger.register_compute = lambda **kwargs: (_ for _ in ()).throw(
            sqlite3.IntegrityError("UNIQUE constraint failed")
        )
        client = FakeClient()
        with self.assertRaises(sqlite3.IntegrityError):
            self.worker(client, FakeRunner()).run(
                run_id="run-1",
                job_id="job-1",
                commit=COMMIT,
                version="v1-smoke",
                limit=32,
                confirmation=CONFIRMATION,
            )
        self.assertIsNotNone(client.created)
        self.assertTrue(client.deleted)

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_hf_settings_fail_before_provisioning(self):
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            self.worker(client, FakeRunner()).run(
                run_id="run-1",
                job_id="job-1",
                commit=COMMIT,
                version="v1-smoke",
                limit=32,
                confirmation=CONFIRMATION,
                hf_repo="owner/adapters",
            )
        self.assertIsNone(client.created)

    def test_explicit_hf_settings_are_what_the_worker_uploads(self):
        client = FakeClient()
        runner = FakeRunner()
        uploaded = {}
        runner.upload_environment = lambda host, values: uploaded.update(values)
        with patch.dict(os.environ, {"HF_ADAPTER_REPO": "env/other"}, clear=True):
            self.worker(client, runner).run(
                run_id="run-1",
                job_id="job-1",
                commit=COMMIT,
                version="v1-smoke",
                limit=32,
                confirmation=CONFIRMATION,
                hf_repo="owner/adapters",
                hf_token="hf-secret",
            )
        self.assertEqual(uploaded["HF_ADAPTER_REPO"], "owner/adapters")
        self.assertEqual(uploaded["HF_TOKEN"], "hf-secret")

    def test_status_lists_only_dapta_instances(self):
        client = FakeClient(
            active=[
                {"id": "leaked", "tags": ["dapta-managed"], "status": "active"},
                {"id": "serving-box", "tags": ["serving"], "status": "active"},
            ]
        )
        self.assertEqual([item["id"] for item in managed_status(client)], ["leaked"])

    def test_destroy_requires_confirmation_and_the_managed_tag(self):
        class Client(FakeClient):
            def __init__(self, tags):
                super().__init__()
                self.tags = tags

            def get_instance(self, instance_id):
                if self.deleted:
                    return None
                return {"id": instance_id, "tags": self.tags}

        client = Client(["dapta-managed"])
        with self.assertRaisesRegex(ValueError, "confirm-destroy"):
            destroy_managed(client, self.ledger, instance_id="leaked", confirmation="yes")
        self.assertFalse(client.deleted)

        serving = Client(["serving"])
        with self.assertRaisesRegex(VultrError, "refusing"):
            destroy_managed(serving, self.ledger, instance_id="serving-box", confirmation="DESTROY")
        self.assertFalse(serving.deleted)

        result = destroy_managed(
            client, self.ledger, instance_id="leaked", confirmation="DESTROY", sleep=lambda _: None
        )
        self.assertTrue(client.deleted)
        self.assertEqual(result["status"], "deleted")
        self.assertFalse(result["ledger_updated"], "an unregistered VM is still deleted")
        self.assertEqual(
            destroy_managed(client, self.ledger, instance_id="leaked", confirmation="DESTROY")["status"],
            "already_deleted",
        )

    def test_remote_scripts_wait_for_apt_and_install_pinned_stack(self):
        scripts = []
        runner = SshRunner(self.config, Path(self.temp.name))
        runner._run = lambda host, script, timeout: scripts.append(script)
        try:
            runner.prepare("8.8.8.8", "job-1", limit=8, timeout=60)
            runner.train("8.8.8.8", "job-1", "v1-smoke", timeout=60)
        finally:
            runner.close()
        prepare, train = scripts
        self.assertIn("DPkg::Lock::Timeout", prepare)
        self.assertIn("requirements-data.txt", prepare)
        self.assertIn("requirements-train.txt", train)
        self.assertIn("torch.cuda.is_available()", train)
        self.assertLess(train.index("cuda.is_available"), train.index("dapta.train"))

    def test_pinned_requirements_are_exact(self):
        backend = Path(__file__).resolve().parents[1]
        for name in ("requirements-train.txt", "requirements-data.txt"):
            lines = [
                line.strip()
                for line in (backend / name).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
            self.assertTrue(lines)
            self.assertTrue(all("==" in line for line in lines), name)
        train = (backend / "requirements-train.txt").read_text(encoding="utf-8")
        data = (backend / "requirements-data.txt").read_text(encoding="utf-8")
        for line in data.splitlines():
            if line and not line.startswith("#"):
                self.assertIn(line, train, "data and training stacks must agree")

    def test_existing_managed_worker_blocks_another_instance(self):
        client = FakeClient(active=[{"id": "existing", "tags": ["dapta-managed"]}])
        with self.assertRaisesRegex(VultrError, "already exists"):
            self.worker(client, FakeRunner()).plan("job-1", COMMIT, limit=32)


if __name__ == "__main__":
    unittest.main()
