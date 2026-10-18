import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dapta.cancellation import BuildCancelled, cancellation_scope
from dapta.factory_runtime import FactoryRunner, RunPodTrainingBackend
from dapta.jobs import AdapterJobOrchestrator, InvalidTransition, JobStore, JOB_STAGES
from dapta.ledger import UsageLedger


class CancellationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "jobs.sqlite3"
        self.store = JobStore(self.path)
        self.ledger = UsageLedger(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def make_job(self):
        orchestrator = AdapterJobOrchestrator(self.store, self.ledger)
        job = orchestrator.create_job(workflow="Create a small routing specialist", idempotency_key=os.urandom(12).hex())
        return job, orchestrator

    def wait_stopped(self, job_id, runner):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with runner._cancel_lock:
                pending = job_id in runner._cancelling
            if not pending:
                return self.store.get_job(job_id)
            time.sleep(.01)
        self.fail("cancellation cleanup did not finish")

    def test_stop_is_durable_idempotent_and_covers_every_active_stage(self):
        for status in JOB_STAGES:
            if status in {"ready", "failed", "cancelled", "cancelling"}:
                continue
            job, orchestrator = self.make_job()
            # Set up each stage without initiating paid work.
            with self.store._connect(immediate=True) as connection:
                connection.execute("UPDATE jobs SET status=? WHERE job_id=?", (status, job["job_id"]))
            self.store.request_cancel(job["job_id"])
            self.assertEqual(self.store.request_cancel(job["job_id"])["status"], "cancelling")
            self.assertEqual(JobStore(self.path).get_job(job["job_id"])["status"], "cancelling")
            with self.assertRaises(BuildCancelled):
                self.store.claim_stage(job["job_id"], "unexpected_paid_stage")
            self.assertEqual(self.store.finish_cancel(job["job_id"])["status"], "cancelled")
            with self.assertRaises(InvalidTransition):
                self.store.retry_failed(job["job_id"])
            self.assertEqual(self.store.fail(job["job_id"], code="LateError", message="late")["status"], "cancelled")

    def test_stop_waits_for_inflight_stage_and_discards_late_result_transition(self):
        job, orchestrator = self.make_job()
        entered = threading.Event(); release = threading.Event(); errors = []
        class Agents:
            def architect(self, **kwargs):
                entered.set(); release.wait(5)
                return {"skill_spec": {"name": "example"}, "output_schema": {"type": "object"}, "evaluation_plan": {}}
        orchestrator.agents = Agents()
        runner = FactoryRunner(orchestrator, self.store)
        def run():
            try: orchestrator.advance(job["job_id"])
            except BuildCancelled: pass
            except Exception as error: errors.append(error)
        worker = threading.Thread(target=run)
        worker.start(); self.assertTrue(entered.wait(3))
        runner.cancel(job["job_id"])
        self.assertEqual(self.store.finish_cancel(job["job_id"])["status"], "cancelling")
        release.set(); worker.join(5)
        self.assertFalse(worker.is_alive()); self.assertEqual(errors, [])
        stopped = self.wait_stopped(job["job_id"], runner)
        self.assertEqual(stopped["status"], "cancelled")
        self.assertEqual(stopped["stage_attempts"][0]["status"], "cancelled")
        self.assertFalse(any(event["to_status"] == "integrating" for event in stopped["events"]))
        self.assertFalse(runner.run_once())

    def test_late_devin_session_is_terminated_before_cancellation_completes(self):
        job, orchestrator = self.make_job()
        self.store.transition(job["job_id"], "architecting")
        self.store.transition(job["job_id"], "integrating")
        self.store.claim_stage(job["job_id"], "skill_integration")
        stopped = []
        class Client:
            def terminate_session(self, session_id): stopped.append(session_id)
            def get_session(self, session_id): return SimpleNamespace(session_id=session_id, status="exit")
        orchestrator.integrator = SimpleNamespace(client=Client())
        runner = FactoryRunner(orchestrator, self.store)
        runner.cancel(job["job_id"])
        self.store.set_artifact(job["job_id"], "devin_session.json", {"session_id": "late-created-session"})
        self.store.complete_stage(job["job_id"], "skill_integration", status="cancelled")
        self.assertEqual(self.wait_stopped(job["job_id"], runner)["status"], "cancelled")
        self.assertEqual(stopped, ["late-created-session"])

    def test_provider_cleanup_failure_stays_stopping_and_can_be_retried(self):
        job, orchestrator = self.make_job()
        self.store.set_artifact(job["job_id"], "devin_session.json", {"session_id": "session"})
        class Client:
            fail = True
            def terminate_session(self, session_id):
                if self.fail: raise RuntimeError("provider unavailable")
            def get_session(self, session_id): return SimpleNamespace(session_id=session_id, status="exit")
        client = Client(); orchestrator.integrator = SimpleNamespace(client=client)
        runner = FactoryRunner(orchestrator, self.store)
        runner.cancel(job["job_id"])
        pending = self.wait_stopped(job["job_id"], runner)
        self.assertEqual(pending["status"], "cancelling")
        self.assertIn("cleanup_error", pending["cancellation"])
        client.fail = False; runner.cancel(job["job_id"])
        self.assertEqual(self.wait_stopped(job["job_id"], runner)["status"], "cancelled")

    def test_gpu_worker_is_terminated_reaped_and_inference_reloaded(self):
        lifecycle = SimpleNamespace(unload=lambda: actions.append("unload"), load=lambda: actions.append("load"))
        actions = []; processes = []; cancellation = threading.Event()
        real_popen = subprocess.Popen
        def spawn(*args, **kwargs):
            process = real_popen([sys.executable, "-u", "-c", "import time; print('test worker ready',flush=True); time.sleep(60)"], **kwargs)
            processes.append(process); cancellation.set()
            return process
        def check():
            if cancellation.is_set(): raise BuildCancelled("test stop")
        backend = RunPodTrainingBackend(ledger=None, lifecycle=lifecycle)
        with patch("dapta.factory_runtime.subprocess.Popen", side_effect=spawn), cancellation_scope(check):
            with self.assertRaises(BuildCancelled):
                backend._execute_gpu_worker(bundle_path=Path("unused"), result_path=Path("unused-result"), training_clock=time.monotonic(), on_progress=lambda _: None)
        self.assertEqual(actions, ["unload", "load"])
        self.assertIsNotNone(processes[0].poll())

    def test_cancel_route_requires_auth_and_unknown_job_is_404(self):
        from fastapi.testclient import TestClient
        from dapta.api import create_app
        with patch.dict(os.environ, {"DAPTA_CONTROL_TOKEN": "test-control", "DAPTA_FACTORY_AUTOSTART": "false"}):
            client = TestClient(create_app(self.path))
            job, _ = self.make_job()
            path = f"/v1/jobs/{job['job_id']}/cancel"
            self.assertEqual(client.post(path).status_code, 401)
            runner = client.app.state.factory_runner
            response = client.post(path, headers={"X-Dapta-Token": "test-control"})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(self.wait_stopped(job["job_id"], runner)["status"], "cancelled")
            self.assertEqual(client.post("/v1/jobs/missing/cancel", headers={"X-Dapta-Token": "test-control"}).status_code, 404)

    def test_cancelled_publication_cannot_register_an_adapter(self):
        job, _ = self.make_job()
        self.store.request_cancel(job["job_id"])
        with self.assertRaises(BuildCancelled):
            self.store.register_adapter(adapter_id="generated/test/v1", skill_id="test", version="v1", base_model="base", artifact_uri="hf://test", job_id=job["job_id"])
        self.assertEqual(self.store.list_adapters(), [])

    def test_orphan_recovery_refuses_reused_pid_and_never_signals_inference(self):
        job, _ = self.make_job()
        self.store.set_artifact(job["job_id"], "gpu_worker.json", {"restore_required": True, "pid": 123, "start_time": "old", "bundle_path": "/owned/bundle.json"})
        backend = RunPodTrainingBackend(ledger=self.ledger, lifecycle=SimpleNamespace(load=lambda: None))
        with patch("dapta.factory_runtime._process_start", return_value="new"), patch.object(Path, "read_bytes", return_value=b"python\0-m\0dapta.factory_gpu\0--bundle\0/owned/bundle.json\0"), patch("dapta.factory_runtime.os.killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                backend.recover_cancelled_worker(job["job_id"], self.store)
            kill.assert_not_called()
