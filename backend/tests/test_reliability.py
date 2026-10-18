"""Regression tests for reliability bugs found while instrumenting Dapta with Sentry."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dapta.comparisons import ComparisonService, ComparisonStore
from dapta.config import Settings
from dapta.ledger import UsageLedger

ADAPTER = {"adapter_id": "nda-intake/qwen3-vl-8b-v1", "skill_id": "nda-intake", "metadata": {}}


class InterruptedComparisonTests(unittest.TestCase):
    """A controller restart used to leave in-flight comparisons queued/running forever.

    The run executes in a daemon thread of the controller process. If that process stops (deploy,
    crash, OOM kill), nothing ever finishes the run, and the frontend keeps polling every 1.5 s while
    any run is not terminal: a permanently "running" comparison and an endless poll loop.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "dapta.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def start_run_then_lose_the_process(self) -> str:
        service = ComparisonService(store=ComparisonStore(self.path), ledger=UsageLedger(self.path),
                                    settings=Settings.from_env())
        # The background thread never runs: the process is gone before it executes.
        run = service.start(workflow="Summarise this NDA", document_text="synthetic", adapter=ADAPTER,
                            idempotency_key="restart-001", background=lambda *args: None)
        return run["run_id"]

    def test_runs_interrupted_by_a_restart_are_failed_on_startup(self):
        run_id = self.start_run_then_lose_the_process()
        restarted = ComparisonStore(self.path)
        self.assertEqual(restarted.recover_interrupted(), 1)
        run = restarted.get(run_id)
        self.assertEqual(run["status"], "failed")
        self.assertIn("controller restarted", run["error_message"])
        self.assertEqual(restarted.recover_interrupted(), 0, "recovery is idempotent")

    def test_the_controller_recovers_them_when_it_starts(self):
        from fastapi.testclient import TestClient

        from dapta.api import create_app

        run_id = self.start_run_then_lose_the_process()
        environment = {"DAPTA_CONTROL_TOKEN": "control-token", "DAPTA_FACTORY_AUTOSTART": "false",
                       "OPENROUTER_API_KEY": "", "JEV_API_KEY": "", "DEVIN_API_KEY": "", "SUPABASE_URL": ""}
        with patch.dict(os.environ, environment):
            client = TestClient(create_app(self.path))
            run = client.get(f"/v1/runs/{run_id}", headers={"X-Dapta-Token": "control-token"}).json()
        self.assertEqual(run["status"], "failed")


if __name__ == "__main__":
    unittest.main()
