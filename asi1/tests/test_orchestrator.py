from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from asi1.orchestrator import LoraFactoryOrchestrator
from asi1.session_store import SessionStore


class Client:
    JOB_ID = "06138942-d2cc-4484-9dde-f09d26b29fea"

    def __init__(self):
        self.calls = []
        self.readiness = {
            "factory_autostart": True,
            "openrouter_configured": True,
            "devin_configured": True,
            "training_configured": True,
            "publishing_configured": True,
        }
        self.route = {
            "decision": "create_adapter",
            "reason": "No existing specialist matched",
        }
        self.adapters = [
            {
                "adapter_id": "receipt/qwen-v1",
                "version": "v1",
                "metadata": {"skill_prompt": "Extract and validate receipt fields."},
            },
            {"adapter_id": "nda/qwen-v1", "version": "v1"},
        ]
        self.job = {"job_id": self.JOB_ID, "status": "queued"}
        self.status = {
            "job_id": self.JOB_ID,
            "status": "training",
            "progress": {"percent": 50},
        }
        self.execution = {
            "output": {"subtotal": "10.00", "tax": "0.80", "total": "10.80"},
            "input_tokens": 20,
            "output_tokens": 8,
            "latency_ms": 1200,
        }

    def health(self):
        self.calls.append(("health",))
        return self.readiness

    def list_adapters(self):
        self.calls.append(("list",))
        return self.adapters

    def route_workflow(self, workflow):
        self.calls.append(("route", workflow))
        return self.route

    def create_adapter(self, *, workflow, idempotency_key, skill_id=None):
        self.calls.append(("create", workflow, idempotency_key, skill_id))
        return self.job

    def get_job_status(self, job_id):
        self.calls.append(("status", job_id))
        return self.status

    def execute_adapter(self, *, run_id, workflow, document_text, adapter_id):
        self.calls.append(("execute", run_id, workflow, document_text, adapter_id))
        return self.execution


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.client = Client()
        self.sessions = SessionStore(Path(self.temp.name) / "sessions.sqlite3")
        self.orchestrator = LoraFactoryOrchestrator(
            self.client,
            self.sessions,
            default_adapter_id="receipt/qwen-v1",
            factory_builds_enabled=True,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_create_requires_confirmation_before_factory_call(self):
        first = self.orchestrator.handle(
            "agent1", "Create a LoRA adapter that extracts synthetic receipts"
        )

        self.assertIn("Reply `confirm build`", first[0])
        self.assertFalse(any(call[0] == "route" for call in self.client.calls))
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))

        second = self.orchestrator.handle("agent1", "confirm build")

        self.assertIn(Client.JOB_ID, second[0])
        create_calls = [call for call in self.client.calls if call[0] == "create"]
        self.assertEqual(len(create_calls), 1)
        self.assertTrue(create_calls[0][2].startswith("asi1-"))

    def test_second_confirmation_never_duplicates_job(self):
        self.orchestrator.handle("agent1", "Build a LoRA adapter for receipt extraction")
        self.orchestrator.handle("agent1", "confirm build")

        reply = self.orchestrator.handle("agent1", "confirm build")

        self.assertIn("I can create", reply[0])
        self.assertEqual(
            len([call for call in self.client.calls if call[0] == "create"]), 1
        )

    def test_existing_adapter_is_selected_instead_of_training(self):
        self.client.route = {
            "decision": "existing_adapter",
            "adapter_id": "nda/qwen-v1",
            "reason": "matched",
        }

        first = self.orchestrator.handle(
            "agent1", "Create a LoRA adapter that extracts NDA fields"
        )
        self.assertIn("confirm build", first[0])

        reply = self.orchestrator.handle("agent1", "confirm build")

        self.assertIn("existing adapter `nda/qwen-v1`", reply[0])
        self.assertEqual(self.sessions.get("agent1").adapter_id, "nda/qwen-v1")
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))

    def test_status_uses_current_job_and_reports_real_progress(self):
        session = self.sessions.get("agent1")
        session.job_id = Client.JOB_ID
        self.sessions.save(session)

        reply = self.orchestrator.handle("agent1", "status")

        self.assertIn("training", reply[0])
        self.assertIn("50%", reply[0])

    def test_status_accepts_an_explicit_uuid_job_id(self):
        reply = self.orchestrator.handle(
            "agent1", f"status for {Client.JOB_ID}"
        )

        self.assertIn("training", reply[0])
        self.assertIn(("status", Client.JOB_ID), self.client.calls)

    def test_factory_readiness_blocks_job_creation(self):
        self.client.readiness["training_configured"] = False
        self.orchestrator.handle("agent1", "Build a LoRA adapter for receipt extraction")

        reply = self.orchestrator.handle("agent1", "confirm build")

        self.assertIn("GPU training", reply[0])
        self.assertIn("No factory job was created", reply[0])
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))

    def test_run_collects_document_then_executes_lora(self):
        prompt = self.orchestrator.handle("agent1", "run adapter receipt/qwen-v1")
        self.assertIn("Document:", prompt[0])

        reply = self.orchestrator.handle(
            "agent1", "Document: Merchant ACME, subtotal 10.00, tax 0.80, total 10.80"
        )

        self.assertIn("LoRA execution completed", reply[0])
        self.assertIn('"arithmetic_valid":true', reply[0])
        execute = next(call for call in self.client.calls if call[0] == "execute")
        self.assertEqual(
            execute[-2], "Merchant ACME, subtotal 10.00, tax 0.80, total 10.80"
        )
        self.assertEqual(execute[-1], "receipt/qwen-v1")
        self.assertEqual(execute[-3], "Extract and validate receipt fields.")

    def test_unknown_adapter_is_not_sent_to_the_worker(self):
        reply = self.orchestrator.handle("agent1", "run adapter missing/qwen-v1")

        self.assertIn("not registered", reply[0])
        self.assertFalse(any(call[0] == "execute" for call in self.client.calls))

    def test_document_is_not_persisted(self):
        self.orchestrator.handle("agent1", "run adapter receipt/qwen-v1")
        document = "PRIVATE-SYNTHETIC-DOCUMENT"
        self.orchestrator.handle("agent1", document)

        database = (Path(self.temp.name) / "sessions.sqlite3").read_bytes()
        self.assertNotIn(document.encode(), database)

    def test_cancel_never_starts_build(self):
        self.orchestrator.handle("agent1", "Create a LoRA adapter for receipt extraction")

        reply = self.orchestrator.handle("agent1", "cancel")

        self.assertIn("No adapter build was started", reply[0])
        self.assertFalse(any(call[0] == "create" for call in self.client.calls))

    def test_public_factory_builds_can_be_disabled(self):
        orchestrator = LoraFactoryOrchestrator(
            self.client,
            self.sessions,
            factory_builds_enabled=False,
        )

        reply = orchestrator.handle(
            "agent1", "Create a LoRA adapter for receipt extraction"
        )

        self.assertIn("disabled", reply[0])
        self.assertEqual(self.client.calls, [])

    def test_oversized_messages_are_rejected_before_execution(self):
        reply = self.orchestrator.handle("agent1", "x" * 500_001)

        self.assertIn("too large", reply[0])
        self.assertEqual(self.client.calls, [])


if __name__ == "__main__":
    unittest.main()
