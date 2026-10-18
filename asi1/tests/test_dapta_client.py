from __future__ import annotations

import unittest

from asi1.dapta_client import DaptaClient, DaptaClientError


class Transport:
    def __init__(self):
        self.calls = []
        self.responses = []

    def request(self, method, url, *, headers, payload=None, timeout):
        self.calls.append(
            {"method": method, "url": url, "headers": dict(headers),
             "payload": payload, "timeout": timeout}
        )
        return self.responses.pop(0)


class DaptaClientTests(unittest.TestCase):
    def setUp(self):
        self.transport = Transport()
        self.client = DaptaClient(
            controller_url="http://controller:8000/",
            controller_token="controller-secret",
            qwen_url="http://worker:8002/",
            qwen_token="worker-secret",
            transport=self.transport,
            timeout=12,
        )

    def test_create_adapter_uses_existing_agent_tool(self):
        self.transport.responses.append({"job_id": "job-1", "status": "queued"})

        result = self.client.create_adapter(
            workflow="Extract receipts into an expense entry",
            idempotency_key="asi1-key-123",
        )

        self.assertEqual(result["job_id"], "job-1")
        call = self.transport.calls[0]
        self.assertEqual(
            call["url"], "http://controller:8000/v1/agent/tools/create_adapter"
        )
        self.assertEqual(call["headers"], {"X-Dapta-Token": "controller-secret"})
        self.assertEqual(call["payload"]["idempotency_key"], "asi1-key-123")

    def test_health_is_read_without_a_request_body(self):
        self.transport.responses.append({"factory_autostart": True})

        result = self.client.health()

        self.assertTrue(result["factory_autostart"])
        call = self.transport.calls[0]
        self.assertEqual(call["method"], "GET")
        self.assertEqual(call["url"], "http://controller:8000/health")
        self.assertIsNone(call["payload"])

    def test_execute_adapter_uses_private_worker_and_bearer_token(self):
        self.transport.responses.append(
            {"output": {"total": "4.20"}, "input_tokens": 10, "output_tokens": 4}
        )

        result = self.client.execute_adapter(
            run_id="asi1-run",
            workflow="Extract a receipt",
            document_text="Synthetic receipt",
            adapter_id="receipt/v1",
        )

        self.assertEqual(result["output"]["total"], "4.20")
        call = self.transport.calls[0]
        self.assertEqual(call["url"], "http://worker:8002/generate")
        self.assertEqual(call["headers"], {"Authorization": "Bearer worker-secret"})
        self.assertEqual(call["payload"]["adapter_id"], "receipt/v1")

    def test_invalid_adapter_list_is_rejected(self):
        self.transport.responses.append({"not": "a list"})

        with self.assertRaisesRegex(DaptaClientError, "invalid adapter list"):
            self.client.list_adapters()


if __name__ == "__main__":
    unittest.main()
