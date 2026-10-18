import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@unittest.skipUnless(HAS_FASTAPI, "FastAPI test dependencies are not installed")
class ApiTests(unittest.TestCase):
    def test_job_and_read_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "dapta.sqlite3"
            environment = {
                "DAPTA_LEDGER_PATH": str(database),
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_FACTORY_MODEL": "moonshotai/kimi-k3",
                "OPENROUTER_CRITIC_MODEL": "openai/gpt-oss-120b",
            }
            with patch.dict(os.environ, environment, clear=False):
                from dapta.api import create_app

                client = TestClient(create_app(database))
                created = client.post(
                    "/jobs",
                    headers={"Idempotency-Key": "api-test-001"},
                    json={
                        "workflow": "Turn receipt images into a structured expense report.",
                        "skill_id": "expense-report",
                    },
                )
                self.assertEqual(created.status_code, 202)
                job_id = created.json()["job_id"]
                self.assertEqual(client.get(f"/jobs/{job_id}").status_code, 200)
                self.assertEqual(client.get("/adapters").json(), [])
                self.assertEqual(client.get("/benchmarks").json(), [])
                self.assertEqual(client.post("/compare").status_code, 501)

    def test_v1_seed_registry_auth_and_agent_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "dapta.sqlite3"
            environment = {
                "DAPTA_LEDGER_PATH": str(database),
                "DAPTA_CONTROL_TOKEN": "controller-test-token",
                "DAPTA_SEED_DEMO_DATA": "true",
            }
            with patch.dict(os.environ, environment, clear=False):
                from dapta.api import create_app

                client = TestClient(create_app(database))
                self.assertEqual(client.get("/v1/adapters").status_code, 401)
                headers = {"X-Dapta-Token": "controller-test-token"}
                adapters = client.get("/v1/adapters", headers=headers).json()
                self.assertEqual(adapters[0]["adapter_id"], "nda-intake/qwen3-vl-8b-v1")
                benchmarks = client.get("/v1/benchmarks", headers=headers).json()
                self.assertEqual(benchmarks[0]["test_documents"], 83)
                tools = client.get("/v1/agent/tools", headers=headers).json()
                self.assertEqual(
                    {tool["name"] for tool in tools},
                    {"list_adapters", "run_comparison", "create_adapter", "get_job_status"},
                )

    def test_agent_training_requires_the_access_file_and_stays_small(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "dapta.sqlite3"
            access = root / "access.json"
            environment = {
                "DAPTA_LEDGER_PATH": str(database),
                "DAPTA_CONTROL_TOKEN": "controller-test-token",
                "DAPTA_FACTORY_AUTOSTART": "false",
                "DAPTA_TRAINING_ACCESS_PATH": str(access),
            }
            with patch.dict(os.environ, environment, clear=False):
                from dapta.api import create_app
                from dapta.jobs import JobStore

                client = TestClient(create_app(database))
                headers = {"X-Dapta-Token": "controller-test-token"}
                body = {
                    "idempotency_key": "smoke-access-001",
                    "arguments": {
                        "workflow": "Classify synthetic receipts into expense categories.",
                        "idempotency_key": "smoke-access-001",
                    },
                }
                closed = client.post("/v1/agent/tools/create_adapter", headers=headers, json=body)
                self.assertEqual(closed.status_code, 403)
                operator = client.post(
                    "/jobs",
                    headers={"Idempotency-Key": "operator-job-001"},
                    json={"workflow": "Turn receipt images into a structured expense report."},
                )
                self.assertEqual(operator.status_code, 202)

                access.write_text('{"enabled":true,"expires_at":9999999999}\n', encoding="utf-8")
                opened = client.post("/v1/agent/tools/create_adapter", headers=headers, json=body)
                self.assertEqual(opened.status_code, 200)
                limits = JobStore(database).get_artifact(opened.json()["job_id"], "training_limits.json")
                self.assertEqual(limits["profile"], "smoke")
                self.assertEqual(limits["max_examples"], 10)
                self.assertEqual(limits["max_train_seconds"], 120)


if __name__ == "__main__":
    unittest.main()
