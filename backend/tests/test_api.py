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
                missing_agent = client.post(
                    "/lora",
                    json={
                        "task": "Turn receipt images into a structured expense report.",
                        "skill_id": "receipt-to-expense",
                    },
                )
                self.assertEqual(missing_agent.status_code, 503)


if __name__ == "__main__":
    unittest.main()
