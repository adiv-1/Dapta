import tempfile
import unittest
from pathlib import Path

import httpx

from dapta.jobs import JobStore
from dapta.supabase_store import SupabaseFactoryStore


class SupabaseStoreTests(unittest.TestCase):
    def test_sync_upserts_sanitized_job_events_and_agent_artifacts(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append({
                "path": request.url.path,
                "query": str(request.url.query),
                "headers": dict(request.headers),
                "body": request.read().decode(),
            })
            return httpx.Response(201)

        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory) / "dapta.sqlite3")
            job = store.create_job(
                workflow="Build a repetitive legal intake specialist.",
                idempotency_key="supabase-test-001",
            )
            store.put_artifact(job["job_id"], "skill_spec.json", {
                "name": "Legal Intake",
                "api_key": "must-not-survive",
            })
            job = store.get_job(job["job_id"])
            mirror = SupabaseFactoryStore(
                "https://example.supabase.co",
                "server-secret",
                client=httpx.Client(transport=httpx.MockTransport(handler)),
            )
            mirror.sync_job(job, store)

        self.assertEqual(
            [entry["path"] for entry in seen],
            [
                "/rest/v1/dapta_factory_runs",
                "/rest/v1/dapta_factory_events",
                "/rest/v1/dapta_factory_artifacts",
            ],
        )
        self.assertEqual(seen[0]["headers"]["apikey"], "server-secret")
        self.assertNotIn("authorization", seen[0]["headers"])
        self.assertIn('"api_key":"[redacted]"', seen[2]["body"])
        self.assertNotIn("must-not-survive", seen[2]["body"])


if __name__ == "__main__":
    unittest.main()
