from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from asi1.session_store import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_state_survives_new_store_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.sqlite3"
            first = SessionStore(path)
            session = first.get("agent1")
            session.job_id = "job-123"
            session.adapter_id = "receipt/qwen-v1"
            first.save(session)

            restored = SessionStore(path).get("agent1")

            self.assertEqual(restored.job_id, "job-123")
            self.assertEqual(restored.adapter_id, "receipt/qwen-v1")


if __name__ == "__main__":
    unittest.main()
