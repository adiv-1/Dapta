import unittest

from dapta.devin import _session


class DevinResponseTests(unittest.TestCase):
    def test_parses_waiting_status_and_plural_pull_requests(self):
        parsed = _session({
            "session_id": "session-1",
            "status": "running",
            "status_detail": "waiting_for_user",
            "pull_requests": [{"html_url": "https://github.com/example/repo/pull/7"}],
            "on_demand_usage": 2.5,
        })

        self.assertEqual(parsed.status_detail, "waiting_for_user")
        self.assertEqual(parsed.pull_request_url, "https://github.com/example/repo/pull/7")
        self.assertEqual(parsed.on_demand_usage, 2.5)


if __name__ == "__main__":
    unittest.main()
