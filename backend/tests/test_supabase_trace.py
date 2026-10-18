import json
import unittest

import httpx

from dapta.supabase_trace import SupabaseTrace, UnconfiguredSupabase


class SupabaseTraceTests(unittest.TestCase):
    def test_each_call_sends_the_prompt_and_the_thought(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "query": request.url.query.decode(),
                    "body": json.loads(request.content),
                    "apikey": request.headers["apikey"],
                }
            )
            return httpx.Response(201)

        trace = SupabaseTrace(
            "https://example.supabase.co",
            "test-secret",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        trace.start(run_id="11111111-1111-1111-1111-111111111111", task="build a lora", skill_id="dcf")
        trace.record(
            run_id="11111111-1111-1111-1111-111111111111",
            stage="understood",
            agent="task_agent",
            prompt="Read the task",
            thought={"summary": "build a lora", "refusals": ["outside the task"]},
        )
        trace.finish(
            run_id="11111111-1111-1111-1111-111111111111",
            status="stored",
            device="CPU",
            artifact_uri="hf://example/adapter",
        )

        self.assertEqual(seen[0]["method"], "POST")
        self.assertEqual(seen[0]["path"], "/rest/v1/lora_runs")
        self.assertEqual(seen[0]["body"]["task"], "build a lora")
        self.assertEqual(seen[0]["apikey"], "test-secret")
        self.assertEqual(seen[1]["path"], "/rest/v1/lora_events")
        self.assertEqual(seen[1]["body"]["prompt"], "Read the task")
        self.assertEqual(seen[1]["body"]["thought"]["refusals"], ["outside the task"])
        self.assertEqual(seen[2]["method"], "PATCH")
        self.assertIn("run_id=eq.11111111-1111-1111-1111-111111111111", seen[2]["query"])
        self.assertEqual(seen[2]["body"]["status"], "stored")
        self.assertEqual(seen[2]["body"]["artifact_uri"], "hf://example/adapter")

    def test_missing_configuration_refuses_to_start(self):
        with self.assertRaisesRegex(RuntimeError, "SUPABASE_URL"):
            UnconfiguredSupabase().start(run_id="x", task="a task here", skill_id="skill")


if __name__ == "__main__":
    unittest.main()
