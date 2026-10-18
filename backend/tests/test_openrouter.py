import sqlite3
import tempfile
import unittest
from pathlib import Path

from dapta.ledger import UsageLedger
from dapta.openrouter import FactoryAgents, OpenRouterClient, OpenRouterError, _response_json


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, *, headers, payload=None):
        self.calls.append(
            {"method": method, "url": url, "headers": dict(headers), "payload": payload}
        )
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class OpenRouterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "ledger.sqlite3"
        self.ledger = UsageLedger(self.path)
        self.ledger.create_run(
            run_id="job-1",
            run_type="adapter_build",
            job_id="job-1",
            started_at="2026-10-17T08:00:00Z",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_json_call_reconciles_generation_and_ledger(self):
        transport = FakeTransport(
            [
                {
                    "id": "gen-test",
                    "model": "moonshotai/kimi-k3",
                    "choices": [
                        {"message": {"role": "assistant", "content": '{"result":"ok"}'}}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cost": "0.001",
                    },
                },
                {
                    "data": {
                        "id": "gen-test",
                        "request_id": "req-test",
                        "provider_name": "Moonshot AI",
                        "model": "moonshotai/kimi-k3",
                        "native_tokens_prompt": 9,
                        "native_tokens_completion": 4,
                        "native_tokens_reasoning": 2,
                        "total_cost": "0.001234",
                        "latency": 100,
                    }
                },
            ]
        )
        client = OpenRouterClient(
            api_key="test-key",
            ledger=self.ledger,
            transport=transport,
            sleep=lambda _: None,
        )
        result = client.complete_json(
            run_id="job-1",
            stage="skill_architect",
            agent_role="skill_architect",
            model="moonshotai/kimi-k3",
            messages=[{"role": "user", "content": "workflow"}],
            max_tokens=100,
        )

        self.assertEqual(result, {"result": "ok"})
        self.assertEqual(len(transport.calls), 2)
        request = transport.calls[0]
        self.assertEqual(request["payload"]["usage"], {"include": True})
        self.assertEqual(request["payload"]["response_format"], {"type": "json_object"})
        self.assertEqual(request["payload"]["trace"]["trace_id"], "job-1")
        self.assertEqual(request["headers"]["X-OpenRouter-Metadata"], "enabled")
        self.assertTrue(transport.calls[1]["url"].endswith("/generation?id=gen-test"))

        summary = self.ledger.run_summary("job-1")
        self.assertEqual(summary["model_request_count"], 1)
        self.assertEqual(summary["prompt_tokens"], 9)
        self.assertEqual(summary["completion_tokens"], 4)
        self.assertEqual(summary["reasoning_tokens"], 2)
        self.assertEqual(summary["model_cost_usd"], "0.001234")

    def test_failed_request_is_recorded_without_prompt_content(self):
        transport = FakeTransport([OpenRouterError("payment required", status=402)])
        client = OpenRouterClient(
            api_key="test-key",
            ledger=self.ledger,
            transport=transport,
            sleep=lambda _: None,
        )
        with self.assertRaises(OpenRouterError):
            client.complete_json(
                run_id="job-1",
                stage="skill_architect",
                agent_role="skill_architect",
                model="moonshotai/kimi-k3",
                messages=[{"role": "user", "content": "private workflow"}],
                max_tokens=100,
            )

        with sqlite3.connect(self.path) as connection:
            status, metadata = connection.execute(
                "SELECT status, metadata_json FROM model_calls"
            ).fetchone()
        self.assertEqual(status, "failed")
        self.assertNotIn("private workflow", metadata)
        self.assertEqual(self.ledger.run_summary("job-1")["model_cost_usd"], "0")

    def test_structured_text_content_is_parsed(self):
        result = _response_json({
            "choices": [{
                "message": {
                    "content": [{"type": "text", "text": '{"result":"ok"}'}],
                },
            }],
        })
        self.assertEqual(result, {"result": "ok"})

    def test_object_content_is_accepted_directly(self):
        result = _response_json({
            "choices": [{"message": {"content": {"result": "ok"}}}],
        })
        self.assertEqual(result, {"result": "ok"})

    def test_reasoning_budget_exhaustion_has_a_specific_error(self):
        with self.assertRaisesRegex(OpenRouterError, "reasoning exhausted"):
            _response_json({
                "choices": [{"message": {"content": None}}],
                "usage": {
                    "completion_tokens": 12000,
                    "completion_tokens_details": {"reasoning_tokens": 12000},
                },
            })

    def test_data_builder_disables_reasoning_to_preserve_output_budget(self):
        class Client:
            def complete_json(inner_self, **kwargs):
                return kwargs

        agents = FactoryAgents(Client(), factory_model="moonshotai/kimi-k3", critic_model="critic")
        result = agents.build_dataset(
            run_id="job-1",
            workflow="clean CSV",
            skill_contract={"output_schema": {}},
        )
        self.assertEqual(result["reasoning_effort"], "none")


if __name__ == "__main__":
    unittest.main()
