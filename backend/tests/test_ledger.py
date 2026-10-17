import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from dapta.ledger import LedgerConflict, UsageLedger


START = "2026-10-17T01:00:00Z"
END = "2026-10-17T01:00:02Z"


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "ledger.sqlite3"
        self.ledger = UsageLedger(self.path)
        self.ledger.create_run(
            run_id="benchmark-1",
            run_type="benchmark",
            job_id="job-1",
            benchmark_id="expensebench-v1",
            started_at=START,
            metadata={"code_commit": "abc123", "dataset_hash": "def456"},
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_costs_and_tokens_remain_separate_by_condition(self):
        self.ledger.record_model_call(
            call_id="call-lora",
            run_id="benchmark-1",
            stage="compare",
            agent_role="candidate",
            benchmark_case_id="receipt-001",
            condition="qwen_lora",
            provider="vultr",
            requested_model="Qwen/Qwen3-VL-8B-Instruct",
            started_at=START,
            completed_at=END,
            prompt_tokens=101,
            completion_tokens=20,
            cost_usd="0.10",
            cost_source="calculated",
        )
        self.ledger.record_model_call(
            call_id="call-frontier",
            run_id="benchmark-1",
            stage="compare",
            agent_role="baseline",
            benchmark_case_id="receipt-001",
            condition="frontier",
            provider="openrouter",
            requested_model="frontier/model",
            started_at=START,
            completed_at=END,
            prompt_tokens=503,
            completion_tokens=41,
            cost_usd="0.20",
            cost_source="provider_reported",
        )

        summary = self.ledger.run_summary("benchmark-1")

        self.assertEqual(summary["model_request_count"], 2)
        self.assertEqual(summary["prompt_tokens"], 604)
        self.assertEqual(summary["model_cost_usd"], "0.3")
        self.assertEqual(
            {item["condition"]: item["cost_usd"] for item in summary["conditions"]},
            {"frontier": "0.2", "qwen_lora": "0.1"},
        )

    def test_openrouter_generation_metadata_is_attributed_to_one_call(self):
        self.ledger.record_openrouter_call(
            call_id="openrouter-call-1",
            run_id="benchmark-1",
            stage="skill_architect",
            agent_role="skill_architect",
            benchmark_case_id="workflow-001",
            condition="kimi_k3",
            requested_model="moonshotai/kimi-k3",
            started_at=START,
            completed_at=END,
            response={
                "id": "gen-1",
                "model": "moonshotai/kimi-k3",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "total_tokens": 125,
                    "cost": 0.001,
                },
            },
            generation={
                "data": {
                    "id": "gen-1",
                    "request_id": "req-1",
                    "provider_name": "Moonshot AI",
                    "native_tokens_prompt": 98,
                    "native_tokens_completion": 24,
                    "native_tokens_reasoning": 7,
                    "native_tokens_cached": 3,
                    "total_cost": "0.001234567",
                    "upstream_inference_cost": "0.0011",
                    "latency": 1800,
                }
            },
        )

        summary = self.ledger.run_summary("benchmark-1")
        self.assertEqual(summary["prompt_tokens"], 98)
        self.assertEqual(summary["completion_tokens"], 24)
        self.assertEqual(summary["reasoning_tokens"], 7)
        self.assertEqual(summary["cached_tokens"], 3)
        self.assertEqual(summary["model_cost_usd"], "0.001234567")

        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT provider_generation_id, provider_request_id, cost_nanos, "
                "duration_ms, ttft_ms "
                "FROM model_calls WHERE call_id = 'openrouter-call-1'"
            ).fetchone()
        self.assertEqual(row, ("gen-1", "req-1", 1234567, 2000, 1800))

    def test_duplicate_provider_generation_cannot_be_billed_twice(self):
        values = {
            "run_id": "benchmark-1",
            "stage": "compare",
            "provider": "openrouter",
            "requested_model": "provider/model",
            "provider_generation_id": "gen-duplicate",
            "started_at": START,
            "completed_at": END,
            "cost_usd": "0.01",
            "cost_source": "provider_reported",
        }
        self.ledger.record_model_call(call_id="call-1", **values)
        self.ledger.record_model_call(call_id="call-1", **values)
        with self.assertRaises(LedgerConflict):
            self.ledger.record_model_call(call_id="call-2", **values)

        summary = self.ledger.run_summary("benchmark-1")
        self.assertEqual(summary["model_request_count"], 1)
        self.assertEqual(summary["model_cost_usd"], "0.01")

    def test_compute_cost_uses_full_instance_lifetime_and_provider_truth_wins(self):
        self.ledger.register_compute(
            run_id="benchmark-1",
            allocation_id="allocation-1",
            provider="vultr",
            instance_id="instance-1",
            plan_id="vcg-a40-24c-120g-48vram",
            region="ewr",
            gpu_type="NVIDIA A40",
            gpu_count=1,
            hourly_cost_usd="1.712",
            pricing_source="vultr-plan-api",
            registered_at=START,
        )
        self.ledger.record_compute_event(
            allocation_id="allocation-1",
            event_id="created",
            event_type="created",
            occurred_at="2026-10-17T01:00:00Z",
        )
        self.ledger.record_compute_event(
            allocation_id="allocation-1",
            event_id="destroyed",
            event_type="destroyed",
            occurred_at="2026-10-17T02:30:00Z",
        )
        self.ledger.estimate_destroyed_compute_cost("allocation-1")

        self.assertEqual(self.ledger.run_summary("benchmark-1")["compute_cost_usd"], "2.568")

        self.ledger.record_compute_cost(
            allocation_id="allocation-1",
            cost_id="provider-cost",
            amount_usd="2.60",
            source="provider_reported",
            period_start="2026-10-17T01:00:00Z",
            period_end="2026-10-17T02:30:00Z",
            recorded_at="2026-10-17T03:00:00Z",
        )
        self.assertEqual(self.ledger.run_summary("benchmark-1")["compute_cost_usd"], "2.6")

    def test_money_uses_decimal_not_binary_float_arithmetic(self):
        for call_id, amount in (("a", Decimal("0.1")), ("b", Decimal("0.2"))):
            self.ledger.record_model_call(
                call_id=call_id,
                run_id="benchmark-1",
                stage="compare",
                provider="test",
                requested_model="test/model",
                started_at=START,
                completed_at=END,
                cost_usd=amount,
                cost_source="provider_reported",
            )
        self.assertEqual(self.ledger.run_summary("benchmark-1")["total_cost_usd"], "0.3")
        self.assertEqual(self.ledger.run_summary("benchmark-1")["total_cost_nanos"], 300000000)

    def test_sensitive_payloads_are_rejected_from_metadata(self):
        with self.assertRaisesRegex(ValueError, "sensitive metadata"):
            self.ledger.record_run_event(
                run_id="benchmark-1",
                stage="compare",
                status="started",
                metadata={"messages": [{"content": "private receipt data"}]},
            )


if __name__ == "__main__":
    unittest.main()
