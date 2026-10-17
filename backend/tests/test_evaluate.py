import json
import tempfile
import unittest
from pathlib import Path

from dapta.data import freeze_test
from dapta.evaluate import evaluate_receipts


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/adapter-assets/skills/receipt-to-expense"


def receipt(*, merchant="Acme", total=12.5):
    return {
        "receipts": [{
            "merchant": merchant,
            "address": "1 Main St",
            "transaction_date": "2026-10-17",
            "currency": None,
            "subtotal": None,
            "tax": None,
            "service_charge": None,
            "total": total,
            "line_items": [],
        }]
    }


class ReceiptEvaluatorTests(unittest.TestCase):
    def make_dataset(self, root: Path):
        for split in ("train", "dev", "test"):
            image = root / f"{split}.png"
            image.write_bytes(split.encode())
            row = {
                "id": f"{split}-1",
                "images": [image.name],
                "prompt": "Extract the receipt.",
                "completion": receipt(),
                "metadata": {"source_id": "receipt-sroie"},
            }
            (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        freeze_test(root)

    def test_compares_accuracy_time_cost_and_tokens_per_condition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            predictions = root / "predictions.jsonl"
            rows = [
                {
                    "id": "test-1",
                    "condition": "qwen-lora",
                    "output": receipt(),
                    "usage": {
                        "latency_ms": 100,
                        "cost_usd": "0.001",
                        "input_tokens": 20,
                        "output_tokens": 30,
                    },
                },
                {
                    "id": "test-1",
                    "condition": "frontier",
                    "output": receipt(merchant="Wrong"),
                    "usage": {
                        "latency_ms": 50,
                        "cost_usd": "0.02",
                        "input_tokens": 40,
                        "output_tokens": 30,
                    },
                },
            ]
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            report = evaluate_receipts(root, predictions, contract_dir=CONTRACT)

            self.assertEqual(report["conditions"]["qwen-lora"]["accuracy"], 1.0)
            self.assertEqual(report["conditions"]["frontier"]["accuracy"], 0.75)
            self.assertEqual(report["conditions"]["qwen-lora"]["cost_usd"], "0.001")
            self.assertEqual(report["conditions"]["qwen-lora"]["tokens"]["total"], 50)
            self.assertEqual(report["performance_summary"]["most_accurate"], "qwen-lora")
            self.assertEqual(report["performance_summary"]["fastest"], "frontier")
            self.assertEqual(report["performance_summary"]["lowest_cost"], "qwen-lora")

    def test_schema_or_exclusion_failure_zeroes_the_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            output = receipt()
            output["category"] = "meals"
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"id": "test-1", "condition": "model", "output": output}) + "\n",
                encoding="utf-8",
            )

            report = evaluate_receipts(root, predictions, contract_dir=CONTRACT)

            case = report["cases"][0]
            self.assertFalse(case["gates"]["schema_valid"])
            self.assertFalse(case["gates"]["passed"])
            self.assertEqual(report["conditions"]["model"]["accuracy"], 0.0)

    def test_requires_one_prediction_per_case_and_condition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_dataset(root)
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                json.dumps({"id": "dev-1", "condition": "model", "output": receipt()}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown case"):
                evaluate_receipts(root, predictions, contract_dir=CONTRACT)


if __name__ == "__main__":
    unittest.main()
