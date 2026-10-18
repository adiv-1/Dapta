import json
import tempfile
import unittest
from pathlib import Path

from dapta.factory_limits import (
    SMOKE_MAX_EXAMPLES,
    SMOKE_MAX_TRAIN_SECONDS,
    SMOKE_MINIMUMS,
    apply_smoke_training_limits,
    smoke_training_limits,
    training_access_open,
)
from dapta.factory_runtime import AgentDatasetSource


def _cases(split: str, count: int) -> list[dict]:
    return [
        {"id": f"{split}-{index}", "input": f"{split} document {index} text", "output": {"value": index}}
        for index in range(count)
    ]


class FactoryLimitTests(unittest.TestCase):
    def test_smoke_profile_is_ten_examples_and_two_minutes(self):
        limits = smoke_training_limits()

        self.assertEqual(sum(SMOKE_MINIMUMS.values()), SMOKE_MAX_EXAMPLES)
        self.assertEqual(limits["max_examples"], 10)
        self.assertEqual(limits["max_train_seconds"], SMOKE_MAX_TRAIN_SECONDS)
        self.assertEqual(limits["max_rounds"], 1)
        bundle = {
            "epochs": 2,
            "max_train_examples": 64,
            "max_test_examples": 12,
            "max_training_seconds": 0,
        }
        apply_smoke_training_limits(bundle, {"profile": "smoke", "max_train_seconds": 99999})
        self.assertEqual(bundle["max_train_examples"], 4)
        self.assertEqual(bundle["max_test_examples"], 2)
        self.assertEqual(bundle["max_training_seconds"], 120)
        self.assertEqual(bundle["epochs"], 1.0)

    def test_missing_or_expired_access_file_is_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "access.json"
            self.assertFalse(training_access_open(path, now=1_000))
            path.write_text(json.dumps({"enabled": True, "expires_at": 1_000}), encoding="utf-8")
            self.assertFalse(training_access_open(path, now=1_000))
            path.write_text(json.dumps({"enabled": True, "expires_at": 1_001}), encoding="utf-8")
            self.assertTrue(training_access_open(path, now=1_000))

    def test_smoke_dataset_accepts_ten_examples(self):
        source = AgentDatasetSource(agents=None)
        dataset = {
            "train": _cases("train", 4),
            "dev": _cases("dev", 2),
            "test": _cases("test", 2),
            "ood": _cases("ood", 2),
        }
        schema = {
            "output_schema": {
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "integer"}},
            }
        }
        smoke = source.validate(
            job={"training_limits": smoke_training_limits()},
            round=1,
            dataset=dataset,
            integration={},
            skill_contract=schema,
        )
        full = source.validate(
            job={},
            round=1,
            dataset=dataset,
            integration={},
            skill_contract=schema,
        )

        self.assertTrue(smoke["passed"], smoke["errors"])
        self.assertEqual(sum(smoke["counts"].values()), 10)
        self.assertFalse(full["passed"])


if __name__ == "__main__":
    unittest.main()
