import unittest

from dapta.factory_runtime import (
    AgentDatasetSource,
    SavedEvaluation,
    training_allocation_id,
    training_instance_id,
)


class FactoryRuntimeTests(unittest.TestCase):
    def test_training_retries_share_a_stable_job_round_allocation(self):
        first = training_allocation_id("job-1", 1)
        second = training_allocation_id("job-1", 1)

        self.assertEqual(first, second)
        self.assertEqual(first, "runpod:job-1:round-1:lease-v2")
        self.assertEqual(
            training_instance_id("pod-123", "job-1", 1),
            "pod-123:job-job-1:round-1",
        )

    def test_validator_rejects_input_leakage_and_schema_violations(self):
        source = AgentDatasetSource(agents=None)
        dataset = {
            "train": [
                {"id": f"train-{index}", "input": f"training document {index}", "output": {"value": index}}
                for index in range(16)
            ],
            "dev": [
                {"id": f"dev-{index}", "input": f"validation document {index}", "output": {"value": index}}
                for index in range(4)
            ],
            "test": [
                {"id": f"test-{index}", "input": f"test document {index}", "output": {"value": index}}
                for index in range(4)
            ],
            "ood": [
                {"id": "ood-0", "input": "training document 0", "output": {"value": 0}},
                {"id": "ood-1", "input": "out of distribution document", "output": {"wrong": True}},
            ],
        }
        result = source.validate(
            job={},
            round=1,
            dataset=dataset,
            integration={},
            skill_contract={
                "output_schema": {
                    "type": "object",
                    "required": ["value"],
                    "properties": {"value": {"type": "integer"}},
                }
            },
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("input leakage" in error for error in result["errors"]))
        self.assertTrue(any("violates schema" in error for error in result["errors"]))

    def test_saved_evaluation_requires_lift_and_schema_validity(self):
        evaluator = SavedEvaluation()
        result = evaluator.evaluate(
            job={},
            round=1,
            training={
                "evaluation": {
                    "conditions": {
                        "base_with_skill": {"field_f1": 0.60},
                        "lora_short_prompt": {
                            "field_f1": 0.70,
                            "schema_pass_rate": 0.90,
                        },
                    }
                }
            },
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["thresholds"]["minimum_lift"], 0.02)


if __name__ == "__main__":
    unittest.main()
