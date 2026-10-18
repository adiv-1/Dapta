import unittest
from unittest.mock import patch

from dapta.factory_runtime import (
    AgentDatasetSource,
    DevinIntegrator,
    SavedEvaluation,
    training_allocation_id,
    training_instance_id,
)


class FactoryRuntimeTests(unittest.TestCase):
    def test_devin_wait_suspends_the_stage_and_resumes_for_the_result(self):
        class Stage:
            active = None

            def __init__(self):
                self.actions = []

            def suspend(self, reason):
                self.actions.append(("suspend", reason))

            def resume(self, reason):
                self.actions.append(("resume", reason))

        class Store:
            def get_artifact(self, *_):
                return {"session_id": "session-1"}

            def set_artifact(self, *_):
                pass

            def record_progress(self, *_):
                pass

        class Client:
            def get_session(self, _):
                from dapta.devin import DevinSession
                return DevinSession("session-1", None, "completed", acus_consumed=1.5)

        stage = Stage()
        integrator = DevinIntegrator(Client(), Store())
        with patch("dapta.factory_runtime.current_factory_stage", return_value=stage):
            result = integrator.integrate(job={"job_id": "job-1"}, skill_contract={})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(stage.actions, [("suspend", "devin_session"), ("resume", "devin_completed")])

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
        self.assertTrue(result["repairable"])
        self.assertEqual(result["thresholds"]["minimum_lift"], 0.02)

    def test_saved_evaluation_marks_truncation_as_configuration_failure(self):
        evaluator = SavedEvaluation()
        result = evaluator.evaluate(
            job={},
            round=1,
            training={
                "evaluation": {
                    "conditions": {
                        "base_with_skill": {"field_f1": 0.0},
                        "lora_short_prompt": {
                            "field_f1": 0.0,
                            "schema_pass_rate": 0.0,
                            "truncated_cases": ["test-1"],
                        },
                    }
                }
            },
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["repairable"])
        self.assertEqual(result["failure_reason"], "evaluation_output_truncated")
        self.assertIn("do not generate repair data", result["dev_feedback"]["instruction"])


if __name__ == "__main__":
    unittest.main()
