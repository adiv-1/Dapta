import json
import unittest

from dapta.jobs import JobStore
from dapta.ledger import UsageLedger
from dapta.lora import (
    DevinReviewer,
    KimiDataBuilder,
    KimiSkillWriter,
    KIMI_K3,
    LoraAgent,
    QwenTrainingCoder,
    TaskReader,
    devin_review_prompt,
)
from dapta.vultr import CONFIRMATION


TASK = "Turn receipt images into a strict expense-report JSON object."


def skill_document(task=TASK, skill_id="receipt-to-expense"):
    return KimiSkillWriter().compose(
        understanding=TaskReader().understand(task=task),
        skill_id=skill_id,
    )


class Recording:
    def __init__(self):
        self.order = []

    def understand(self, *, task):
        self.order.append("understand")
        return TaskReader().understand(task=task)

    def compose(self, *, understanding, skill_id):
        self.order.append("skill")
        self.understanding = understanding
        return skill_document(understanding["task"], skill_id)

    def generate(self, *, skill, run_id):
        self.order.append("data")
        self.skill = skill
        return KimiDataBuilder().generate(skill=skill, run_id=run_id)

    def claim(self, *, confirm_spend):
        self.order.append("gpu")
        self.confirm_spend = confirm_spend
        return {"name": "NVIDIA A40", "vram_gb": 48, "warm": True, "will_provision": False}

    def review(self, *, understanding, skill, training_code, data, gpu):
        self.order.append("devin")
        self.code = training_code
        self.reviewed_skill = skill
        return {"passed": True, "findings": []}

    def train(self, *, task, skill_id, training_code, data, gpu, run_id):
        self.order.append("train")
        return {"version": "v1", "base_model": "Qwen/Qwen3-VL-8B-Instruct", "loss": 0.4}

    def evaluate(self, *, training, data):
        self.order.append("eval")
        return {"passed": True, "condition": "qwen_lora", "held_out_cases": 1, "benchmark": "bounded"}

    def store(self, *, training, evaluation):
        self.order.append("store")
        return "hf://owner/adapters/receipt-to-expense/v1"


class LoraAgentTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.temp = tempfile.TemporaryDirectory()
        database = Path(self.temp.name) / "dapta.sqlite3"
        self.parts = Recording()
        self.agent = LoraAgent(
            JobStore(database),
            UsageLedger(database),
            task_agent=self.parts,
            skill_agent=self.parts,
            data=self.parts,
            gpu=self.parts,
            reviewer=self.parts,
            trainer=self.parts,
            evaluator=self.parts,
            publisher=self.parts,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_task_becomes_a_stored_adapter_after_devin_review(self):
        result = self.agent.build(
            task=TASK,
            skill_id="receipt-to-expense",
            confirm_spend=CONFIRMATION,
        )

        self.assertEqual(result["status"], "stored")
        self.assertEqual(result["artifact_uri"], "hf://owner/adapters/receipt-to-expense/v1")
        self.assertEqual(
            self.parts.order,
            ["understand", "skill", "data", "gpu", "devin", "train", "eval", "store"],
        )
        self.assertEqual(
            result["stages"],
            [
                "understood",
                "skill_written",
                "data_generated",
                "gpu_warm",
                "code_written",
                "reviewed",
                "trained",
                "evaluated",
                "stored",
            ],
        )
        self.assertEqual(self.parts.skill["model"], KIMI_K3)
        self.assertIn("rules", result["skill"])
        self.assertTrue(result["understanding"]["refusals"])
        self.assertTrue(result["gpu"]["warm"])
        self.assertFalse(result["gpu"]["will_provision"])
        self.assertIn("CUDA_VISIBLE_DEVICES", self.parts.code)
        self.assertIn("load_in_4bit = True", self.parts.code)
        self.assertIn("gradient_checkpointing = True", self.parts.code)
        self.assertNotIn('device_map="auto"', self.parts.code)
        self.assertEqual(result["evaluation"]["condition"], "qwen_lora")
        self.assertLessEqual(result["evaluation"]["held_out_cases"], 32)
        self.assertEqual(result["adapter"]["artifact_uri"], result["artifact_uri"])
        self.assertEqual(self.parts.confirm_spend, CONFIRMATION)
        saved = {event["stage"]: event for event in self.agent.trace.events}
        self.assertEqual(saved["prompted"]["prompt"], TASK)
        self.assertIn(TASK, saved["understood"]["prompt"])
        self.assertEqual(saved["understood"]["thought"]["refusals"][0], result["understanding"]["refusals"][0])
        self.assertEqual(saved["skill_written"]["thought"]["rules"], result["skill"]["rules"])
        self.assertIn("frozen test", saved["data_generated"]["prompt"])
        self.assertEqual(saved["data_generated"]["thought"]["test"][0]["id"], "test-1")
        self.assertEqual(saved["code_written"]["thought"], result["training_code"])
        self.assertIn("skill document", saved["reviewed"]["prompt"].lower())
        self.assertEqual(saved["reviewed"]["thought"]["passed"], True)
        self.assertIn(result["training_code"], saved["trained"]["prompt"])
        self.assertEqual(saved["stored"]["thought"]["artifact_uri"], result["artifact_uri"])
        self.assertEqual(self.agent.trace.finished[-1]["status"], "stored")
        self.assertEqual(result["stored_events"], len(self.agent.trace.events))

    def test_devin_rejection_stops_before_training(self):
        def reject(**kwargs):
            self.parts.order.append("devin")
            return {"passed": False, "findings": ["test overlaps train"]}

        self.parts.review = reject

        result = self.agent.build(task=TASK, skill_id="receipt-to-expense", confirm_spend=CONFIRMATION)

        self.assertEqual(result["status"], "review_failed")
        self.assertEqual(self.parts.order, ["understand", "skill", "data", "gpu", "devin"])
        self.assertIn("skill_written", result["stages"])
        self.assertNotIn("trained", result["stages"])

    def test_spend_confirmation_is_required_after_review(self):
        result = self.agent.build(task=TASK, skill_id="receipt-to-expense")

        self.assertEqual(result["status"], "awaiting_spend")
        self.assertNotIn("train", self.parts.order)
        self.assertTrue(result["gpu"]["warm"])
        self.assertIsNone(self.parts.confirm_spend)

    def test_default_writers_keep_test_disjoint_from_train(self):
        data = KimiDataBuilder().generate(skill=skill_document(), run_id="run-1")
        test_ids = {row["id"] for row in data["test"]}
        earlier = {row["id"] for row in data["train"] + data["dev"]}
        self.assertTrue(test_ids)
        self.assertFalse(test_ids & earlier)
        self.assertEqual(data["train"][0]["model"], KIMI_K3)

    def test_cpu_training_does_not_wait_for_spend(self):
        self.parts.claim = lambda **kwargs: {
            "name": "CPU",
            "device": "cpu",
            "warm": True,
            "will_provision": False,
        }

        result = self.agent.build(task=TASK, skill_id="receipt-to-expense")

        self.assertEqual(result["status"], "stored")
        self.assertIn("cpu_ready", result["stages"])
        self.assertNotIn("gpu_warm", result["stages"])
        self.assertIn("gpu=False", result["training_code"])
        self.assertIn('device_map = {"": "cpu"}', result["training_code"])
        self.assertNotIn("load_in_4bit = True", result["training_code"])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", result["training_code"])

    def test_training_loop_shrinks_for_a_smaller_gpu(self):
        skill = skill_document()
        large = QwenTrainingCoder().write(skill=skill, gpu={"name": "NVIDIA A40", "vram_gb": 48})
        small = QwenTrainingCoder().write(skill=skill, gpu={"name": "NVIDIA A16", "vram_gb": 16})
        self.assertGreater(_pixels(large), _pixels(small))

    def test_devin_prompt_covers_skill_data_code_and_gpu(self):
        understanding = TaskReader().understand(task=TASK)
        skill = skill_document()
        prompt = devin_review_prompt(
            understanding,
            skill,
            "code",
            {"train": [1], "dev": [], "test": [1], "ood": []},
            {"name": "NVIDIA A40", "vram_gb": 48, "warm": True},
        )
        self.assertIn("skill document", prompt.lower())
        self.assertIn(KIMI_K3, prompt)
        self.assertIn("test does not overlap", prompt)
        self.assertIn("device_map", prompt)
        self.assertIn("warm", prompt)
        reply = json.dumps({"passed": True, "findings": []})
        review = DevinReviewer(type("Client", (), {"ask": lambda self, prompt: reply})()).review(
            understanding=understanding,
            skill=skill,
            training_code="code",
            data={"train": [], "dev": [], "test": [], "ood": []},
            gpu={"name": "NVIDIA A40", "vram_gb": 48, "warm": True},
        )
        self.assertTrue(review["passed"])
        self.assertEqual(review["reviewer"], "devin")

    def test_gpu_claim_cannot_provision(self):
        self.parts.claim = lambda **kwargs: {
            "name": "NVIDIA A40",
            "vram_gb": 48,
            "warm": True,
            "will_provision": True,
        }
        with self.assertRaisesRegex(ValueError, "must not provision"):
            self.agent.build(task=TASK, skill_id="receipt-to-expense", confirm_spend=CONFIRMATION)


def _pixels(code: str) -> int:
    for line in code.splitlines():
        if line.strip().startswith("max_pixels="):
            return int(line.split("=", 1)[1].rstrip(","))
    raise AssertionError("max_pixels missing")


if __name__ == "__main__":
    unittest.main()
