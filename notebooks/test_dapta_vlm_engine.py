"""CPU tests for dapta_vlm_engine: data, masking audit, gradients, scorer and selection.

Unsloth needs a GPU, so its collator and loader are exercised by the notebook's GPU test
cell instead. Here a reference collator built from the real Qwen3-VL processor stands in,
so the audit is checked against real tokenization, including planted masking mistakes.
Run: python -m unittest notebooks/test_dapta_vlm_engine.py  (pinned stack: transformers 4.57.6)
"""

import json
import os
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import dapta_vlm_engine as engine

CORD_PARQUET = os.environ.get("CORD_VALIDATION_PARQUET")
PROCESSOR_ID = "Qwen/Qwen3-VL-8B-Instruct"
TINY_MODEL = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"


def receipt_rows():
    from datasets import load_dataset

    if not CORD_PARQUET or not Path(CORD_PARQUET).is_file():
        raise unittest.SkipTest("set CORD_VALIDATION_PARQUET to the pinned CORD validation parquet")
    from datasets import Image

    # A bare parquet file has no HF feature types; the Hub dataset types this column as Image.
    rows = load_dataset("parquet", data_files=CORD_PARQUET, split="train")
    return rows.cast_column("image", Image())


def reference_batch(processor, examples):
    """Collate like Unsloth with train_on_responses_only: supervise only the assistant turn."""
    import torch

    texts = [processor.apply_chat_template(e["messages"], tokenize=False) for e in examples]
    images = [[e["messages"][0]["content"][0]["image"]] for e in examples]
    batch = processor(text=texts, images=images, padding=True, return_tensors="pt", add_special_tokens=False)
    labels = batch["input_ids"].clone()
    response = processor.tokenizer.encode(engine.RESPONSE_PART, add_special_tokens=False)
    for row in range(labels.shape[0]):
        ids = batch["input_ids"][row].tolist()
        start = next(i for i in range(len(ids) - len(response), -1, -1) if ids[i:i + len(response)] == response)
        labels[row, : start + len(response)] = -100
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch


class ConfigTests(unittest.TestCase):
    def test_defaults_validate_and_prompt_is_frozen(self):
        engine.EngineConfig().validate()
        with self.assertRaisesRegex(ValueError, "short prompt"):
            engine.EngineConfig(prompt="Describe the receipt in detail.").validate()

    def test_train_spec_rules(self):
        spec = engine.TrainSpec("smoke", rank=8, lr=1e-4, batch=1, accum=2, steps=5).validate()
        self.assertEqual(spec.effective_batch, 2)
        self.assertEqual(spec.warmup_steps(12), 1)
        full = engine.TrainSpec("full", rank=16, lr=1e-4, batch=2, accum=4, epochs=3, eval_every=50)
        self.assertEqual(full.total_steps(800), 300)
        self.assertEqual(full.warmup_steps(800), 15)
        for bad in (dict(steps=5, epochs=1), dict(), dict(steps=5, rank=7), dict(steps=5, lr=1.0),
                    dict(steps=5, early_stop_patience=2)):
            options = {"name": "x", "rank": 8, "lr": 1e-4, "batch": 1, "accum": 1, **bad}
            with self.assertRaises(ValueError, msg=bad):
                engine.TrainSpec(**options).validate()


class DataTests(unittest.TestCase):
    def test_target_round_trips_and_rejects_non_objects(self):
        gold = {"menu": [{"nm": "Kopi Susu", "price": "18,000"}], "total": {"total_price": "18,000"}}
        text = engine.target_text(gold)
        self.assertEqual(json.loads(text), gold)
        self.assertNotIn(" ", text.replace("Kopi Susu", ""))
        with self.assertRaises(TypeError):
            engine.gold_parse({"ground_truth": json.dumps({"gt_parse": ["not", "an", "object"]})})

    def test_prepare_image_caps_long_side_and_requires_pil(self):
        from PIL import Image

        image = engine.prepare_image(Image.new("RGBA", (900, 3000)), 1280)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(max(image.size), 1280)
        self.assertEqual(engine.prepare_image(Image.new("L", (300, 200)), 1280).size, (300, 200))
        with self.assertRaisesRegex(TypeError, "expected a PIL image"):
            engine.prepare_image([[0, 0]], 1280)

    def test_untyped_image_column_is_rejected(self):
        from datasets import load_dataset

        if not CORD_PARQUET or not Path(CORD_PARQUET).is_file():
            raise unittest.SkipTest("set CORD_VALIDATION_PARQUET")
        raw = load_dataset("parquet", data_files=CORD_PARQUET, split="train")
        with self.assertRaisesRegex(TypeError, "datasets.Image feature"):
            engine.receipt_dataset(raw, [0], engine.EngineConfig())

    def test_real_cord_rows_become_lazy_conversations(self):
        from datasets import Dataset
        from PIL import Image

        rows = receipt_rows()
        self.assertEqual(len(rows), 100)
        self.assertEqual(engine.check_rows(rows, range(len(rows))), {"rows_checked": 100})
        cfg = engine.EngineConfig()
        dataset = engine.receipt_dataset(rows, [3, 0, 7], cfg)
        self.assertIsInstance(dataset, Dataset, "TRL requires datasets.Dataset (error C)")
        self.assertEqual(len(dataset), 3)
        example = dataset[0]
        user, assistant = example["messages"]
        image = user["content"][0]["image"]
        self.assertIsInstance(image, Image.Image)
        self.assertLessEqual(max(image.size), cfg.image_long_side)
        self.assertEqual(user["content"][1]["text"], engine.PROMPT)
        self.assertEqual(json.loads(assistant["content"][0]["text"]), engine.gold_parse(rows[3]))


class MaskingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoProcessor

        cls.rows = receipt_rows()
        cls.processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
        cls.dataset = engine.receipt_dataset(cls.rows, [0, 1], engine.EngineConfig())
        cls.examples = [cls.dataset[0], cls.dataset[1]]
        cls.answers = [e["messages"][1]["content"][0]["text"] for e in cls.examples]

    def test_correct_masking_passes_on_a_padded_real_batch(self):
        report = engine.audit_batch(reference_batch(self.processor, self.examples), self.processor.tokenizer, self.answers)
        self.assertEqual(report["rows"], 2)
        self.assertGreater(report["visual_tokens"], 100)
        self.assertGreater(report["ignored_tokens"], report["visual_tokens"])

    def test_planted_masking_mistakes_are_caught(self):
        import torch

        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

        batch = reference_batch(self.processor, self.examples)
        position = (batch["input_ids"][0] == pad_id).nonzero()[0, 0]
        batch["labels"][0, position] = batch["input_ids"][0, position]
        with self.assertRaisesRegex(ValueError, "visual tokens are supervised"):
            engine.audit_batch(batch, tokenizer, self.answers)

        batch = reference_batch(self.processor, self.examples)
        first = (batch["labels"][0] != -100).nonzero()[0, 0]
        batch["labels"][0, first - 1] = batch["input_ids"][0, first - 1]
        with self.assertRaisesRegex(ValueError, "other than the answer"):
            engine.audit_batch(batch, tokenizer, self.answers)

        batch = reference_batch(self.processor, self.examples)
        batch["labels"][1] = -100
        with self.assertRaisesRegex(ValueError, "no supervised tokens"):
            engine.audit_batch(batch, tokenizer, self.answers)

        batch = reference_batch(self.processor, self.examples)
        end_id = tokenizer.convert_tokens_to_ids(engine.END_OF_TURN)
        supervised = (batch["labels"][0] != -100).nonzero().flatten()
        end = next(int(i) for i in supervised if int(batch["input_ids"][0, i]) == end_id)
        batch["labels"][0, end:] = -100  # end-of-turn no longer learned: generation would never stop
        with self.assertRaisesRegex(ValueError, "supervised tail"):
            engine.audit_batch(batch, tokenizer, self.answers)

        batch = reference_batch(self.processor, self.examples)
        last = (batch["labels"][0] != -100).nonzero()[-1, 0]
        batch["labels"][0, last] = -100  # dropping only the newline after <|im_end|> is harmless
        engine.audit_batch(batch, tokenizer, self.answers)

        batch = reference_batch(self.processor, self.examples)
        kept = (batch["labels"][0] != -100).nonzero().flatten()
        batch["labels"][0, kept[len(kept) // 2]] = -100
        with self.assertRaises(ValueError):
            engine.audit_batch(batch, tokenizer, self.answers)

        padded_row = int(batch["attention_mask"].sum(dim=1).argmin())
        batch = reference_batch(self.processor, self.examples)
        hole = (batch["attention_mask"][padded_row] == 0).nonzero()
        if len(hole):
            batch["labels"][padded_row, hole[0, 0]] = 0
            with self.assertRaisesRegex(ValueError, "padding tokens are supervised"):
                engine.audit_batch(batch, tokenizer, self.answers)

    def test_real_sequence_lengths_fit_the_limit(self):
        cfg = engine.EngineConfig()
        dataset = engine.receipt_dataset(self.rows, range(len(self.rows)), cfg)
        report = engine.length_report(engine.sequence_lengths(self.processor, dataset), cfg.max_seq_len)
        print("\nCORD validation token lengths at 1280px:", report)
        self.assertEqual(report["examples"], 100)
        self.assertLess(report["max"], cfg.max_seq_len)
        with self.assertRaisesRegex(ValueError, "exceed max_seq_len"):
            engine.length_report([100, 5000], 4096)


class GradientTests(unittest.TestCase):
    def test_lora_moves_and_frozen_weights_stay_put_on_a_tiny_qwen3_vl(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        torch.manual_seed(0)
        processor = AutoProcessor.from_pretrained(TINY_MODEL)
        model = Qwen3VLForConditionalGeneration.from_pretrained(TINY_MODEL, dtype=torch.float32)
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
        params = engine.trainable_report(model)
        self.assertGreater(params["trainable_tensors"], 0)

        rows = receipt_rows()
        dataset = engine.receipt_dataset(rows, [0], engine.EngineConfig(image_long_side=256))

        class Collator:
            def __call__(self, examples):
                return reference_batch(processor, examples)

        report = engine.gradient_check(model, Collator(), dataset)
        self.assertGreater(report["lora_tensors_with_grad"], 0)
        self.assertEqual(report["sampled_frozen_unchanged"], 16)

        state = engine.lora_state(model)
        self.assertTrue(state and all("lora_" in key and not key.startswith("base_model") for key in state))


class ScorerTests(unittest.TestCase):
    def test_baseline_scorer_behaviour_is_unchanged(self):
        self.assertTrue(engine.score_example({"menu": {"nm": "Coffee"}}, {"menu": [{"nm": "Coffee"}]})["document_exact"])
        self.assertEqual(engine.score_example({"total": {"total_price": 16500}}, {"total": {"total_price": "16,500"}})["tp"], 0)
        self.assertEqual(engine.parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(engine.parse_json_object('text {"a": {"b": 2}} tail'), {"a": {"b": 2}})
        self.assertIsNone(engine.parse_json_object("no json here"))
        gold = {"menu": [{"nm": "A", "price": "1"}, {"nm": "B", "price": "2"}]}
        swapped = {"menu": [{"nm": "B", "price": "2"}, {"nm": "A", "price": "1"}]}
        self.assertEqual(engine.score_example(swapped, gold)["tp"], 0, "menu positions matter")

    def test_summary_matches_hand_computed_f1(self):
        rows = [
            {"score": {"tp": 3, "pred_fields": 4, "gold_fields": 5, "document_exact": False, "total_present": True,
                       "total_exact": True}, "valid_json": True, "elapsed_s": 1.0, "input_tokens": 10, "output_tokens": 5},
            {"score": {"tp": 1, "pred_fields": 2, "gold_fields": 1, "document_exact": False, "total_present": False,
                       "total_exact": False}, "valid_json": False, "elapsed_s": 3.0, "input_tokens": 20, "output_tokens": 7},
        ]
        summary = engine.summarize(rows)
        p, r = 4 / 6, 4 / 6
        self.assertAlmostEqual(summary["field_f1"], round(2 * p * r / (p + r), 6))
        self.assertEqual(summary["total_price_exact_fraction"], 1.0)
        self.assertEqual(summary["valid_json_fraction"], 0.5)


SKILL_DIR = Path(__file__).resolve().parents[1] / "research/adapter-assets/skills/receipt-to-expense"


class ExpenseSkillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from jsonschema import Draft202012Validator

        cls.spec = json.loads((SKILL_DIR / "skill_spec.json").read_text())
        cls.schema = json.loads((SKILL_DIR / "output_schema.json").read_text())
        cls.validator = Draft202012Validator(cls.schema)

    def test_amounts_follow_indonesian_receipt_conventions(self):
        cases = {"18,000": 18000, "18.000": 18000, "1.250.000": 1250000, "Rp 23.000": 23000, "12.50": 12.5,
                 "1,234.56": 1234.56, "1.234,56": 1234.56, "-5,000": -5000, "2": 2, 3: 3, None: None,
                 "abc": None, "1 x": 1}
        for raw, expected in cases.items():
            self.assertEqual(engine.parse_amount(raw), expected, raw)
        self.assertEqual(engine.parse_amount(["7.500", None]), 7500)

    def test_whole_skill_reaches_the_prompt_unchanged(self):
        prompt = engine.full_skill_prompt(self.spec, self.schema)
        self.assertIn(json.dumps(self.spec, indent=2, ensure_ascii=False), prompt)
        self.assertIn(json.dumps(self.schema, indent=2, ensure_ascii=False), prompt)

    def test_converted_gold_is_schema_valid_and_scores_perfectly(self):
        rows = receipt_rows()
        scores = []
        for i in range(len(rows)):
            gold = engine.cord_to_expense(engine.gold_parse(rows[i]))
            answer = {"receipts": [gold]}
            self.assertEqual(list(self.validator.iter_errors(answer)), [], f"row {i}")
            scores.append({"expense": engine.score_expense(answer, gold, self.validator), "valid_json": True})
        summary = engine.summarize_expense(scores)
        self.assertEqual((summary["field_f1"], summary["schema_valid_fraction"]), (1.0, 1.0))

    def test_errors_are_penalised(self):
        gold = {"merchant": None, "address": None, "transaction_date": None, "currency": None, "subtotal": 100.0,
                "tax": 10.0, "service_charge": None, "total": 110.0,
                "line_items": [{"description": "Kopi Susu", "quantity": 1.0, "amount": 100.0}]}
        good = {"receipts": [dict(gold, merchant="Toko A", line_items=[
            {"description": " kopi  susu ", "quantity": 1, "amount": 100}])]}
        score = engine.score_expense(good, gold, self.validator)
        self.assertEqual((score["schema_valid"], score["items_tp"]), (True, 1))
        self.assertTrue(all(v["correct"] for k, v in score["scalars"].items() if k != "service_charge"))

        wrong = {"receipts": [dict(gold, total=11.0, service_charge=5.0,
                                   line_items=[{"description": "Kopi Susu", "quantity": 1, "amount": 10}])]}
        score = engine.score_expense(wrong, gold, self.validator)
        self.assertEqual(score["items_tp"], 0)
        self.assertFalse(score["scalars"]["total"]["correct"])
        self.assertTrue(score["scalars"]["service_charge"]["pred"] and not score["scalars"]["service_charge"]["gold"])

        invalid = {"receipts": [dict(gold, category="meals")]}
        score = engine.score_expense(invalid, gold, self.validator)
        self.assertFalse(score["schema_valid"])
        self.assertEqual((score["items_tp"], score["items_gold"]), (0, 1), "gold still counts against recall")
        summary = engine.summarize_expense([{"expense": score, "valid_json": True}])
        self.assertEqual(summary["field_recall"], 0.0)


class PersistTests(unittest.TestCase):
    def test_without_a_token_nothing_is_uploaded(self):
        self.assertIsNone(engine.persist_to_hub(Path("/tmp"), "o/r", None, "run", ["*.json"], "stage"))


class SelectionTests(unittest.TestCase):
    def test_batch_choice_respects_headroom_and_skips_oom(self):
        rows = [
            {"microbatch": 1, "grad_accum": 8, "stable": True, "peak_reserved_gib": 20.0, "examples_per_second": 1.0},
            {"microbatch": 2, "grad_accum": 4, "stable": True, "peak_reserved_gib": 30.0, "examples_per_second": 1.6},
            {"microbatch": 4, "grad_accum": 2, "stable": True, "peak_reserved_gib": 36.5, "examples_per_second": 2.0},
            {"microbatch": 8, "grad_accum": 1, "oom": True},
        ]
        choice = engine.choose_batch(rows, capacity_gib=39.5)
        self.assertEqual(choice["microbatch"], 2, "batch 4 is faster but leaves under 15% headroom")
        with self.assertRaises(ValueError):
            engine.choose_batch([{"oom": True}], capacity_gib=39.5)

    def test_trial_choice_flags_inseparable_results(self):
        def records(tp, seed):
            rng = random.Random(seed)
            return [{"index": i, "score": {"tp": tp + rng.randint(-1, 1), "pred_fields": 10, "gold_fields": 10}}
                    for i in range(20)]

        results = [{"name": "a", "val_field_f1": 0.61, "val_loss": 0.5},
                   {"name": "b", "val_field_f1": 0.60, "val_loss": 0.4}]
        close = engine.choose_trial(results, {"a": records(6, 1), "b": records(6, 2)})
        self.assertEqual(close["best"]["name"], "a")
        self.assertFalse(close["conclusive"])
        self.assertIn("not separable", close["recommendation"])
        clear = engine.choose_trial(results, {"a": records(9, 1), "b": records(3, 2)})
        self.assertTrue(clear["conclusive"])


class ProgressTests(unittest.TestCase):
    def test_progress_line_has_every_field(self):
        import torch

        if not torch.cuda.is_available():
            engine.gpu_memory = lambda: {"allocated_gib": 24.1, "total_gib": 39.5, "peak_reserved_gib": 25.0}
        progress = engine.LiveProgress("full", Path("/tmp/history.jsonl"))
        progress.started = __import__("time").perf_counter() - 1110
        progress.last_eval_loss = 0.91
        state = SimpleNamespace(global_step=145, max_steps=300, epoch=1.45, num_train_epochs=3)
        line = progress.format_line(state, {"loss": 0.82, "learning_rate": 1e-4})
        for field in ("Training: 145/300 steps", "Progress: 48.3%", "Epoch: 1.45/3", "Loss: 0.8200",
                      "Validation loss: 0.9100", "Learning rate: 1.0e-04", "GPU memory: 24.1/39.5 GB",
                      "Elapsed: 18m 30s", "ETA:"):
            self.assertIn(field, line)


if __name__ == "__main__":
    unittest.main()
