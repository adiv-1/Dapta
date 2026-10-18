"""End-to-end tests of dapta_bench against fake_vllm_server.py (no GPU, no vLLM).

Run: CORD_VALIDATION_PARQUET=... python -m unittest notebooks/test_dapta_bench.py
"""

import csv
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import dapta_bench as bench
import dapta_vlm_engine as engine

HERE = Path(__file__).resolve().parent
SKILL = HERE.parent / "research/adapter-assets/skills/receipt-to-expense"
PARQUET = os.environ.get("CORD_VALIDATION_PARQUET")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def docs(n):
    from datasets import Image, load_dataset

    if not PARQUET:
        raise unittest.SkipTest("set CORD_VALIDATION_PARQUET")
    rows = load_dataset("parquet", data_files=PARQUET, split="train").cast_column("image", Image())
    return bench.encode_docs(rows, range(n), engine.EngineConfig())


class UnitTests(unittest.TestCase):
    def test_server_commands_differ_only_in_caching_and_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "adapter_config.json").write_text(json.dumps({"r": 16}))
            cfg = bench.BenchConfig(adapter_dir=tmp)
            plain, cached, lora = (bench.server_command(cfg, s) for s in ("plain", "cached", "lora"))
        self.assertIn("--no-enable-prefix-caching", plain)
        self.assertIn("--enable-prefix-caching", cached)
        self.assertIn("--no-enable-prefix-caching", lora)
        self.assertEqual(lora[lora.index("--max-lora-rank") + 1], "16")
        strip = lambda c: [x for x in c if x not in ("--enable-prefix-caching", "--no-enable-prefix-caching")]
        self.assertEqual(strip(plain), strip(cached))
        self.assertEqual(strip(lora)[: len(strip(plain))], strip(plain))
        merged = bench.server_command(bench.BenchConfig(lora_mode="merged"), "lora")
        self.assertEqual(merged[merged.index("--model") + 1], bench.BenchConfig().merged_dir)
        self.assertNotIn("--enable-lora", merged)

    def test_skill_prompt_is_a_shared_prefix(self):
        a, b = {"data_url": "data:a"}, {"data_url": "data:b"}
        first, second = (bench.messages_for("skill_cached", d, "SKILL") for d in (a, b))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[0]["role"], "system")
        self.assertEqual(bench.messages_for("lora", a, "SKILL")[0]["content"][1]["text"], engine.PROMPT)

    def test_percentiles_and_exact_match(self):
        self.assertEqual(bench._pct([1, 2, 3, 4], 0.5), 2.5)
        self.assertIsNone(bench._pct([], 0.5))
        perfect = {"schema_valid": True, "items_tp": 2, "items_gold": 2, "items_pred": 2,
                   "scalars": {"total": {"gold": True, "pred": True, "correct": True},
                               "tax": {"gold": False, "pred": False, "correct": False}}}
        self.assertTrue(bench.expense_exact(perfect))
        extra = {**perfect, "scalars": {**perfect["scalars"], "tax": {"gold": False, "pred": True, "correct": False}}}
        self.assertFalse(bench.expense_exact(extra))
        self.assertFalse(bench.expense_exact({**perfect, "items_pred": 3}))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs, cls.warm = docs(6), docs(3)
        cls.spec = json.loads((SKILL / "skill_spec.json").read_text())
        cls.schema = json.loads((SKILL / "output_schema.json").read_text())
        cls.skill = engine.full_skill_prompt(cls.spec, cls.schema)

    def test_missing_tuning_logs_are_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_bench(tmp, quick=True, training={"train_seconds": 1500})
            cost = report["one_time_lora_cost"]
            self.assertIsNone(cost["usd_including_tuning"])
            self.assertIsNone(cost["break_even_docs_including_tuning"])
            self.assertIn("not available", (Path(tmp) / "quick" / "benchmark_report.md").read_text())

    def run_bench(self, tmp, quick, training=None):
        adapter = Path(tmp) / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(json.dumps({"r": 16}))
        cfg = bench.BenchConfig(results_dir=tmp, adapter_dir=str(adapter), port=free_port(), quick_docs=5,
                                bootstrap_samples=200, concurrencies=(1, 4, 8), server_start_timeout_s=60,
                                server_command_override=(sys.executable, str(HERE / "fake_vllm_server.py")))
        return bench.run(cfg, self.docs, self.warm, self.skill, self.schema, quick=quick,
                         provenance={"test_sha256": "abc"},
                         training=training or {"train_seconds": 1500, "train_seconds_including_tuning": 3000})

    def test_quick_run_writes_every_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_bench(tmp, quick=True)
            out = Path(tmp) / "quick"
            for name in ("benchmark_runs.csv", "benchmark_summary.csv", "benchmark_report.md", "DECISIONS.md",
                         "accuracy_by_condition.png", "prompt_tokens_by_condition.png", "ttft_and_latency.png",
                         "throughput_vs_concurrency.png", "server_plain.log", "server_cached.log", "server_lora.log"):
                self.assertTrue((out / name).is_file(), name)
            with (out / "benchmark_runs.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4 * 3 * 5, "conditions x concurrencies x docs, one repeat")
            self.assertEqual(report["documents"], 5)
            self.assertEqual(report["skill_outputs_identical"]["identical_in_sample"], 5)
            md = (out / "benchmark_report.md").read_text()
            self.assertTrue(md.splitlines()[2].startswith("| Condition | Main accuracy"), "headline table comes first")
            for c in report["conditions"].values():
                self.assertLessEqual(c["ttft_s"]["p50"], c["latency_s"]["p50"])
                self.assertIsNotNone(c["throughput"][8]["kv_cache_usage_peak"])
            # The fake skill answer is schema-valid and right on total; the fake CORD answer converts the same way.
            self.assertEqual(report["conditions"]["skill_cached"]["quality"]["schema_valid"], 1.0)
            self.assertGreater(report["conditions"]["skill_cached"]["prompt_tokens_mean"],
                               report["conditions"]["lora"]["prompt_tokens_mean"] + 900)
            self.assertIsNotNone(report["conditions"]["skill_cached"]["throughput"][1]["prefix_cache_hit_rate"])
            self.assertIn("lora_mode", (out / "DECISIONS.md").read_text().replace("LoRA serving.", "lora_mode"))
            estimate = bench.estimate_full_minutes(report, bench.BenchConfig(concurrencies=(1, 4, 8)), documents=100)
            self.assertGreater(estimate["minutes"], 0)
            self.assertAlmostEqual(estimate["usd"], estimate["minutes"] / 60 * 1.5)

    def test_full_run_repeats_and_reports_spread(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_bench(tmp, quick=False)
            self.assertEqual(report["repeats"], 3)
            for c in report["conditions"].values():
                self.assertEqual(sorted(c["speed_by_repeat"]), [1, 2, 3])
                low, high = c["throughput"][4]["docs_per_s_range"]
                self.assertLessEqual(low, high)
            be = report["one_time_lora_cost"]
            self.assertAlmostEqual(be["usd"], 1500 / 3600 * 1.5)
            self.assertIn("lora_vs_skill_cached", report)
            self.assertIn("main_f1", report["lora_vs_skill_cached"])


if __name__ == "__main__":
    unittest.main()
