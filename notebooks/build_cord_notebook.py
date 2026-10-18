"""Regenerate dapta_cord_qlora.ipynb after cell 3, embedding dapta_vlm_engine.py verbatim.

Cells 0-5 (location check, pinned install, environment check) and their saved outputs are
kept as they are. Run: python notebooks/build_cord_notebook.py
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "dapta_cord_qlora.ipynb"
ENGINE = HERE / "dapta_vlm_engine.py"
KEEP = 6  # cells written before this builder existed


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


engine_source = ENGINE.read_text(encoding="utf-8")
engine_sha = hashlib.sha256(engine_source.encode()).hexdigest()[:12]

SKILL_DIR = HERE.parent / "research/adapter-assets/skills/receipt-to-expense"
skill_spec_text = (SKILL_DIR / "skill_spec.json").read_text(encoding="utf-8").strip()
output_schema_text = (SKILL_DIR / "output_schema.json").read_text(encoding="utf-8").strip()
review = json.loads((SKILL_DIR / "review.json").read_text(encoding="utf-8"))
assert "'''" not in skill_spec_text + output_schema_text

SKILL_BASELINE = f"""
import hashlib, jsonschema
RUN_SKILL_BASELINE = True

# The repo's receipt-to-expense skill, verbatim, checked against the reviewed contract's hashes.
SKILL_SPEC = json.loads(r'''{skill_spec_text}''')
OUTPUT_SCHEMA = json.loads(r'''{output_schema_text}''')
REVIEWED_SHA256 = {json.dumps(review["artifact_sha256"])}
for name, value in (("skill_spec.json", SKILL_SPEC), ("output_schema.json", OUTPUT_SCHEMA)):
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == REVIEWED_SHA256[name], f"{{name}} differs from the reviewed skill"
FULL_SKILL = engine.full_skill_prompt(SKILL_SPEC, OUTPUT_SCHEMA)

if not RUN_SKILL_BASELINE:
    print("Skipped: skill baseline.")
else:
    base_model, base_processor = engine.load_base(CFG)
    everything = range(len(val_raw))
    short = engine.evaluate_condition(base_model, base_processor, val_raw, everything, CFG, desc="base + short prompt",
                                      user_prompt=CFG.prompt, answer_format="cord", output_schema=OUTPUT_SCHEMA,
                                      output_dir=ROOT / "baseline_4bit_short_prompt")["summary"]
    skill = engine.evaluate_condition(base_model, base_processor, val_raw, everything, CFG, desc="base + full skill",
                                      system_prompt=FULL_SKILL, user_prompt=engine.SKILL_USER_PROMPT,
                                      answer_format="expense", output_schema=OUTPUT_SCHEMA,
                                      output_dir=ROOT / "baseline_4bit_full_skill")["summary"]
    del base_model, base_processor
    engine.free_gpu()
    same = short["expense_format"]
    BASELINES = {{"base_short_prompt": short, "base_full_skill": skill, "bf16_short_prompt_reference": engine.BASELINE_BF16}}
    engine.write_json(ROOT / "skill_baseline.json", BASELINES)

    def cell(value, digits=3):
        return "n/a" if value is None else (f"{{value:.{{digits}}f}}" if isinstance(value, float) else str(value))
    table = [
        ("Skill-format field F1", same["field_f1"], skill["field_f1"]),
        ("Skill-format field precision", same["field_precision"], skill["field_precision"]),
        ("Skill-format field recall", same["field_recall"], skill["field_recall"]),
        ("Line-item F1", same["line_item_f1"], skill["line_item_f1"]),
        ("Total accuracy", same["total_accuracy"], skill["total_accuracy"]),
        ("Subtotal accuracy", same["subtotal_accuracy"], skill["subtotal_accuracy"]),
        ("Tax accuracy", same["tax_accuracy"], skill["tax_accuracy"]),
        ("Valid JSON", short["valid_json_fraction"], skill["valid_json_fraction"]),
        ("Schema valid (skill gate)", None, skill["schema_valid_fraction"]),
        ("CORD field F1 (baseline scorer)", short["field_f1"], None),
        ("Mean input tokens", short["mean_input_tokens"], skill["mean_input_tokens"]),
        ("Mean output tokens", short["mean_output_tokens"], skill["mean_output_tokens"]),
        ("Seconds per receipt", short["mean_seconds_per_receipt"], skill["mean_seconds_per_receipt"]),
        ("Truncated outputs", short["truncated_outputs"], skill["truncated_outputs"]),
    ]
    print("BASELINE RESULTS (4-bit base, 100 validation receipts)")
    print(f"{{'Metric':34s}} | {{'base + short prompt':>20s}} | {{'base + full skill':>18s}}")
    for name, a, b in table:
        print(f"BASELINE ROW {{name:34s}} | {{cell(a):>20s}} | {{cell(b):>18s}}")
    print("Reference, BF16 + short prompt (earlier run, CORD scorer): field F1", engine.BASELINE_BF16["field_f1"])
    persist(["skill_baseline.json", "data_report.json", "baseline_4bit_*/**"], "skill baseline")
    print("SKILL BASELINE DONE")
"""

FINAL_COMPARISON = """
if not globals().get("FULL_DONE"):
    print("Skipped: needs a finished full training run.")
else:
    del full_trainer
    engine.free_gpu()
    everything = range(len(val_raw))
    lora = engine.evaluate_condition(full_model, full_processor, val_raw, everything, CFG, desc="LoRA + short prompt",
                                     user_prompt=CFG.prompt, answer_format="cord", output_schema=OUTPUT_SCHEMA,
                                     output_dir=ROOT / "full" / "validation")["summary"]
    base_model, base_processor = engine.load_base(CFG)
    base = engine.evaluate_condition(base_model, base_processor, val_raw, everything, CFG, desc="base + short prompt",
                                     user_prompt=CFG.prompt, answer_format="cord", output_schema=OUTPUT_SCHEMA,
                                     output_dir=ROOT / "baseline_4bit_short_prompt")["summary"]
    skill = engine.evaluate_condition(base_model, base_processor, val_raw, everything, CFG, desc="base + full skill",
                                      system_prompt=FULL_SKILL, user_prompt=engine.SKILL_USER_PROMPT,
                                      answer_format="expense", output_schema=OUTPUT_SCHEMA,
                                      output_dir=ROOT / "baseline_4bit_full_skill")["summary"]
    del base_model, base_processor
    engine.free_gpu()

    b, k, l = base, skill, lora
    be, le = base["expense_format"], lora["expense_format"]
    rows = [
        ("CORD field F1 (baseline scorer)", b["field_f1"], None, l["field_f1"]),
        ("CORD field precision", b["field_precision"], None, l["field_precision"]),
        ("CORD field recall", b["field_recall"], None, l["field_recall"]),
        ("CORD total price exact", b["total_price_exact_fraction"], None, l["total_price_exact_fraction"]),
        ("CORD document exact", b["document_exact_fraction"], None, l["document_exact_fraction"]),
        ("Skill-format field F1", be["field_f1"], k["field_f1"], le["field_f1"]),
        ("Skill-format line-item F1", be["line_item_f1"], k["line_item_f1"], le["line_item_f1"]),
        ("Skill-format total accuracy", be["total_accuracy"], k["total_accuracy"], le["total_accuracy"]),
        ("Skill-format subtotal accuracy", be["subtotal_accuracy"], k["subtotal_accuracy"], le["subtotal_accuracy"]),
        ("Valid JSON", b["valid_json_fraction"], k["valid_json_fraction"], l["valid_json_fraction"]),
        ("Schema valid (skill gate)", None, k["schema_valid_fraction"], None),
        ("Mean input tokens", b["mean_input_tokens"], k["mean_input_tokens"], l["mean_input_tokens"]),
        ("Mean output tokens", b["mean_output_tokens"], k["mean_output_tokens"], l["mean_output_tokens"]),
        ("Seconds per receipt", b["mean_seconds_per_receipt"], k["mean_seconds_per_receipt"], l["mean_seconds_per_receipt"]),
        ("Truncated outputs", b["truncated_outputs"], k["truncated_outputs"], l["truncated_outputs"]),
        ("Peak GPU memory (GiB)", b["peak_allocated_gib"], k["peak_allocated_gib"], l["peak_allocated_gib"]),
    ]
    def cell(value):
        return "n/a" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))
    print("FINAL COMPARISON (4-bit Qwen3-VL-8B, 100 CORD validation receipts)")
    print(f"FINAL ROW {'Metric':32s} | {'base':>10s} | {'base + full skill':>17s} | {'LoRA':>10s}")
    for name, x, y, z in rows:
        print(f"FINAL ROW {name:32s} | {cell(x):>10s} | {cell(y):>17s} | {cell(z):>10s}")
    print("FINAL ROW Base + cached skill: not measured (needs a verified prefix-cache decoder; accuracy is identical to the full skill by construction)")
    print(f"FINAL ROW Reference: BF16 base + short prompt, earlier run: CORD field F1 {engine.BASELINE_BF16['field_f1']}")
    FINAL_COMPARISON_REPORT = {
        "base_short_prompt": base, "base_full_skill": skill, "lora_short_prompt": lora,
        "base_cached_skill": "not measured", "bf16_reference": engine.BASELINE_BF16,
        "lora_vs_base_cord": engine.compare(lora, base),
        "notes": ["20 of the 100 validation receipts were used for hyperparameter selection.",
                  "Skill-format metrics follow research/adapter-assets/skills/receipt-to-expense/evaluation_plan.json.",
                  "Latency uses an unmerged 4-bit LoRA through Unsloth generate; not a serving benchmark."],
    }
    engine.write_json(ROOT / "final_comparison.json", FINAL_COMPARISON_REPORT)
    print("FINAL COMPARISON DONE")
"""

bench_source = (HERE / "dapta_bench.py").read_text(encoding="utf-8")
bench_sha = hashlib.sha256(bench_source.encode()).hexdigest()[:12]

VLLM_INSTALL = """
# Separate environment for the vLLM server: vLLM pins torch 2.13 / transformers 5, the training stack uses
# torch 2.11 / transformers 4.57.6. Nothing here changes the notebook kernel's packages.
import os, subprocess, sys
VLLM_ENV = "/content/vllm-env"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv==0.12.19"], check=True)
if not os.path.exists(f"{VLLM_ENV}/bin/python"):
    subprocess.run(["uv", "venv", VLLM_ENV, "--python", sys.executable], check=True)
subprocess.run(["uv", "pip", "install", "--python", f"{VLLM_ENV}/bin/python", "-q",
                "vllm==0.30.0", "transformers==5.17.0"], check=True)
check = subprocess.run([f"{VLLM_ENV}/bin/python", "-c",
                        "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, "
                        "'| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"],
                       capture_output=True, text=True)
print(check.stdout, check.stderr[-2000:])
assert check.returncode == 0 and "vllm 0.30.0" in check.stdout, "vLLM environment check failed"
print("VLLM ENV OK")
"""

MERGE = """
# Only needed when vLLM cannot load the adapter directly (LORA_MODE = "merged" in cell 20).
MERGE_ADAPTER = False
ADAPTER, MERGED = "/content/dapta_lora/full/adapter", "/content/dapta_lora/full/merged_bf16"
if not MERGE_ADAPTER:
    print("Skipped: adapter is served directly by vLLM (--enable-lora).")
else:
    import subprocess, sys, textwrap
    script = textwrap.dedent(f'''
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        base = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cpu")
        merged = PeftModel.from_pretrained(base, "{ADAPTER}").merge_and_unload()
        merged.save_pretrained("{MERGED}", safe_serialization=True)
        AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct").save_pretrained("{MERGED}")
        print("merged into", "{MERGED}")
    ''')
    # A subprocess keeps the 16 GB merge out of this kernel's memory.
    subprocess.run([sys.executable, "-c", script], check=True)
"""

FINAL_BENCH = f"""
import hashlib, importlib, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "/content")
import dapta_vlm_engine as engine, dapta_bench as bench
importlib.reload(engine); importlib.reload(bench)
from datasets import Image as ImageFeature, load_dataset
from huggingface_hub import hf_hub_download

QUICK = True            # 5 test receipts per condition; set False only after approving the full-run estimate
LORA_MODE = "adapter"   # "adapter" = vLLM --enable-lora; "merged" = run cell 18 with MERGE_ADAPTER = True first
GPU_HOUR_USD = 1.50     # price of one A100 40GB GPU-hour; used for cost and break-even
ROOT = Path("/content/dapta_lora")
TEST_FILE, TEST_SHA256 = "data/test-00000-of-00001-9c204eb3f4e11791.parquet", "51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7"
VALIDATION_FILE = "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet"

used = bench.gpu_memory_used_mib()
assert used is not None and used < 2500, (f"{{used}} MiB of GPU memory is in use. Restart the kernel (files in /content are kept), "
                                         "then run cells 17-20 only.")
CFG_E = engine.EngineConfig().validate()

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

test_path = hf_hub_download(CFG_E.dataset_id, TEST_FILE, repo_type="dataset", revision=CFG_E.dataset_revision)
assert sha256_file(test_path) == TEST_SHA256, "the CORD test split does not match the locked hash"
test = load_dataset("parquet", data_files=test_path, split="train").cast_column("image", ImageFeature())
assert len(test) == 100
val_path = hf_hub_download(CFG_E.dataset_id, VALIDATION_FILE, repo_type="dataset", revision=CFG_E.dataset_revision)
validation = load_dataset("parquet", data_files=val_path, split="train").cast_column("image", ImageFeature())
DOCS = bench.encode_docs(test, range(len(test)), CFG_E)
WARM = bench.encode_docs(validation, [97, 98, 99], CFG_E)  # warmup never touches test receipts

SKILL_SPEC = json.loads(r'''{skill_spec_text}''')
OUTPUT_SCHEMA = json.loads(r'''{output_schema_text}''')
REVIEWED_SHA256 = {json.dumps(review["artifact_sha256"])}
for name, value in (("skill_spec.json", SKILL_SPEC), ("output_schema.json", OUTPUT_SCHEMA)):
    assert hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == REVIEWED_SHA256[name]
FULL_SKILL = engine.full_skill_prompt(SKILL_SPEC, OUTPUT_SCHEMA)

full_summary = ROOT / "full" / "train_summary.json"
assert full_summary.exists(), "full training summary missing: run cell 13 first"
train_seconds = json.loads(full_summary.read_text())["train_seconds"]
tuning_files = [p for p in ROOT.glob("*/train_summary.json") if p.parent.name not in ("full", "tests")]
tuning_seconds = sum(json.loads(p.read_text())["train_seconds"] for p in tuning_files) if tuning_files else None
if LORA_MODE == "merged":
    assert (ROOT / "full" / "merged_bf16" / "config.json").exists(), "run cell 18 with MERGE_ADAPTER = True first"
versions = subprocess.run(["/content/vllm-env/bin/python", "-c", "import vllm, torch; print(vllm.__version__, torch.__version__)"],
                          capture_output=True, text=True, check=True).stdout.split()
PROVENANCE = {{"test_file": TEST_FILE, "test_sha256": TEST_SHA256, "dataset_revision": CFG_E.dataset_revision,
              "test_documents": len(DOCS), "warmup_documents": "validation 97-99", "vllm": versions[0], "vllm_torch": versions[1],
              "adapter_config_sha256": sha256_file(ROOT / "full" / "adapter" / "adapter_config.json"),
              "engine_sha": "{engine_sha}", "bench_sha": "{bench_sha}", "run_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}}

CFG_B = bench.BenchConfig(lora_mode=LORA_MODE, gpu_hour_usd=GPU_HOUR_USD, results_dir=str(ROOT / "results")).validate()
REPORT = bench.run(CFG_B, DOCS, WARM, FULL_SKILL, OUTPUT_SCHEMA, quick=QUICK, provenance=PROVENANCE,
                   training={{"train_seconds": train_seconds,
                             "train_seconds_including_tuning": train_seconds + tuning_seconds if tuning_seconds is not None else None}})
out = Path(CFG_B.results_dir) / ("quick" if QUICK else "full")
print("\\nFINAL BENCHMARK", "(QUICK RUN, 5 test receipts)" if QUICK else "(FULL RUN, 100 test receipts)")
for line in bench.headline(REPORT):
    print("FINAL ROW", line)
for line in bench.plain_read(REPORT):
    print("FINAL READ", line)
if QUICK:
    estimate = bench.estimate_full_minutes(REPORT, CFG_B, documents=len(DOCS))
    print(f"FINAL ESTIMATE full run: about {{estimate['minutes']:.0f}} minutes of A100 time (~${{estimate['usd']:.2f}}). "
          "Approve, then set QUICK = False and rerun this cell.")
print("Outputs:", sorted(p.name for p in out.iterdir()))
if "persist" in globals():
    persist([f"results/{{out.name}}/**"], f"four-way benchmark ({{out.name}})")
else:
    print("Kernel was restarted: results are only in", out, "- run cell 16 or copy them before the runtime ends.")
from IPython.display import Image as Show, display
for chart in ("accuracy_by_condition.png", "prompt_tokens_by_condition.png", "ttft_and_latency.png", "throughput_vs_concurrency.png"):
    display(Show(filename=str(out / chart)))
"""

CELLS = [
    md("""
## Execution order
Run cells top to bottom (**Run All** is safe). Each stage has a gate:

| Cell | Stage | Stops the run if |
|---|---|---|
| 1-3 | Location, pinned install, environment check | wrong GPU, any version mismatch, Pillow/torchvision mismatch |
| 4-6 | Engine module, settings, CORD data and schema checks | invalid config, bad rows, any example over `max_seq_len` |
| 7 | **Skill baseline**: 4-bit base model with the short prompt vs with the whole receipt skill, 100 validation receipts | skill files differ from the reviewed contract |
| 8 | GPU compatibility tests | collator masking, gradients, frozen weights or save/reload is wrong |
| 9 | Smoke test (12 receipts, 5 steps) | non-finite loss or no answer tokens supervised |
| 10 | Microbatch benchmark (1, 2, 4 at effective batch 8) | nothing completes stably |
| 11 | Hyperparameter trials (3 trials, 256 train, 20 validation, 35 steps) | |
| 12 | Recommendation and **approval gate** | full training runs only after you set `APPROVE_FULL_TRAINING = True` |
| 13-16 | Full training, evaluation vs 4-bit control and BF16 baseline, final test, export | skipped until approved |

Progress is printed as plain text so it stays readable in the saved notebook. Every stage also writes JSON under `/content/dapta_lora/`.
"""),
    md(f"## Cell 4: engine module\nThe tested training engine, written verbatim from `notebooks/dapta_vlm_engine.py` (sha `{engine_sha}`)."),
    code("%%writefile /content/dapta_vlm_engine.py\n" + engine_source),
    md("## Cell 5: settings\nEdit here only. The extraction prompt is fixed in the engine so results stay comparable with the baseline."),
    code("""
import importlib, json, math, random, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "/content")
import dapta_vlm_engine as engine
importlib.reload(engine)

CFG = engine.EngineConfig(
    image_long_side=1280,   # same resize for training and evaluation
    max_seq_len=4096,       # every example is checked against this; nothing is silently truncated
    max_new_tokens=1536,    # matches the zero-shot baseline
).validate()
ROOT = Path(CFG.root); ROOT.mkdir(parents=True, exist_ok=True)

SMOKE_TRAIN, SMOKE_STEPS, SMOKE_EVAL = 12, 5, [20, 21, 22]   # eval receipts are outside the tuning subset
BATCH_CANDIDATES, EFFECTIVE_BATCH, BENCH_STEPS = [1, 2, 4], 8, 5
TUNE_TRAIN_SIZE, TUNE_VAL_INDICES, TUNE_STEPS = 256, list(range(20)), 35
TUNE_TRIALS = [{"rank": 8, "lr": 1e-4}, {"rank": 16, "lr": 1e-4}, {"rank": 16, "lr": 2e-4}]
FULL_EPOCHS, EVAL_EVERY, SAVE_EVERY, PATIENCE = 3, 25, 25, 2

engine.set_seed(CFG.seed)
torch.backends.cuda.matmul.allow_tf32 = True

# Results are copied to your private Hub repo after each stage: Colab deletes /content when a runtime ends.
import getpass
HF_REPO = "Pavan2020/DAPTA"
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
HF_TOKEN = getpass.getpass("Hugging Face write token (press Enter to skip saving to the Hub): ").strip() or None
def persist(patterns, message):
    return engine.persist_to_hub(ROOT, HF_REPO, HF_TOKEN, RUN_ID, patterns, message)
print(json.dumps({"config": CFG.__dict__, "engine": engine.__file__}, indent=2))
"""),
    md("## Cell 6: CORD data, schema and length checks\nPinned dataset revision. Conversations are built lazily, so images are decoded and resized on access."),
    code("""
from datasets import load_dataset

train_raw = load_dataset(CFG.dataset_id, revision=CFG.dataset_revision, split="train")
val_raw = load_dataset(CFG.dataset_id, revision=CFG.dataset_revision, split="validation")
assert (len(train_raw), len(val_raw)) == (800, 100), (len(train_raw), len(val_raw))
print("schema:", engine.check_rows(train_raw, range(len(train_raw))), engine.check_rows(val_raw, range(len(val_raw))))

rng = random.Random(CFG.seed)
train_order = list(range(len(train_raw))); rng.shuffle(train_order)
smoke_ds = engine.receipt_dataset(train_raw, train_order[:SMOKE_TRAIN], CFG)
tune_train_ds = engine.receipt_dataset(train_raw, train_order[:TUNE_TRAIN_SIZE], CFG)
full_train_ds = engine.receipt_dataset(train_raw, range(len(train_raw)), CFG)
tune_val_ds = engine.receipt_dataset(val_raw, TUNE_VAL_INDICES, CFG)
smoke_val_ds = engine.receipt_dataset(val_raw, SMOKE_EVAL, CFG)
full_val_ds = engine.receipt_dataset(val_raw, range(len(val_raw)), CFG)
# Largest images and longest answers, so the benchmark measures worst-case memory.
stress_ids = sorted(train_order[:96], key=lambda i: train_raw[i]["image"].width * train_raw[i]["image"].height
                    + 128 * len(train_raw[i]["ground_truth"]), reverse=True)[:40]
stress_ds = engine.receipt_dataset(train_raw, stress_ids, CFG)

from transformers import AutoProcessor
_processor = AutoProcessor.from_pretrained(CFG.model_id)
LENGTHS = {
    "train": engine.length_report(engine.sequence_lengths(_processor, full_train_ds), CFG.max_seq_len),
    "validation": engine.length_report(engine.sequence_lengths(_processor, full_val_ds), CFG.max_seq_len),
}
del _processor
engine.write_json(ROOT / "data_report.json", {"lengths": LENGTHS, "stress_ids": stress_ids})
print(json.dumps(LENGTHS, indent=2))
"""),
    md("""
## Cell 7: skill baseline (base model, short prompt vs whole skill)
The 4-bit base model, no adapter, on all 100 validation receipts, twice:
1. **Short prompt**: the baseline's prompt, CORD-format answers, graded by the baseline scorer.
2. **Whole skill**: the repo's `receipt-to-expense` skill document and output schema, unchanged, as the system prompt.

The skill asks for a different format (expense report, amounts as plain numbers), so both conditions are also graded on that
format with the skill's own `evaluation_plan.json` rules; CORD gold and the short-prompt answers are converted with the same
code. Amount parsing reads Indonesian `18.000` as eighteen thousand.
"""),
    code(SKILL_BASELINE),
    md("## Cell 8: GPU compatibility tests\nReal 4-bit model, Unsloth collator, LoRA. Raises on the first failure."),
    code("""
model, processor = engine.load_lora(CFG, rank=8)
collator = engine.make_collator(model, processor, CFG)
TESTS = {"parameters": engine.trainable_report(model)}
TESTS["collator_batch_of_2"] = engine.probe_collator(collator, processor, smoke_ds, rows=2)
TESTS["gradients"] = engine.gradient_check(model, collator, smoke_ds)
TESTS["save_reload"] = engine.save_reload_check(model, processor, CFG, ROOT / "tests" / "adapter")
del model, processor, collator
TESTS["memory_after_release"] = engine.free_gpu()
engine.write_json(ROOT / "tests.json", TESTS)
print(json.dumps(TESTS, indent=2))
print("GPU TESTS PASSED")
"""),
    md("## Cell 9: smoke test\n12 receipts, 5 optimizer steps, rank 8, lr 1e-4, batch 1 x accumulation 2. Then generation on 3 separate validation receipts."),
    code("""
SMOKE_PASSED = False
spec = engine.TrainSpec("smoke", rank=8, lr=1e-4, batch=1, accum=2, steps=SMOKE_STEPS)
smoke, model, processor, trainer = engine.train_run(spec, CFG, smoke_ds, val_ds=smoke_val_ds)
assert smoke["stable"], f"non-finite training loss: {smoke}"
assert smoke["steps_done"] == SMOKE_STEPS
generation = engine.evaluate_receipts(model, processor, val_raw, SMOKE_EVAL, CFG,
                                      desc="smoke eval", output_dir=ROOT / "smoke" / "generation", print_every=1)
smoke["generation"] = generation["summary"]
del model, processor, trainer
smoke["memory_after_release"] = engine.free_gpu()
engine.write_json(ROOT / "smoke_metrics.json", smoke)
print(json.dumps(smoke, indent=2))
SMOKE_PASSED = True
print("SMOKE TEST PASSED")
"""),
    md("## Cell 10: microbatch benchmark\nReal training steps on the 40 largest receipts, effective batch 8. An OOM is recorded, not hidden."),
    code("""
assert SMOKE_PASSED, "run the smoke test first"
BENCH = []
steps_per_epoch = math.ceil(len(full_train_ds) / EFFECTIVE_BATCH)
for batch in BATCH_CANDIDATES:
    accum = EFFECTIVE_BATCH // batch
    print(f"\\n=== microbatch {batch} x accumulation {accum} ===", flush=True)
    row = {"microbatch": batch, "grad_accum": accum, "oom": False}
    try:
        result, model, processor, trainer = engine.train_run(
            engine.TrainSpec(f"bench_bs{batch}", rank=16, lr=1e-4, batch=batch, accum=accum, steps=BENCH_STEPS),
            CFG, stress_ds)
        row.update({k: result[k] for k in ("stable", "median_step_seconds", "examples_per_second",
                                            "peak_allocated_gib", "peak_reserved_gib", "total_gib",
                                            "gpu_utilization_mean", "first_loss", "last_loss")})
        row["estimated_full_training_minutes"] = round(steps_per_epoch * FULL_EPOCHS * result["median_step_seconds"] / 60, 1)
        del model, processor, trainer
    except torch.OutOfMemoryError as error:
        row.update({"oom": True, "error": str(error).splitlines()[0]})
    engine.free_gpu()
    BENCH.append(row)
    print(json.dumps(row, indent=2), flush=True)

capacity = torch.cuda.get_device_properties(0).total_memory / 2**30
CHOICE = engine.choose_batch(BENCH, capacity_gib=capacity)
BEST_BATCH, BEST_ACCUM = CHOICE["microbatch"], CHOICE["grad_accum"]
engine.write_json(ROOT / "batch_benchmark.json", {"rows": BENCH, "choice": CHOICE, "capacity_gib": round(capacity, 2)})
print("CHOICE:", json.dumps(CHOICE, indent=2))
"""),
    md("## Cell 11: hyperparameter trials\nEach trial loads fresh base weights and a new adapter. Selection: validation field F1, then validation loss."),
    code("""
HPO, PREDICTIONS = [], {}
for number, trial in enumerate(TUNE_TRIALS, 1):
    name = f"hpo_{number}_r{trial['rank']}_lr{trial['lr']:g}"
    print(f"\\n===== trial {number}/{len(TUNE_TRIALS)}: {name} =====", flush=True)
    spec = engine.TrainSpec(name, rank=trial["rank"], lr=trial["lr"], batch=BEST_BATCH, accum=BEST_ACCUM,
                            steps=TUNE_STEPS, log_every=5)
    result, model, processor, trainer = engine.train_run(spec, CFG, tune_train_ds, val_ds=tune_val_ds)
    evaluation = engine.evaluate_receipts(model, processor, val_raw, TUNE_VAL_INDICES, CFG,
                                          desc=name, output_dir=ROOT / name / "generation", print_every=5)
    del model, processor, trainer
    engine.free_gpu()
    summary = evaluation["summary"]
    result.update({"val_field_f1": summary["field_f1"], "val_field_precision": summary["field_precision"],
                   "val_field_recall": summary["field_recall"], "val_valid_json": summary["valid_json_fraction"],
                   "val_total_price_exact": summary["total_price_exact_fraction"]})
    HPO.append(result)
    PREDICTIONS[name] = evaluation["records"]
    engine.write_json(ROOT / "hpo_results.json", HPO)
    print(json.dumps({k: result[k] for k in ("name", "rank", "lr", "training_loss", "val_loss", "val_field_f1",
                                             "val_field_precision", "val_field_recall", "gpu_utilization_mean")}, indent=2), flush=True)

DECISION = engine.choose_trial(HPO, PREDICTIONS)
engine.write_json(ROOT / "best_hpo.json", DECISION)
print(json.dumps({k: v for k, v in DECISION.items() if k != "best"}, indent=2))
"""),
    md("## Cell 12: recommendation and approval gate\nFull training does **not** start until you set `APPROVE_FULL_TRAINING = True` and run cells 13-16."),
    code("""
# Tuning results come from this runtime, else its saved files, else the approved outcome recorded on 2026-10-17
# (the runtime that ran tuning ended: hpo_2 r16 lr1e-4 F1 0.808 on 20 validation receipts, microbatch 4 x 2).
if "DECISION" not in globals() and (ROOT / "best_hpo.json").exists():
    DECISION = json.loads((ROOT / "best_hpo.json").read_text())
    _bench = json.loads((ROOT / "batch_benchmark.json").read_text())
    BENCH, BEST_BATCH, BEST_ACCUM = _bench["rows"], _bench["choice"]["microbatch"], _bench["choice"]["grad_accum"]
if "DECISION" not in globals():
    DECISION = {"best": {"name": "hpo_2_r16_lr0.0001", "rank": 16, "lr": 1e-4, "val_field_f1": 0.80798},
                "conclusive": False, "source": "approved outcome; tuning files lost with the earlier runtime",
                "recommendation": "hpo_2 (lr 1e-4) and hpo_3 (lr 2e-4) were not separable on 20 validation receipts."}
    BENCH = [{"microbatch": 4, "grad_accum": 2, "estimated_full_training_minutes": 26.4}]
    BEST_BATCH, BEST_ACCUM = 4, 2
best = DECISION["best"]
RECOMMENDED = engine.TrainSpec("full", rank=int(best["rank"]), lr=float(best["lr"]), batch=BEST_BATCH, accum=BEST_ACCUM,
                               epochs=FULL_EPOCHS, eval_every=EVAL_EVERY, save_every=SAVE_EVERY,
                               early_stop_patience=PATIENCE, log_every=5).validate()
bench = next(r for r in BENCH if r["microbatch"] == BEST_BATCH)
print(json.dumps({"recommended": RECOMMENDED.__dict__, "conclusive": DECISION["conclusive"],
                  "note": DECISION.get("recommendation", "best trial is separable from the runner-up"),
                  "total_steps": RECOMMENDED.total_steps(len(full_train_ds)),
                  "estimated_training_minutes_excluding_eval": bench.get("estimated_full_training_minutes")}, indent=2))
# Approved by Pavan on 2026-10-17 after reviewing tuning: rank 16, lr 1e-4, batch 4 x 2, 3 epochs.
APPROVE_FULL_TRAINING = True
assert (RECOMMENDED.rank, RECOMMENDED.lr, RECOMMENDED.batch, RECOMMENDED.accum) == (16, 1e-4, 4, 2), RECOMMENDED
print("Full training approved: run cell 13.")
"""),
    md("## Cell 13: full training (800 receipts)\nCheckpoints every `SAVE_EVERY` steps, early stopping on validation loss, best checkpoint restored. If Colab disconnects or runs out of memory, rerun this cell: it resumes from the last checkpoint."),
    code("""
FULL_DONE = False
if not globals().get("APPROVE_FULL_TRAINING"):
    print("Skipped: full training not approved.")
else:
    try:
        FULL, full_model, full_processor, full_trainer = engine.train_run(
            RECOMMENDED, CFG, full_train_ds, val_ds=full_val_ds, save=True, resume=True)
    except torch.OutOfMemoryError as error:
        print("OUT OF MEMORY:", str(error).splitlines()[0])
        print("Checkpoints are kept. Lower the microbatch (keep the effective batch), then rerun this cell to resume.")
        raise
    engine.write_json(ROOT / "full_summary.json", FULL)
    print(json.dumps(FULL, indent=2))
    persist(["full_summary.json", "full/adapter/**", "full/train_summary.json", "full/train_history.jsonl"], "trained adapter")
    FULL_DONE = True
"""),
    md("""
## Cell 14: final comparison (100 validation receipts)
All conditions use the same 4-bit base weights, image preprocessing, greedy decoding and receipts. The two base-model
conditions reuse cell 7's saved predictions; only the LoRA is generated here.
"""),
    code(FINAL_COMPARISON),
    md("## Cell 15: final held-out test (run once, after every decision is made)"),
    code("""
RUN_FINAL_TEST = True  # selection is final (approved configuration); the test split is scored once
if not (RUN_FINAL_TEST and globals().get("FULL_DONE")):
    print("Test split untouched. Set RUN_FINAL_TEST = True only when the configuration is final.")
else:
    test_raw = load_dataset(CFG.dataset_id, revision=CFG.dataset_revision, split="test")
    assert len(test_raw) == 100
    FINAL = engine.evaluate_condition(full_model, full_processor, test_raw, range(len(test_raw)), CFG,
                                      desc="FINAL test, LoRA", user_prompt=CFG.prompt, answer_format="cord",
                                      output_schema=OUTPUT_SCHEMA, output_dir=ROOT / "full" / "final_test")["summary"]
    engine.write_json(ROOT / "final_test.json", FINAL)
    print(json.dumps(FINAL, indent=2))
"""),
    md("## Cell 16: export\nCollects metrics and the adapter into one zip. `EXPORT_TO_HF = True` also uploads it to your private Hugging Face repo."),
    code("""
import shutil
EXPORT_TO_HF, HF_REPO = False, "Pavan2020/DAPTA"
reports = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.json") if "checkpoints" not in p.parts)
print("Reports:", json.dumps(reports, indent=2))
staging = Path("/content/dapta_export"); shutil.rmtree(staging, ignore_errors=True)
for rel in reports:
    (staging / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, staging / rel)
if (ROOT / "full" / "adapter").is_dir():
    shutil.copytree(ROOT / "full" / "adapter", staging / "full" / "adapter")
archive = shutil.make_archive("/content/dapta_cord_lora_export", "zip", root_dir=staging)
print("Archive:", archive)
if EXPORT_TO_HF:
    import getpass
    from huggingface_hub import HfApi
    run_id = time.strftime("%Y%m%d-%H%M%S")
    HfApi(token=getpass.getpass("Hugging Face write token: ")).upload_folder(
        repo_id=HF_REPO, folder_path=str(staging), path_in_repo=f"cord-runs/{run_id}",
        commit_message=f"DAPTA CORD run {run_id}")
    print("Uploaded to", f"{HF_REPO}/cord-runs/{run_id}")
"""),
    md("""
## Cell 17: vLLM server environment
The four-way benchmark serves every condition through vLLM. vLLM pins its own torch, so it gets a separate virtual
environment; the training packages in this kernel are not touched. Takes about 5 minutes once.

**Before cells 17-20, restart the kernel** (Kernel > Restart): it frees the GPU for vLLM, and every file in `/content` is kept.
"""),
    code(VLLM_INSTALL),
    md("## Cell 18: merged adapter (fallback only)\nRun with `MERGE_ADAPTER = True` only if vLLM rejects the adapter in the quick run; then set `LORA_MODE = \"merged\"` in cell 20."),
    code(MERGE),
    md(f"## Cell 19: benchmark module\nWritten verbatim from `notebooks/dapta_bench.py` (sha `{bench_sha}`)."),
    code("%%writefile /content/dapta_bench.py\n" + bench_source),
    md("""
## Cell 20: four-way benchmark (final)
**base** (short prompt) vs **skill_uncached** vs **skill_cached** (vLLM prefix caching, warmed) vs **lora** (short prompt + adapter),
all BF16 through vLLM on the locked CORD test split (sha256 asserted). Quality with 95% bootstrap CIs, cost per 1,000 documents,
TTFT / latency / decode speed at concurrency 1, throughput at 1, 8 and 32, and LoRA break-even. Runs `QUICK = True` first
(5 receipts per condition) and prints the full-run GPU estimate; set `QUICK = False` only after you approve it.
Writes `results/{quick,full}/benchmark_runs.csv`, `benchmark_summary.csv`, `benchmark_report.md`, `DECISIONS.md` and four charts.
"""),
    code(FINAL_BENCH),
]

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
notebook["cells"] = notebook["cells"][:KEEP] + CELLS
NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {NOTEBOOK.name}: {len(notebook['cells'])} cells, engine sha {engine_sha}")
