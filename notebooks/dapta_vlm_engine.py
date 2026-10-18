"""DAPTA VLM QLoRA engine: Qwen3-VL document extraction with Unsloth.

The notebook is the interface; this module holds the stable logic so every stage
(tests, smoke, benchmark, tuning, full training, evaluation) uses the same code.
GPU-only dependencies (torch, unsloth, trl) are imported inside the functions that
need them, so data, scoring, masking audits and selection logic run anywhere.
"""

from __future__ import annotations

import gc
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# The baseline's short extraction prompt. Changing it breaks comparability with the
# zero-shot baseline, so it is fixed here rather than in the notebook settings.
PROMPT = (
    "Extract this receipt as JSON. Use CORD-style keys: menu (nm, cnt, "
    "unitprice, price), sub_total (subtotal_price, tax_price), and "
    "total (total_price, cashprice, changeprice). "
    "Include only fields visible on the receipt. Return only JSON."
)
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"
END_OF_TURN = "<|im_end|>"
VISUAL_TOKENS = ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")
BASELINE_BF16 = {  # zero-shot, BF16, all 100 CORD validation receipts, same scorer and prompt
    "field_f1": 0.443793,
    "field_precision": 0.400794,
    "field_recall": 0.497129,
    "valid_json_fraction": 1.0,
    "total_price_exact_fraction": 0.387755,
    "mean_seconds_per_receipt": 9.918,
    "mean_input_tokens": 1201.33,
    "mean_output_tokens": 198.64,
}


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class EngineConfig:
    model_id: str = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
    dataset_id: str = "naver-clova-ix/cord-v2"
    # Pinned so every run sees the same 800/100/100 receipts.
    dataset_revision: str = "7f0115a4b758a71d6473b8d085751692da2fef98"
    prompt: str = PROMPT
    seed: int = 42
    image_long_side: int = 1280
    # Unsloth defaults to 2048 and its collator then silently truncates; every example
    # is length-checked against this limit before training instead.
    max_seq_len: int = 4096
    max_new_tokens: int = 1536
    root: str = "/content/dapta_lora"

    def validate(self) -> "EngineConfig":
        _require(self.prompt == PROMPT, "prompt must stay the baseline's short prompt")
        _require(256 <= self.image_long_side <= 4096, "image_long_side must be 256..4096")
        _require(1024 <= self.max_seq_len <= 32768, "max_seq_len must be 1024..32768")
        _require(64 <= self.max_new_tokens <= 8192, "max_new_tokens must be 64..8192")
        _require(re.fullmatch(r"[0-9a-f]{40}", self.dataset_revision) is not None,
                 "dataset_revision must be a full commit sha")
        return self


@dataclass(frozen=True)
class TrainSpec:
    name: str
    rank: int
    lr: float
    batch: int
    accum: int
    steps: int | None = None
    epochs: float | None = None
    warmup_fraction: float = 0.05
    weight_decay: float = 0.01
    eval_every: int | None = None
    save_every: int | None = None
    early_stop_patience: int | None = None
    log_every: int = 1

    def validate(self) -> "TrainSpec":
        _require(re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.name) is not None, "invalid run name")
        _require(self.rank in (4, 8, 16, 32, 64), "rank must be one of 4, 8, 16, 32, 64")
        _require(0 < self.lr <= 1e-3, "lr must be in (0, 1e-3]")
        _require(self.batch >= 1 and self.accum >= 1, "batch and accum must be >= 1")
        _require((self.steps is None) != (self.epochs is None), "set exactly one of steps or epochs")
        _require(self.steps is None or self.steps >= 1, "steps must be >= 1")
        _require(self.epochs is None or 0 < self.epochs <= 10, "epochs must be in (0, 10]")
        _require(0 <= self.warmup_fraction < 0.5, "warmup_fraction must be in [0, 0.5)")
        _require(self.early_stop_patience is None or self.eval_every is not None,
                 "early stopping needs eval_every")
        return self

    @property
    def effective_batch(self) -> int:
        return self.batch * self.accum

    def total_steps(self, train_size: int) -> int:
        if self.steps is not None:
            return self.steps
        return math.ceil(train_size / self.effective_batch) * math.ceil(self.epochs)

    def warmup_steps(self, train_size: int) -> int:
        # Integer steps: transformers 5 renamed warmup_ratio, and integers mean the same everywhere.
        return max(1, math.ceil(self.warmup_fraction * self.total_steps(train_size)))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        from transformers import set_seed as hf_set_seed
    except ImportError:
        return
    hf_set_seed(seed)


# --------------------------------------------------------------------------- data


def gold_parse(row: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(row["ground_truth"])["gt_parse"]
    if not isinstance(value, dict):
        raise TypeError(f"gt_parse must be a JSON object, got {type(value).__name__}")
    return value


def target_text(gold: Mapping[str, Any]) -> str:
    """The exact training answer: compact JSON that must parse back to the gold object."""
    text = json.dumps(gold, ensure_ascii=False, separators=(",", ":"))
    if json.loads(text) != gold:
        raise ValueError("training target does not round-trip through JSON")
    return text


def prepare_image(image: Any, long_side: int) -> Any:
    """RGB copy with the long side capped; the same function serves training and evaluation."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError(f"expected a PIL image, got {type(image).__module__}.{type(image).__name__}")
    image = image.convert("RGB")
    if max(image.size) > long_side:
        image = image.copy()
        image.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    return image


def build_messages(image: Any, prompt: str, answer: str | None = None) -> list[dict[str, Any]]:
    """Qwen3-VL conversation: image then the short prompt, optionally the assistant answer."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def require_receipt_schema(split: Any) -> None:
    """CORD rows must carry a typed Image feature and a string ground_truth column."""
    from datasets import Image as ImageFeature, Value

    features = split.features
    _require("image" in features and "ground_truth" in features, f"expected image/ground_truth columns, got {list(features)}")
    if not isinstance(features["image"], ImageFeature):
        raise TypeError(f"'image' must be a datasets.Image feature, got {features['image']!r}; "
                        "use split.cast_column('image', datasets.Image())")
    _require(features["ground_truth"] == Value("string"), "'ground_truth' must be a string column")


def receipt_dataset(split: Any, indices: Sequence[int], cfg: EngineConfig) -> Any:
    """An HF Dataset (TRL accepts no other type) that builds conversations lazily on access."""
    from PIL import Image

    require_receipt_schema(split)
    subset = split.select(list(indices))
    # with_transform receives the stored record ({"bytes", "path"}), not a decoded image;
    # decode it through the split's own Image feature so the type is always a PIL image.
    image_feature = split.features["image"]

    def to_messages(batch: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
        conversations = []
        for stored, ground_truth in zip(batch["image"], batch["ground_truth"]):
            image = stored if isinstance(stored, Image.Image) else image_feature.decode_example(stored)
            answer = target_text(gold_parse({"ground_truth": ground_truth}))
            conversations.append(build_messages(prepare_image(image, cfg.image_long_side), cfg.prompt, answer))
        return {"messages": conversations}

    return subset.with_transform(to_messages)


def check_rows(split: Any, indices: Iterable[int]) -> dict[str, Any]:
    """Schema check of raw rows: gold JSON object, round-trip target, decodable RGB-able image."""
    require_receipt_schema(split)
    checked, failures = 0, []
    for index in indices:
        row = split[int(index)]
        try:
            target_text(gold_parse(row))
            row["image"].convert("RGB")
            checked += 1
        except (TypeError, ValueError, KeyError, OSError, json.JSONDecodeError) as error:
            failures.append({"index": int(index), "error": f"{type(error).__name__}: {error}"})
    if failures:
        raise ValueError(f"{len(failures)} rows failed schema checks: {failures[:5]}")
    return {"rows_checked": checked}


def sequence_lengths(processor: Any, dataset: Any) -> list[int]:
    """Exact token count of each training conversation, image tokens included."""
    lengths = []
    for index in range(len(dataset)):
        messages = dataset[index]["messages"]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image = messages[0]["content"][0]["image"]
        batch = processor(text=[text], images=[image], return_tensors="pt", add_special_tokens=False)
        lengths.append(int(batch["input_ids"].shape[-1]))
    return lengths


def length_report(lengths: Sequence[int], limit: int) -> dict[str, Any]:
    ordered = sorted(lengths)
    report = {
        "examples": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
        "limit": limit,
        "over_limit": sum(value > limit for value in ordered),
    }
    if report["over_limit"]:
        raise ValueError(f"{report['over_limit']} examples exceed max_seq_len {limit}; raise it or drop them")
    return report


# --------------------------------------------------------------------------- masking audit


def audit_batch(batch: Mapping[str, Any], tokenizer: Any, answers: Sequence[str]) -> dict[str, int]:
    """Prove the loss sees exactly each answer + end-of-turn, and never prompt or image tokens."""
    import torch

    input_ids, labels = batch["input_ids"], batch["labels"]
    _require(input_ids.shape == labels.shape, "labels must align with input_ids")
    _require(len(answers) == input_ids.shape[0], "one expected answer per row")
    visual_ids = [tokenizer.convert_tokens_to_ids(token) for token in VISUAL_TOKENS]
    visual = torch.isin(input_ids, torch.tensor(visual_ids, device=input_ids.device))
    _require(int(visual.sum()) > 0, "batch contains no visual tokens: images did not reach the processor")
    _require(not bool((labels[visual] != -100).any()), "visual tokens are supervised")
    if "attention_mask" in batch:
        padding = batch["attention_mask"] == 0
        _require(not bool((labels[padding] != -100).any()), "padding tokens are supervised")

    supervised_total = 0
    for row, answer in enumerate(answers):
        kept = labels[row] != -100
        _require(int(kept.sum()) > 0, f"row {row} has no supervised tokens")
        decoded = tokenizer.decode(input_ids[row][kept], skip_special_tokens=False)
        _require(decoded.startswith(answer), f"row {row} supervises text other than the answer: {decoded[:120]!r}")
        tail = decoded[len(answer):]
        _require(tail.strip() == END_OF_TURN, f"row {row} supervised tail is {tail!r}, expected {END_OF_TURN!r}")
        positions = kept.nonzero().flatten()
        _require(bool((positions[1:] - positions[:-1] == 1).all()), f"row {row} supervision is not contiguous")
        supervised_total += int(kept.sum())
    return {
        "rows": len(answers),
        "supervised_tokens": supervised_total,
        "ignored_tokens": int((labels == -100).sum()),
        "visual_tokens": int(visual.sum()),
        "sequence_length": int(input_ids.shape[1]),
    }


# --------------------------------------------------------------------------- scorer
# Unchanged from the baseline notebook: flattened (field path, normalized value) pairs,
# position-sensitive menu indices. Not the official CORD metric.


def parse_json_object(text):
    raw = text.strip()
    for s in (raw, re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)):
        try:
            obj = json.loads(s.strip())
            if isinstance(obj, dict): return obj
        except (json.JSONDecodeError, ValueError): pass
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", raw):
        try:
            obj, _ = decoder.raw_decode(raw[m.start():])
            if isinstance(obj, dict): return obj
        except json.JSONDecodeError: pass
    return None


def canonical(obj):
    if isinstance(obj, dict) and isinstance(obj.get("gt_parse"), dict): obj = obj["gt_parse"]
    if not isinstance(obj, dict): return {}
    obj = dict(obj)
    if isinstance(obj.get("menu"), dict): obj["menu"] = [obj["menu"]]
    return obj


def norm(v): return str(v).lower() if isinstance(v, bool) else " ".join(str(v).strip().casefold().split())


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        return [item for k, v in sorted(obj.items()) for item in flatten(v, f"{prefix}.{k}" if prefix else k)]
    if isinstance(obj, list):
        return [item for i, v in enumerate(obj) for item in flatten(v, f"{prefix}[{i}]")]
    return [] if obj is None else [(prefix, norm(obj))]


def score_example(pred, gold):
    p, g = canonical(pred), canonical(gold)
    a, b = Counter(flatten(p)), Counter(flatten(g))
    gt_total = g.get("total") if isinstance(g.get("total"), dict) else {}
    pr_total = p.get("total") if isinstance(p.get("total"), dict) else {}
    gold_price, pred_price = gt_total.get("total_price"), pr_total.get("total_price")
    return {"tp": sum((a & b).values()), "pred_fields": sum(a.values()), "gold_fields": sum(b.values()),
            "document_exact": a == b, "total_present": gold_price is not None,
            "total_exact": gold_price is not None and pred_price is not None and norm(gold_price) == norm(pred_price)}


def summarize(rows):
    tp = sum(r["score"]["tp"] for r in rows)
    npred = sum(r["score"]["pred_fields"] for r in rows)
    ngold = sum(r["score"]["gold_fields"] for r in rows)
    p = tp / npred if npred else 0.
    r = tp / ngold if ngold else 0.
    totals = [x for x in rows if x["score"]["total_present"]]
    n = len(rows)
    return {"samples": n, "field_precision": round(p, 6), "field_recall": round(r, 6),
            "field_f1": round(2*p*r/(p+r) if p+r else 0., 6),
            "valid_json_fraction": round(sum(x["valid_json"] for x in rows)/n, 6) if n else 0,
            "document_exact_fraction": round(sum(x["score"]["document_exact"] for x in rows)/n, 6) if n else 0,
            "total_price_exact_fraction": round(sum(x["score"]["total_exact"] for x in totals)/len(totals), 6) if totals else None,
            "mean_seconds_per_receipt": round(statistics.mean(x["elapsed_s"] for x in rows), 3) if n else None,
            "mean_input_tokens": round(statistics.mean(x["input_tokens"] for x in rows), 2) if n else None,
            "mean_output_tokens": round(statistics.mean(x["output_tokens"] for x in rows), 2) if n else None}


# --------------------------------------------------------------------------- receipt-to-expense skill
# Grading for answers in the repo's receipt-to-expense skill format, following that skill's
# evaluation_plan.json (cord_field_accuracy): line items match on normalized description and
# amount within 0.01, scalar amounts within 0.01. CORD gold is converted to the same format.

SKILL_SYSTEM_PREFIX = "Extract visible receipt evidence and return only JSON matching the receipt expense schema."
SKILL_USER_PROMPT = "Convert this receipt image into the normalized expense report. Use null for absent fields."
EXPENSE_SCALARS = ("subtotal", "tax", "service_charge", "total")
AMOUNT_ABS = 0.01


def full_skill_prompt(skill_spec: Mapping[str, Any], output_schema: Mapping[str, Any]) -> str:
    """The whole skill document, unchanged, as the system prompt."""
    return (f"{SKILL_SYSTEM_PREFIX}\n\nFollow this skill exactly.\n\nSKILL\n"
            f"{json.dumps(skill_spec, indent=2, ensure_ascii=False)}\n\nOUTPUT JSON SCHEMA\n"
            f"{json.dumps(output_schema, indent=2, ensure_ascii=False)}")


def parse_amount(value: Any) -> float | None:
    """Printed amount to a number. CORD receipts are Indonesian: '18.000' is eighteen thousand."""
    from decimal import Decimal, InvalidOperation

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        parsed = [p for p in (parse_amount(v) for v in value) if p is not None]
        return parsed[0] if parsed else None
    text = re.sub(r"[^0-9,.\-]", "", str(value).strip())
    if not re.search(r"\d", text):
        return None
    if "," in text and "." in text:
        decimal = "." if text.rfind(".") > text.rfind(",") else ","
        text = text.replace("," if decimal == "." else ".", "").replace(decimal, ".")
    elif "," in text:
        text = text.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", text) else text.replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")
    try:
        return float(Decimal(text))
    except InvalidOperation:
        return None


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def cord_to_expense(obj: Any) -> dict[str, Any]:
    """CORD gt_parse (gold, or a CORD-format prediction) in the skill's expense-receipt shape."""
    obj = canonical(obj)
    menu = obj.get("menu") if isinstance(obj.get("menu"), list) else []
    sub = obj.get("sub_total") if isinstance(obj.get("sub_total"), dict) else {}
    total = obj.get("total") if isinstance(obj.get("total"), dict) else {}
    return {
        "merchant": None, "address": None, "transaction_date": None, "currency": None,
        "subtotal": parse_amount(sub.get("subtotal_price")),
        "tax": parse_amount(sub.get("tax_price")),
        "service_charge": parse_amount(sub.get("service_price")),
        "total": parse_amount(total.get("total_price")),
        "line_items": [
            {"description": _text_or_none(item.get("nm")),
             "quantity": parse_amount(item.get("cnt")),
             "amount": parse_amount(item.get("price"))}
            for item in menu if isinstance(item, dict)
        ],
    }


def _same_amount(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or isinstance(a, bool) or isinstance(b, bool):
        return False
    return abs(float(a) - float(b)) <= AMOUNT_ABS


def _description(value: Any) -> str | None:
    return None if value is None else (" ".join(str(value).casefold().split()) or None)


def score_expense(prediction: Any, gold_receipt: Mapping[str, Any], validator: Any) -> dict[str, Any]:
    """One receipt. A schema-invalid answer scores zero (the skill plan's gate) but its gold still counts."""
    schema_valid = (isinstance(prediction, dict) and not any(True for _ in validator.iter_errors(prediction))
                    and len(prediction.get("receipts", [])) == 1)
    receipt = prediction["receipts"][0] if schema_valid else None
    gold_items = [(_description(i["description"]), i["amount"], i["quantity"]) for i in gold_receipt["line_items"]]
    pred_items = ([(_description(i.get("description")), i.get("amount"), i.get("quantity")) for i in receipt["line_items"]]
                  if receipt else [])
    used = [False] * len(pred_items)
    items_tp = 0
    for description, amount, quantity in gold_items:
        for j, (p_desc, p_amount, p_quantity) in enumerate(pred_items):
            if (not used[j] and p_desc == description and _same_amount(p_amount, amount)
                    and (quantity is None or _same_amount(p_quantity, quantity))):
                used[j] = True
                items_tp += 1
                break
    scalars = {}
    for field in EXPENSE_SCALARS:
        gold_value = gold_receipt[field]
        pred_value = receipt.get(field) if receipt else None
        scalars[field] = {"gold": gold_value is not None, "pred": pred_value is not None,
                          "correct": gold_value is not None and _same_amount(pred_value, gold_value)}
    return {"schema_valid": schema_valid, "items_tp": items_tp, "items_pred": len(pred_items),
            "items_gold": len(gold_items), "scalars": scalars,
            "currency_set": bool(receipt and receipt.get("currency") is not None)}


def summarize_expense(rows: Sequence[Mapping[str, Any]], *, schema_measured: bool = True) -> dict[str, Any]:
    scores = [r["expense"] for r in rows]
    n = len(scores)

    def prf(tp: int, pred: int, gold: int) -> tuple[float, float, float]:
        p = tp / pred if pred else 0.0
        r = tp / gold if gold else 0.0
        return round(p, 6), round(r, 6), round(2 * p * r / (p + r) if p + r else 0.0, 6)

    items = prf(sum(s["items_tp"] for s in scores), sum(s["items_pred"] for s in scores), sum(s["items_gold"] for s in scores))
    report: dict[str, Any] = {
        "samples": n,
        "schema_valid_fraction": round(sum(s["schema_valid"] for s in scores) / n, 6) if schema_measured and n else None,
        "line_item_precision": items[0], "line_item_recall": items[1], "line_item_f1": items[2],
    }
    tp, pred, gold = sum(s["items_tp"] for s in scores), sum(s["items_pred"] for s in scores), sum(s["items_gold"] for s in scores)
    for field in EXPENSE_SCALARS:
        labelled = [s["scalars"][field] for s in scores if s["scalars"][field]["gold"]]
        report[f"{field}_accuracy"] = round(sum(x["correct"] for x in labelled) / len(labelled), 6) if labelled else None
        report[f"{field}_unlabelled_but_predicted"] = sum(1 for s in scores if s["scalars"][field]["pred"] and not s["scalars"][field]["gold"])
        tp += sum(s["scalars"][field]["correct"] for s in scores)
        pred += sum(s["scalars"][field]["pred"] for s in scores)
        gold += sum(s["scalars"][field]["gold"] for s in scores)
    overall = prf(tp, pred, gold)
    report.update({"field_precision": overall[0], "field_recall": overall[1], "field_f1": overall[2],
                   "currency_set_count": sum(s["currency_set"] for s in scores)})
    for key in ("elapsed_s", "input_tokens", "output_tokens"):
        values = [r[key] for r in rows if key in r]
        report[f"mean_{'seconds_per_receipt' if key == 'elapsed_s' else key}"] = (
            round(statistics.mean(values), 3 if key == "elapsed_s" else 2) if values else None)
    report["valid_json_fraction"] = round(sum(r["valid_json"] for r in rows) / n, 6) if n else 0
    return report


def micro_f1(rows: Sequence[Mapping[str, Any]]) -> float:
    tp = sum(r["score"]["tp"] for r in rows)
    npred = sum(r["score"]["pred_fields"] for r in rows)
    ngold = sum(r["score"]["gold_fields"] for r in rows)
    p = tp / npred if npred else 0.0
    r = tp / ngold if ngold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def paired_bootstrap(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]],
                     *, samples: int = 2000, seed: int = 42) -> dict[str, float]:
    """95% interval of micro-F1(a) - micro-F1(b) over the same receipts, resampled together."""
    by_index_b = {row["index"]: row for row in b}
    pairs = [(row, by_index_b[row["index"]]) for row in a if row["index"] in by_index_b]
    _require(len(pairs) == len(a) == len(b), "bootstrap needs predictions for the same receipts")
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        drawn = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(micro_f1([x for x, _ in drawn]) - micro_f1([y for _, y in drawn]))
    deltas.sort()
    return {
        "delta": micro_f1(a) - micro_f1(b),
        "ci95_low": deltas[int(0.025 * samples)],
        "ci95_high": deltas[int(0.975 * samples) - 1],
    }


# --------------------------------------------------------------------------- selection


def choose_batch(rows: Sequence[Mapping[str, Any]], capacity_gib: float, headroom: float = 0.15) -> dict[str, Any]:
    """Fastest stable candidate whose peak reserved memory leaves `headroom` of the GPU free."""
    viable = [row for row in rows if not row.get("oom") and row.get("stable")]
    _require(bool(viable), "no microbatch completed stably; lower image_long_side and rerun")
    safe = [row for row in viable if row["peak_reserved_gib"] <= capacity_gib * (1 - headroom)]
    pool = safe or [min(viable, key=lambda row: row["peak_reserved_gib"])]
    best = max(pool, key=lambda row: row["examples_per_second"])
    return {
        "microbatch": best["microbatch"],
        "grad_accum": best["grad_accum"],
        "reason": ("fastest within memory headroom" if safe
                   else "no candidate within headroom; chose the lowest-memory stable candidate"),
        "headroom_fraction": headroom,
    }


def choose_trial(results: Sequence[Mapping[str, Any]], predictions: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Rank by validation field F1, then validation loss; flag when the top two are not separable."""
    ranked = sorted(results, key=lambda r: (-r["val_field_f1"], r["val_loss"]))
    best = dict(ranked[0])
    decision: dict[str, Any] = {"best": best, "ranking": [r["name"] for r in ranked], "conclusive": True}
    if len(ranked) > 1:
        runner = ranked[1]
        interval = paired_bootstrap(predictions[best["name"]], predictions[runner["name"]])
        decision["versus_runner_up"] = {"runner_up": runner["name"], **interval}
        if interval["ci95_low"] <= 0.0:
            decision["conclusive"] = False
            decision["recommendation"] = (
                f"{best['name']} and {runner['name']} are not separable on {len(predictions[best['name']])} "
                "validation receipts (95% interval of the F1 difference includes 0). Validate both on more "
                "receipts before treating either as best."
            )
    return decision


# --------------------------------------------------------------------------- GPU: model, collator, trainer


def gpu_memory() -> dict[str, float]:
    import torch

    gib = 2 ** 30
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / gib, 2),
        "reserved_gib": round(torch.cuda.memory_reserved() / gib, 2),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 2),
        "total_gib": round(torch.cuda.get_device_properties(0).total_memory / gib, 2),
    }


def free_gpu() -> dict[str, float]:
    """Collect garbage and empty the CUDA cache. Callers must drop their own references first."""
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    return gpu_memory()


def load_base(cfg: EngineConfig) -> tuple[Any, Any]:
    from unsloth import FastVisionModel

    set_seed(cfg.seed)
    model, processor = FastVisionModel.from_pretrained(
        cfg.model_id,
        load_in_4bit=True,
        max_seq_length=cfg.max_seq_len,
        use_gradient_checkpointing="unsloth",
    )
    return model, processor


def load_lora(cfg: EngineConfig, rank: int) -> tuple[Any, Any]:
    """Fresh 4-bit base weights plus a newly initialised language-model LoRA."""
    from unsloth import FastVisionModel

    model, processor = load_base(cfg)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # vision encoder stays frozen
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=int(rank), lora_alpha=int(rank), lora_dropout=0,
        bias="none", random_state=cfg.seed,
    )
    FastVisionModel.for_training(model)
    return model, processor


def trainable_report(model: Any) -> dict[str, Any]:
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in model.named_parameters())
    _require(bool(trainable), "no trainable parameters")
    bad = [name for name, _ in trainable if "lora_" not in name]
    _require(not bad, f"non-LoRA parameters are trainable: {bad[:5]}")
    vision = [name for name, _ in trainable if ".visual." in name or "vision_tower" in name]
    _require(not vision, f"vision-encoder LoRA found although vision layers are frozen: {vision[:5]}")
    count = sum(p.numel() for _, p in trainable)
    return {"trainable_parameters": count, "total_parameters": total,
            "trainable_fraction": round(count / total, 6), "trainable_tensors": len(trainable)}


def make_collator(model: Any, processor: Any, cfg: EngineConfig) -> Any:
    from unsloth.trainer import UnslothVisionDataCollator

    return UnslothVisionDataCollator(
        model, processor,
        max_seq_length=cfg.max_seq_len,  # explicit: the inherited default is 2048
        resize="max",  # no collator resize: Qwen3-VL has no image_size, so "min" fell back to 512px
        train_on_responses_only=True,
        instruction_part=INSTRUCTION_PART,
        response_part=RESPONSE_PART,
    )


def probe_collator(collator: Any, processor: Any, dataset: Any, rows: int = 2) -> dict[str, int]:
    """Collate real rows and audit the result; raises before any training step if masking is wrong."""
    examples = [dataset[i] for i in range(min(rows, len(dataset)))]
    batch = collator(examples)
    answers = [example["messages"][1]["content"][0]["text"] for example in examples]
    for key in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"):
        _require(key in batch, f"collator output is missing {key}")
    return audit_batch(batch, processor.tokenizer, answers)


def gradient_check(model: Any, collator: Any, dataset: Any) -> dict[str, Any]:
    """One forward/backward + optimizer step: LoRA moves, frozen weights stay bit-identical."""
    import torch

    model.train()
    batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in collator([dataset[0]]).items()}
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    all_frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    frozen = all_frozen[:: max(1, len(all_frozen) // 16)][:16]
    frozen_before = {n: p.detach().float().flatten()[:4096].clone().cpu() for n, p in frozen}
    lora_before = {n: p.detach().float().clone().cpu() for n, p in list(trainable.items())[:16]}

    loss = model(**batch).loss
    _require(torch.isfinite(loss).item(), f"loss is not finite: {loss.item()}")
    loss.backward()
    with_grad = [n for n, p in trainable.items() if p.grad is not None and float(p.grad.abs().sum()) > 0]
    _require(bool(with_grad), "no LoRA parameter received a gradient")
    _require(all(p.grad is None for _, p in frozen), "a frozen parameter received a gradient")
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    lora_changed = sum(not torch.equal(lora_before[n], trainable[n].detach().float().cpu()) for n in lora_before)
    frozen_changed = [n for n, p in frozen if not torch.equal(frozen_before[n], p.detach().float().flatten()[:4096].cpu())]
    _require(lora_changed > 0, "optimizer step did not change any sampled LoRA tensor")
    _require(not frozen_changed, f"frozen parameters changed: {frozen_changed[:3]}")
    return {"loss": round(float(loss.item()), 4), "lora_tensors_with_grad": len(with_grad),
            "lora_tensors_total": len(trainable), "sampled_lora_changed": lora_changed,
            "sampled_frozen_unchanged": len(frozen)}


def lora_state(model: Any) -> dict[str, Any]:
    return {n.split("base_model.model.")[-1]: p.detach().cpu().clone()
            for n, p in model.named_parameters() if "lora_" in n}


def save_reload_check(model: Any, processor: Any, cfg: EngineConfig, out_dir: Path) -> dict[str, Any]:
    """Save the adapter, reload it on fresh 4-bit base weights, and compare LoRA tensors exactly."""
    import torch
    from unsloth import FastVisionModel

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    files = sorted(p.name for p in out_dir.iterdir())
    _require("adapter_config.json" in files and "adapter_model.safetensors" in files,
             f"adapter files missing after save: {files}")
    before = lora_state(model)
    reloaded, _ = FastVisionModel.from_pretrained(str(out_dir), load_in_4bit=True, max_seq_length=cfg.max_seq_len)
    after = lora_state(reloaded)
    del reloaded
    free_gpu()
    _require(set(before) == set(after) and after, f"reloaded {len(after)} LoRA tensors, saved {len(before)}")
    # Compare in the reloaded dtype: a save/load may legitimately change float32 <-> bfloat16.
    mismatched = [n for n in before if not torch.equal(before[n].to(after[n].dtype), after[n])]
    _require(not mismatched, f"{len(mismatched)} LoRA tensors differ after reload: {mismatched[:3]}")
    dtypes = sorted({str(v.dtype) for v in after.values()})
    return {"adapter_dir": str(out_dir), "files": files, "lora_tensors_compared": len(before), "dtypes": dtypes}


class LiveProgress:
    """Plain-text progress (readable in saved notebook output) plus a JSONL history on disk."""

    def __init__(self, name: str, history_path: Path, log_every: int = 1):
        self.name = name
        self.history_path = history_path
        self.log_every = max(1, log_every)
        self.started = None
        self.last_step_end = None
        self.step_seconds: list[float] = []
        self.last_eval_loss = None
        self.utilization: list[int] = []

    def callback(self) -> Any:
        from transformers import TrainerCallback

        progress = self

        class _Callback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                progress.started = progress.last_step_end = time.perf_counter()
                progress.history_path.parent.mkdir(parents=True, exist_ok=True)

            def on_step_end(self, args, state, control, **kwargs):
                # on_step_end fires once per optimizer step; the gap between calls is the step time.
                now = time.perf_counter()
                progress.step_seconds.append(now - progress.last_step_end)
                progress.last_step_end = now
                sample = _gpu_utilization()
                if sample is not None:
                    progress.utilization.append(sample)

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                if "eval_loss" in logs:
                    progress.last_eval_loss = logs["eval_loss"]
                record = {"step": state.global_step, "epoch": state.epoch, "time": time.time(), **logs, **gpu_memory()}
                with progress.history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                if state.global_step % progress.log_every == 0 or "eval_loss" in logs or state.global_step == state.max_steps:
                    print(progress.format_line(state, logs), flush=True)

        return _Callback()

    def format_line(self, state: Any, logs: Mapping[str, Any]) -> str:
        elapsed = time.perf_counter() - (self.started or time.perf_counter())
        done, total = state.global_step, max(1, state.max_steps)
        eta = elapsed / done * (total - done) if done else float("nan")
        memory = gpu_memory()
        epochs = state.num_train_epochs if getattr(state, "num_train_epochs", None) else float("nan")
        loss = logs.get("loss")
        parts = [
            f"[{self.name}] Training: {done}/{total} steps",
            f"Progress: {100 * done / total:.1f}%",
            f"Epoch: {state.epoch or 0:.2f}/{epochs:.0f}" if epochs == epochs else f"Epoch: {state.epoch or 0:.2f}",
            f"Loss: {loss:.4f}" if loss is not None else "Loss: -",
            f"Validation loss: {self.last_eval_loss:.4f}" if self.last_eval_loss is not None else "Validation loss: -",
            f"Learning rate: {logs['learning_rate']:.1e}" if "learning_rate" in logs else "Learning rate: -",
            f"GPU memory: {memory['allocated_gib']:.1f}/{memory['total_gib']:.1f} GB (peak reserved {memory['peak_reserved_gib']:.1f})",
            f"Elapsed: {_clock(elapsed)}",
            f"ETA: {_clock(eta)}",
        ]
        return " | ".join(parts)


def _gpu_utilization() -> int | None:
    try:
        import torch

        return int(torch.cuda.utilization())
    except (ImportError, RuntimeError, ModuleNotFoundError):
        return None  # pynvml missing: utilization is reported as unavailable, not guessed


def _clock(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):
        return "-"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def make_trainer(model: Any, processor: Any, collator: Any, train_ds: Any, spec: TrainSpec,
                 cfg: EngineConfig, out_dir: Path, *, val_ds: Any = None, save: bool = False,
                 progress: LiveProgress | None = None) -> Any:
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    spec.validate()
    has_eval = val_ds is not None and spec.eval_every is not None
    keep_best = has_eval and save
    config = dict(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=spec.batch,
        gradient_accumulation_steps=spec.accum,
        per_device_eval_batch_size=1,
        learning_rate=spec.lr,
        warmup_steps=spec.warmup_steps(len(train_ds)),
        lr_scheduler_type="cosine",
        weight_decay=spec.weight_decay,
        optim="adamw_8bit",
        max_grad_norm=1.0,
        bf16=True, fp16=False,
        logging_strategy="steps", logging_steps=1,
        report_to="none",
        eval_strategy="steps" if has_eval else "no",
        eval_steps=spec.eval_every if has_eval else None,
        save_strategy="steps" if save else "no",
        save_steps=spec.save_every if save else None,
        save_total_limit=2 if save else None,
        load_best_model_at_end=keep_best,
        metric_for_best_model="eval_loss" if keep_best else None,
        greater_is_better=False if keep_best else None,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,  # the collator owns truncation; lengths were checked against max_seq_len
        dataloader_num_workers=0,
        seed=cfg.seed, data_seed=cfg.seed,
        disable_tqdm=True,
    )
    if spec.steps is not None:
        config["max_steps"] = spec.steps
    else:
        config["num_train_epochs"] = float(spec.epochs)
    callbacks = [progress.callback()] if progress else []
    if spec.early_stop_patience and keep_best:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=spec.early_stop_patience))
    return SFTTrainer(
        model=model, tokenizer=processor, data_collator=collator,
        train_dataset=train_ds, eval_dataset=val_ds if has_eval else None,
        args=SFTConfig(**config), callbacks=callbacks,
    )


def train_run(spec: TrainSpec, cfg: EngineConfig, train_ds: Any, *, val_ds: Any = None,
              save: bool = False, resume: bool = False) -> tuple[dict[str, Any], Any, Any, Any]:
    """Fresh weights, audited collator, training, and a metrics record. Caller releases the model."""
    import torch

    spec.validate()
    out_dir = Path(cfg.root) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = processor = trainer = collator = None
    try:
        torch.cuda.reset_peak_memory_stats()
        model, processor = load_lora(cfg, spec.rank)
        params = trainable_report(model)
        collator = make_collator(model, processor, cfg)
        mask = probe_collator(collator, processor, train_ds)
        progress = LiveProgress(spec.name, out_dir / "train_history.jsonl", spec.log_every)
        trainer = make_trainer(model, processor, collator, train_ds, spec, cfg, out_dir,
                               val_ds=val_ds, save=save, progress=progress)
        checkpoint = None
        if resume and (out_dir / "checkpoints").is_dir():
            from transformers.trainer_utils import get_last_checkpoint
            checkpoint = get_last_checkpoint(str(out_dir / "checkpoints"))
        started = time.perf_counter()
        stats = trainer.train(resume_from_checkpoint=checkpoint)
        seconds = time.perf_counter() - started
        history = [json.loads(line) for line in (out_dir / "train_history.jsonl").read_text().splitlines() if line.strip()]
        losses = [row["loss"] for row in history if "loss" in row]
        steady = progress.step_seconds[1:] or progress.step_seconds
        step_seconds = statistics.median(steady) if steady else float("nan")
        eval_loss = None
        if val_ds is not None:
            eval_loss = float(trainer.evaluate(eval_dataset=val_ds)["eval_loss"])
        if save:
            trainer.save_model(str(out_dir / "adapter"))
            processor.save_pretrained(str(out_dir / "adapter"))
        result = {
            "name": spec.name, **asdict(spec), "effective_batch": spec.effective_batch,
            "resumed_from": checkpoint, "steps_done": int(stats.global_step),
            "training_loss": float(stats.training_loss), "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None, "val_loss": eval_loss,
            "stable": bool(losses) and all(math.isfinite(x) for x in losses),
            "train_seconds": round(seconds, 2), "median_step_seconds": round(step_seconds, 3),
            "examples_per_second": round(spec.effective_batch / step_seconds, 3) if step_seconds == step_seconds else None,
            "gpu_utilization_mean": round(statistics.mean(progress.utilization), 1) if progress.utilization else None,
            "mask_audit": mask, "parameters": params, **gpu_memory(),
        }
        (out_dir / "train_summary.json").write_text(json.dumps(result, indent=2))
        return result, model, processor, trainer
    except BaseException:
        # Drop this frame's references before re-raising so an OOM does not pin the model in memory.
        model = processor = trainer = collator = None
        free_gpu()
        raise


# --------------------------------------------------------------------------- evaluation


def evaluate_condition(model: Any, processor: Any, split: Any, indices: Sequence[int], cfg: EngineConfig,
                       *, desc: str, user_prompt: str, system_prompt: str | None = None,
                       answer_format: str = "cord", output_schema: Mapping[str, Any] | None = None,
                       output_dir: Path | None = None, print_every: int = 10) -> dict[str, Any]:
    """Greedy generation for one prompt condition; resumable from predictions.jsonl.

    answer_format "cord": CORD-keyed JSON, graded by the baseline scorer and, after the same
    conversion applied to gold, by the expense grader. "expense": the receipt-to-expense skill's
    format, graded by the expense grader with schema validation as the gate.
    """
    import torch
    from jsonschema import Draft202012Validator
    from unsloth import FastVisionModel

    _require(answer_format in ("cord", "expense"), "answer_format must be 'cord' or 'expense'")
    _require(answer_format == "cord" or output_schema is not None, "expense grading needs the output schema")
    validator = Draft202012Validator(output_schema) if output_schema is not None else None
    FastVisionModel.for_inference(model)
    path = None
    completed: dict[int, dict[str, Any]] = {}
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "predictions.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    completed[int(record["index"])] = record
    torch.cuda.reset_peak_memory_stats()
    records = []
    started = time.perf_counter()
    for position, index in enumerate(indices, start=1):
        index = int(index)
        if index in completed:
            records.append(completed[index])
            continue
        row = split[index]
        gold = gold_parse(row)
        image = prepare_image(row["image"], cfg.image_long_side)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, prompt_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        input_length = int(inputs["input_ids"].shape[-1])
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=cfg.max_new_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        ids = out[0][input_length:]
        raw = processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        parsed = parse_json_object(raw)
        gold_receipt = cord_to_expense(gold)
        record = {"index": index, "reference": gold, "prediction": parsed, "raw_prediction": raw,
                  "valid_json": parsed is not None, "elapsed_s": round(elapsed, 3),
                  "input_tokens": input_length, "output_tokens": int(ids.numel()),
                  "hit_max_new_tokens": int(ids.numel()) >= cfg.max_new_tokens}
        if answer_format == "cord":
            record["score"] = score_example(parsed, gold)
            if validator is not None:
                converted = {"receipts": [cord_to_expense(parsed)]} if parsed is not None else None
                record["expense"] = score_expense(converted, gold_receipt, validator)
        else:
            record["expense"] = score_expense(parsed, gold_receipt, validator)
        records.append(record)
        if path:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if position % print_every == 0 or position == len(indices):
            wall = time.perf_counter() - started
            now = summarize(records) if answer_format == "cord" else summarize_expense(records)
            print(f"[{desc}] {position}/{len(indices)} receipts | F1 {now['field_f1']:.3f} | "
                  f"valid JSON {now['valid_json_fraction']:.0%} | elapsed {_clock(wall)} | "
                  f"ETA {_clock(wall / position * (len(indices) - position))}", flush=True)
    if answer_format == "cord":
        summary = summarize(records)
        if validator is not None and all("expense" in r for r in records):
            summary["expense_format"] = summarize_expense(records, schema_measured=False)
    else:
        summary = summarize_expense(records)
    summary["truncated_outputs"] = sum(r.get("hit_max_new_tokens", False) for r in records)
    summary["peak_allocated_gib"] = gpu_memory()["peak_allocated_gib"]
    if output_dir:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    FastVisionModel.for_training(model)
    return {"summary": summary, "records": records}


def evaluate_receipts(model: Any, processor: Any, split: Any, indices: Sequence[int], cfg: EngineConfig,
                      *, desc: str, output_dir: Path | None = None, print_every: int = 10) -> dict[str, Any]:
    """The baseline's short prompt, CORD format, baseline scorer."""
    return evaluate_condition(model, processor, split, indices, cfg, desc=desc, user_prompt=cfg.prompt,
                              answer_format="cord", output_dir=output_dir, print_every=print_every)


def compare(report: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    keys = [k for k in BASELINE_BF16 if k in report and report[k] is not None and reference.get(k) is not None]
    return {k: {"run": report[k], "reference": reference[k], "delta": round(report[k] - reference[k], 6)} for k in keys}


def persist_to_hub(root: Path, repo_id: str, token: str | None, run_id: str, patterns: Sequence[str],
                   message: str) -> str | None:
    """Copy finished results off the Colab machine, which loses /content when the runtime ends."""
    if not token:
        print(f"Not saved to the Hub (no token): {message}")
        return None
    from huggingface_hub import HfApi

    destination = f"cord-runs/{run_id}"
    HfApi(token=token).upload_folder(
        repo_id=repo_id, folder_path=str(root), path_in_repo=destination,
        allow_patterns=list(patterns), ignore_patterns=["**/checkpoints/**", "*/checkpoints/*"],
        commit_message=f"DAPTA {run_id}: {message}",
    )
    print(f"Saved to https://huggingface.co/{repo_id}/tree/main/{destination} ({message})", flush=True)
    return destination


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    return path
