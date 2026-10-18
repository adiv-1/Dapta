from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from .data import read_jsonl, validate_dataset


AMOUNT_TOLERANCE = Decimal("0.01")
ARITHMETIC_TOLERANCE = Decimal("0.02")
FORBIDDEN_KEYS = {
    "approval",
    "approved",
    "category",
    "eligibility",
    "eligible",
    "policy",
    "reimbursable",
    "reimbursement",
}


def evaluate_receipts(
    dataset_dir: Path,
    predictions_path: Path,
    *,
    contract_dir: Path,
    split: str = "test",
) -> dict[str, Any]:
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"unsupported split: {split}")
    validate_dataset(dataset_dir, require_frozen_test=split == "test")
    gold_rows = {row["id"]: row for row in read_jsonl(dataset_dir / f"{split}.jsonl")}
    schema = _read_object(contract_dir / "output_schema.json")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    predictions = read_jsonl(predictions_path)
    if not predictions:
        raise ValueError("predictions file is empty")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for index, prediction in enumerate(predictions, start=1):
        condition = _nonempty_string(prediction.get("condition"), f"prediction {index} condition")
        case_id = _nonempty_string(prediction.get("id"), f"prediction {index} id")
        if case_id not in gold_rows:
            raise ValueError(f"prediction references unknown case: {case_id}")
        if case_id in grouped[condition]:
            raise ValueError(f"duplicate prediction for {condition}/{case_id}")
        grouped[condition][case_id] = prediction

    cases: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for condition in sorted(grouped):
        missing = sorted(set(gold_rows) - set(grouped[condition]))
        if missing:
            raise ValueError(f"{condition} is missing {len(missing)} predictions")
        condition_cases = [
            _score_case(gold_rows[case_id], grouped[condition][case_id], validator)
            for case_id in sorted(gold_rows)
        ]
        cases.extend(condition_cases)
        summaries[condition] = _summarize(condition_cases)

    return {
        "schema_version": 1,
        "evaluator": "receipt-evaluator-v1",
        "split": split,
        "cases_per_condition": len(gold_rows),
        "conditions": summaries,
        "performance_summary": _performance_summary(summaries),
        "cases": cases,
    }


def _score_case(
    gold_row: dict[str, Any],
    prediction: dict[str, Any],
    validator: Draft202012Validator,
) -> dict[str, Any]:
    output, parse_error = _parse_output(prediction.get("output"))
    schema_errors = []
    if parse_error is None:
        schema_errors = [item.message for item in validator.iter_errors(output)]
    schema_valid = parse_error is None and not schema_errors

    gold = gold_row["completion"]
    if isinstance(gold, str):
        gold = json.loads(gold)
    expected_receipts = gold["receipts"]
    predicted_receipts = output.get("receipts", []) if isinstance(output, dict) else []
    cardinality_valid = schema_valid and len(predicted_receipts) == len(expected_receipts)
    exclusion_valid = cardinality_valid and not _forbidden_keys(output)
    hallucinations: list[str] = []
    if cardinality_valid:
        for receipt_index, (expected, actual) in enumerate(
            zip(expected_receipts, predicted_receipts)
        ):
            for field, expected_value in expected.items():
                if field != "line_items" and expected_value is None and actual.get(field) is not None:
                    hallucinations.append(f"receipts[{receipt_index}].{field}")
    exclusion_valid = exclusion_valid and not hallucinations
    gate_passed = schema_valid and cardinality_valid and exclusion_valid

    field_correct = 0
    field_total = 0
    line_true_positive = 0
    line_predicted = 0
    line_expected = 0
    arithmetic_passed = 0
    arithmetic_total = 0
    for receipt_index, expected in enumerate(expected_receipts):
        actual = predicted_receipts[receipt_index] if cardinality_valid else {}
        if isinstance(actual, dict):
            source_id = gold_row.get("metadata", {}).get("source_id")
            fields = (
                ("merchant", "address", "transaction_date", "total")
                if source_id == "receipt-sroie"
                else ("subtotal", "tax", "service_charge", "total")
            )
            for field in fields:
                if expected.get(field) is None:
                    continue
                field_total += 1
                if gate_passed and _field_equal(field, expected.get(field), actual.get(field)):
                    field_correct += 1

            if source_id == "receipt-cord-v2":
                expected_items = expected.get("line_items", [])
                actual_items = actual.get("line_items", []) if gate_passed else []
                line_expected += len(expected_items)
                line_predicted += len(actual_items)
                line_true_positive += _matching_line_items(expected_items, actual_items)

            for check in _arithmetic_checks(expected):
                arithmetic_total += 1
                if gate_passed and check(actual):
                    arithmetic_passed += 1

    accuracy = field_correct / field_total if field_total else 0.0
    return {
        "id": gold_row["id"],
        "condition": prediction["condition"],
        "source_id": gold_row.get("metadata", {}).get("source_id"),
        "gates": {
            "schema_valid": schema_valid,
            "cardinality_valid": cardinality_valid,
            "exclusion_valid": exclusion_valid,
            "passed": gate_passed,
        },
        "errors": {
            "parse": parse_error,
            "schema": schema_errors,
            "hallucinated_fields": hallucinations,
        },
        "metrics": {
            "field_correct": field_correct,
            "field_total": field_total,
            "field_accuracy": accuracy,
            "line_item_true_positive": line_true_positive,
            "line_item_predicted": line_predicted,
            "line_item_expected": line_expected,
            "arithmetic_passed": arithmetic_passed,
            "arithmetic_total": arithmetic_total,
        },
        "usage": _usage(prediction),
    }


def _summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    field_correct = sum(case["metrics"]["field_correct"] for case in cases)
    field_total = sum(case["metrics"]["field_total"] for case in cases)
    true_positive = sum(case["metrics"]["line_item_true_positive"] for case in cases)
    predicted = sum(case["metrics"]["line_item_predicted"] for case in cases)
    expected = sum(case["metrics"]["line_item_expected"] for case in cases)
    arithmetic_passed = sum(case["metrics"]["arithmetic_passed"] for case in cases)
    arithmetic_total = sum(case["metrics"]["arithmetic_total"] for case in cases)
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / expected if expected else 0.0
    line_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    latencies = [case["usage"]["latency_ms"] for case in cases if case["usage"]["latency_ms"] is not None]
    cost = sum((Decimal(case["usage"]["cost_usd"] or "0") for case in cases), Decimal())
    input_tokens = sum(case["usage"]["input_tokens"] or 0 for case in cases)
    output_tokens = sum(case["usage"]["output_tokens"] or 0 for case in cases)
    return {
        "accuracy": field_correct / field_total if field_total else 0.0,
        "field_correct": field_correct,
        "field_total": field_total,
        "schema_pass_rate": sum(case["gates"]["schema_valid"] for case in cases) / len(cases),
        "gate_pass_rate": sum(case["gates"]["passed"] for case in cases) / len(cases),
        "line_item_precision": precision,
        "line_item_recall": recall,
        "line_item_f1": line_f1,
        "arithmetic_accuracy": arithmetic_passed / arithmetic_total if arithmetic_total else None,
        "time": {
            "total_ms": sum(latencies),
            "mean_ms": statistics.fmean(latencies) if latencies else None,
            "p50_ms": statistics.median(latencies) if latencies else None,
        },
        "cost_usd": str(cost),
        "tokens": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
    }


def _performance_summary(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    conditions = sorted(summaries)
    fastest = min(
        (name for name in conditions if summaries[name]["time"]["mean_ms"] is not None),
        key=lambda name: (summaries[name]["time"]["mean_ms"], name),
        default=None,
    )
    cheapest = min(conditions, key=lambda name: (Decimal(summaries[name]["cost_usd"]), name))
    most_accurate = max(conditions, key=lambda name: (summaries[name]["accuracy"], name))
    return {
        "most_accurate": most_accurate,
        "fastest": fastest,
        "lowest_cost": cheapest,
        "best_model": most_accurate,
    }


def _parse_output(value: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(value, dict):
        return value, None
    if not isinstance(value, str):
        return {}, "output must be a JSON object or JSON string"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON: {exc.msg}"
    if not isinstance(parsed, dict):
        return {}, "output JSON must be an object"
    return parsed, None


def _field_equal(field: str, expected: Any, actual: Any) -> bool:
    if field in {"subtotal", "tax", "service_charge", "total"}:
        return _amount_equal(expected, actual)
    if field == "transaction_date":
        return str(expected) == str(actual)
    return _normalize_text(expected) == _normalize_text(actual)


def _matching_line_items(expected: list[Any], actual: list[Any]) -> int:
    unused = set(range(len(actual)))
    matches = 0
    for expected_item in expected:
        for index in sorted(unused):
            actual_item = actual[index]
            if not isinstance(expected_item, dict) or not isinstance(actual_item, dict):
                continue
            if _normalize_text(expected_item.get("description")) != _normalize_text(
                actual_item.get("description")
            ):
                continue
            if not _amount_equal(expected_item.get("amount"), actual_item.get("amount")):
                continue
            expected_quantity = expected_item.get("quantity")
            if expected_quantity is not None and not _amount_equal(
                expected_quantity, actual_item.get("quantity")
            ):
                continue
            unused.remove(index)
            matches += 1
            break
    return matches


def _arithmetic_checks(expected: dict[str, Any]) -> Iterable[Any]:
    scalars = [expected.get(field) for field in ("subtotal", "tax", "service_charge", "total")]
    if scalars[0] is not None and scalars[3] is not None:
        yield lambda actual: _scalar_arithmetic(actual)
    items = expected.get("line_items", [])
    if items and expected.get("subtotal") is not None and all(item.get("amount") is not None for item in items):
        yield lambda actual: _line_arithmetic(actual)


def _scalar_arithmetic(receipt: dict[str, Any]) -> bool:
    values = [receipt.get(field) for field in ("subtotal", "tax", "service_charge", "total")]
    if values[0] is None or values[3] is None or any(not _finite(value) for value in values if value is not None):
        return False
    subtotal, tax, service, total = (Decimal(0) if value is None else Decimal(str(value)) for value in values)
    return abs(subtotal + tax + service - total) <= ARITHMETIC_TOLERANCE


def _line_arithmetic(receipt: dict[str, Any]) -> bool:
    subtotal = receipt.get("subtotal")
    items = receipt.get("line_items")
    if not _finite(subtotal) or not isinstance(items, list) or not items:
        return False
    amounts = [item.get("amount") for item in items if isinstance(item, dict)]
    if len(amounts) != len(items) or not all(_finite(value) for value in amounts):
        return False
    return abs(sum((Decimal(str(value)) for value in amounts), Decimal()) - Decimal(str(subtotal))) <= ARITHMETIC_TOLERANCE


def _amount_equal(expected: Any, actual: Any) -> bool:
    if not _finite(expected) or not _finite(actual):
        return False
    return abs(Decimal(str(expected)) - Decimal(str(actual))) <= AMOUNT_TOLERANCE


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys(child))
    return found


def _usage(prediction: dict[str, Any]) -> dict[str, Any]:
    usage = prediction.get("usage", {})
    if not isinstance(usage, dict):
        raise ValueError("prediction usage must be an object")
    latency = usage.get("latency_ms")
    cost = usage.get("cost_usd")
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if latency is not None and (not _finite(latency) or latency < 0):
        raise ValueError("latency_ms must be a non-negative number")
    if cost is not None and Decimal(str(cost)) < 0:
        raise ValueError("cost_usd must be non-negative")
    for name, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise ValueError(f"{name} must be a non-negative integer")
    return {
        "latency_ms": latency,
        "cost_usd": None if cost is None else str(Decimal(str(cost))),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("research/adapter-assets/skills/receipt-to-expense"),
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_receipts(
        args.dataset,
        args.predictions,
        contract_dir=args.contract,
        split=args.split,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["performance_summary"], sort_keys=True))


if __name__ == "__main__":
    main()
