from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from PIL import Image

from .data import REQUIRED_SPLITS, read_jsonl, sha256, validate_dataset


VALIDATOR_VERSION = "receipt-validator-v1"


class DatasetValidationError(ValueError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(f"receipt dataset failed {len(report['errors'])} validation checks")


def validate_receipt_dataset(
    dataset_dir: Path,
    *,
    contract_dir: Path,
    source_manifest: Path,
    min_image_edge: int = 32,
) -> dict[str, Any]:
    from .receipts import PREPROCESSING_VERSION, load_receipt_contract, load_receipt_sources

    dataset_dir = dataset_dir.resolve()
    contract_dir = contract_dir.resolve()
    source_manifest = source_manifest.resolve()
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    checked = {
        "rows": 0,
        "images": 0,
        "schema_instances": 0,
        "scalar_arithmetic": 0,
        "line_item_arithmetic": 0,
    }

    def error(check: str, location: str, message: str) -> None:
        errors.append({"check": check, "location": location, "message": message})

    try:
        counts = validate_dataset(dataset_dir)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        error("dataset_integrity", "dataset", str(exc))

    contract: dict[str, Any] | None = None
    sources = None
    try:
        contract = load_receipt_contract(contract_dir)
        Draft202012Validator.check_schema(contract["output_schema"])
    except Exception as exc:
        error("contract", "output_schema.json", str(exc))

    try:
        sources = load_receipt_sources(source_manifest)
    except Exception as exc:
        error("source_manifest", source_manifest.name, str(exc))

    manifest: dict[str, Any] | None = None
    manifest_path = dataset_dir / "manifest.json"
    try:
        manifest = _read_object(manifest_path)
        _verify_manifest(
            manifest,
            dataset_dir=dataset_dir,
            contract_dir=contract_dir,
            source_manifest=source_manifest,
            sources=sources,
            preprocessing_version=PREPROCESSING_VERSION,
        )
    except Exception as exc:
        error("reproducibility_manifest", "manifest.json", str(exc))

    validator = Draft202012Validator(contract["output_schema"]) if contract else None
    if not errors or counts:
        for split in REQUIRED_SPLITS:
            path = dataset_dir / f"{split}.jsonl"
            if not path.is_file():
                continue
            try:
                rows = read_jsonl(path)
            except Exception as exc:
                error("dataset_integrity", split, str(exc))
                continue
            for index, row in enumerate(rows, start=1):
                location = f"{split}:{index}"
                checked["rows"] += 1
                completion = row.get("completion")
                if isinstance(completion, str):
                    try:
                        completion = json.loads(completion)
                    except json.JSONDecodeError as exc:
                        error("output_schema", location, f"completion is not JSON: {exc.msg}")
                        continue
                if validator:
                    schema_errors = sorted(
                        validator.iter_errors(completion),
                        key=lambda item: "/".join(str(part) for part in item.absolute_path),
                    )
                    if schema_errors:
                        for item in schema_errors:
                            field = ".".join(str(part) for part in item.absolute_path) or "$"
                            error("output_schema", f"{location}:{field}", item.message)
                    else:
                        checked["schema_instances"] += 1

                images = row.get("images", [])
                receipts = completion.get("receipts", []) if isinstance(completion, dict) else []
                if len(receipts) != len(images):
                    error(
                        "receipt_cardinality",
                        location,
                        f"{len(receipts)} receipts for {len(images)} images",
                    )
                for image_index, relative in enumerate(images):
                    image_path = (dataset_dir / relative).resolve()
                    try:
                        with Image.open(image_path) as image:
                            image.verify()
                        with Image.open(image_path) as image:
                            width, height = image.size
                        if min(width, height) < min_image_edge:
                            raise ValueError(
                                f"image is {width}x{height}; minimum edge is {min_image_edge}"
                            )
                        checked["images"] += 1
                    except Exception as exc:
                        error("image_decode", f"{location}:image:{image_index}", str(exc))

                for receipt_index, receipt in enumerate(receipts):
                    if not isinstance(receipt, dict):
                        continue
                    receipt_location = f"{location}:receipt:{receipt_index}"
                    source_id = row.get("metadata", {}).get("source_id")
                    _check_arithmetic(
                        receipt,
                        receipt_location,
                        checked,
                        error,
                        warnings,
                        null_components_are_zero=source_id == "receipt-cord-v2",
                    )

    hashes: dict[str, Any] = {}
    if manifest_path.is_file():
        hashes["manifest_sha256"] = sha256(manifest_path)
    hashes["source_manifest_sha256"] = sha256(source_manifest) if source_manifest.is_file() else None
    hashes["splits"] = {
        split: sha256(dataset_dir / f"{split}.jsonl")
        for split in REQUIRED_SPLITS
        if (dataset_dir / f"{split}.jsonl").is_file()
    }
    return {
        "schema_version": 1,
        "validator_version": VALIDATOR_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "passed": not errors,
        "counts": counts,
        "checked": checked,
        "hashes": hashes,
        "errors": errors,
        "warnings": warnings,
    }


def require_valid_receipt_dataset(
    dataset_dir: Path,
    *,
    contract_dir: Path,
    source_manifest: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    report = validate_receipt_dataset(
        dataset_dir,
        contract_dir=contract_dir,
        source_manifest=source_manifest,
    )
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise DatasetValidationError(report)
    return report


def _verify_manifest(
    manifest: dict[str, Any],
    *,
    dataset_dir: Path,
    contract_dir: Path,
    source_manifest: Path,
    sources: Any,
    preprocessing_version: str,
) -> None:
    if manifest.get("preprocessing_version") != preprocessing_version:
        raise ValueError("preprocessing version mismatch")
    if manifest.get("source_manifest_sha256") != sha256(source_manifest):
        raise ValueError("source manifest hash mismatch")
    for name in ("skill_spec.json", "output_schema.json", "evaluation_plan.json"):
        if manifest.get("contract_sha256", {}).get(name) != sha256(contract_dir / name):
            raise ValueError(f"contract hash mismatch: {name}")
    for split in REQUIRED_SPLITS:
        split_hash = sha256(dataset_dir / f"{split}.jsonl")
        if manifest.get("split_sha256", {}).get(split) != split_hash:
            raise ValueError(f"split hash mismatch: {split}")
        if manifest.get("counts", {}).get(split) != len(read_jsonl(dataset_dir / f"{split}.jsonl")):
            raise ValueError(f"split count mismatch: {split}")
    test_hash = sha256(dataset_dir / "test.jsonl")
    if manifest.get("frozen_test_sha256") != test_hash:
        raise ValueError("frozen TEST hash mismatch")
    if sources is not None:
        expected = {
            source.id: {"repository": source.source, "revision": source.revision}
            for source in sources.values()
        }
        if manifest.get("sources") != expected:
            raise ValueError("pinned source revisions mismatch")


def _check_arithmetic(
    receipt: dict[str, Any],
    location: str,
    checked: dict[str, int],
    error: Any,
    warnings: list[dict[str, str]],
    *,
    null_components_are_zero: bool,
) -> None:
    tolerance = Decimal("0.02")
    scalars = [receipt.get(name) for name in ("subtotal", "tax", "service_charge", "total")]
    scalar_checkable = all(_finite_number(value) for value in scalars)
    if null_components_are_zero:
        scalar_checkable = (
            _finite_number(scalars[0])
            and _finite_number(scalars[3])
            and all(value is None or _finite_number(value) for value in scalars[1:3])
        )
    if scalar_checkable:
        checked["scalar_arithmetic"] += 1
        subtotal, tax, service, total = (
            Decimal(0) if value is None else Decimal(str(value)) for value in scalars
        )
        difference = abs(subtotal + tax + service - total)
        if difference > tolerance:
            error("scalar_arithmetic", location, f"subtotal + tax + service differs by {difference}")

    items = receipt.get("line_items")
    subtotal = receipt.get("subtotal")
    if isinstance(items, list) and items and _finite_number(subtotal):
        amounts = [item.get("amount") for item in items if isinstance(item, dict)]
        if len(amounts) == len(items) and all(_finite_number(value) for value in amounts):
            checked["line_item_arithmetic"] += 1
            difference = abs(sum((Decimal(str(value)) for value in amounts), Decimal()) - Decimal(str(subtotal)))
            if difference > tolerance:
                error("line_item_arithmetic", location, f"line item sum differs by {difference}")

    if any(_finite_number(value) and Decimal(str(value)) < 0 for value in scalars):
        warnings.append({
            "check": "negative_amount",
            "location": location,
            "message": "negative receipt amount retained for possible returns or refunds",
        })


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("research/adapter-assets/skills/receipt-to-expense"),
    )
    parser.add_argument(
        "--sources", type=Path, default=Path("research/adapter-assets/sources.lock.json")
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = require_valid_receipt_dataset(
        args.dataset,
        contract_dir=args.contract,
        source_manifest=args.sources,
        report_path=args.report,
    )
    print(json.dumps({"passed": report["passed"], "report": str(args.report)}))


if __name__ == "__main__":
    main()
