from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .data import REQUIRED_SPLITS, freeze_test, sha256, validate_dataset


PREPROCESSING_VERSION = "receipt-cord-sroie-v1"
SYSTEM_PROMPT = "Extract visible receipt evidence and return only JSON matching the receipt expense schema."
USER_PROMPT = "Convert this receipt image into the normalized expense report. Use null for absent fields."


@dataclass(frozen=True)
class SourceSpec:
    id: str
    source: str
    revision: str


def load_receipt_contract(contract_dir: Path) -> dict[str, Any]:
    contract_dir = contract_dir.resolve()
    artifacts = {
        name: _read_object(contract_dir / f"{name}.json")
        for name in ("skill_spec", "output_schema", "evaluation_plan", "review")
    }
    if artifacts["review"].get("status") != "approved_for_implementation":
        raise ValueError("receipt contract is not approved for implementation")
    schema = artifacts["output_schema"]
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("receipt output schema must be a closed JSON object")
    if artifacts["evaluation_plan"].get("model_based_judging") is not False:
        raise ValueError("receipt objective evaluation must be deterministic")

    expected = artifacts["review"].get("artifact_sha256", {})
    for name in ("skill_spec", "output_schema", "evaluation_plan"):
        digest = _canonical_sha256(artifacts[name])
        if expected.get(f"{name}.json") != digest:
            raise ValueError(f"receipt contract hash mismatch: {name}.json")
    return artifacts


def load_receipt_sources(manifest_path: Path) -> dict[str, SourceSpec]:
    manifest = _read_object(manifest_path)
    if manifest.get("data_residency") != "remote-only":
        raise ValueError("receipt sources must use remote-only data residency")
    if manifest.get("local_dataset_contents_allowed") is not False:
        raise ValueError("local dataset contents must remain disabled")

    wanted = {"receipt-cord-v2", "receipt-sroie"}
    sources = {
        item["id"]: SourceSpec(item["id"], item["source"], item["revision"])
        for item in manifest.get("sources", [])
        if item.get("id") in wanted
    }
    if set(sources) != wanted:
        raise ValueError("source manifest must pin both CORD v2 and SROIE")
    for source in sources.values():
        if not re.fullmatch(r"[0-9a-f]{40}", source.revision):
            raise ValueError(f"source revision is not an immutable commit: {source.id}")
    return sources


def materialize_receipt_dataset(
    job_dir: Path,
    *,
    source_manifest: Path,
    contract_dir: Path,
    loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
    max_rows_per_source_split: int | None = None,
    allow_local_test: bool = False,
) -> Path:
    if not allow_local_test and os.environ.get("DAPTA_REMOTE_WORKER") != "1":
        raise RuntimeError("receipt datasets may only be materialized on a Dapta remote worker")
    if max_rows_per_source_split is not None and max_rows_per_source_split < 1:
        raise ValueError("max_rows_per_source_split must be positive")

    contract_dir = contract_dir.resolve()
    source_manifest = source_manifest.resolve()
    load_receipt_contract(contract_dir)
    sources = load_receipt_sources(source_manifest)

    job_dir = job_dir.resolve()
    dataset_dir = job_dir / "receipt-dataset"
    cache_dir = job_dir / "source-cache"
    if dataset_dir.exists():
        raise FileExistsError(dataset_dir)
    dataset_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    rows: dict[str, list[dict[str, Any]]] = {split: [] for split in REQUIRED_SPLITS}

    if loader is None:
        from datasets import load_dataset

        loader = load_dataset

    try:
        cord = sources["receipt-cord-v2"]
        for source_split, target_split in (
            ("train", "train"),
            ("validation", "dev"),
            ("test", "test"),
        ):
            source_rows = _load_rows(
                loader, cord, source_split, cache_dir, max_rows_per_source_split
            )
            for index, row in enumerate(source_rows):
                record = _cord_record(row, source_split, index)
                rows[target_split].append(
                    _write_record(dataset_dir, cord, source_split, target_split, row, record)
                )

        sroie = sources["receipt-sroie"]
        source_rows = _load_rows(loader, sroie, "train", cache_dir, max_rows_per_source_split)
        for index, row in enumerate(source_rows):
            target_split = _sroie_train_split(str(row.get("key", index)))
            record = _sroie_record(row)
            rows[target_split].append(
                _write_record(dataset_dir, sroie, "train", target_split, row, record)
            )

        source_rows = _load_rows(loader, sroie, "test", cache_dir, max_rows_per_source_split)
        for index, row in enumerate(source_rows):
            record = _sroie_record(row)
            rows["test"].append(
                _write_record(dataset_dir, sroie, "test", "test", row, record)
            )

        for split in REQUIRED_SPLITS:
            _write_jsonl(dataset_dir / f"{split}.jsonl", rows[split])
        frozen_test_sha = freeze_test(dataset_dir)
        counts = validate_dataset(dataset_dir)
        manifest = {
            "preprocessing_version": PREPROCESSING_VERSION,
            "source_manifest_sha256": sha256(source_manifest),
            "contract_sha256": {
                name: sha256(contract_dir / name)
                for name in ("skill_spec.json", "output_schema.json", "evaluation_plan.json")
            },
            "sources": {
                source.id: {"repository": source.source, "revision": source.revision}
                for source in sources.values()
            },
            "counts": counts,
            "split_sha256": {
                split: sha256(dataset_dir / f"{split}.jsonl") for split in REQUIRED_SPLITS
            },
            "frozen_test_sha256": frozen_test_sha,
        }
        (dataset_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    return dataset_dir


def _load_rows(
    loader: Callable[..., Iterable[Mapping[str, Any]]],
    source: SourceSpec,
    split: str,
    cache_dir: Path,
    limit: int | None,
) -> Iterable[Mapping[str, Any]]:
    rows = loader(
        source.source,
        revision=source.revision,
        split=split,
        streaming=True,
        cache_dir=str(cache_dir),
        token=os.environ.get("HF_TOKEN"),
    )
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        yield row


def _write_record(
    dataset_dir: Path,
    source: SourceSpec,
    source_split: str,
    target_split: str,
    row: Mapping[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    key = _row_key(source.id, source_split, row)
    relative_image = Path("images") / source.id / source_split / f"{_safe_name(key)}.png"
    image_path = dataset_dir / relative_image
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = row.get("image")
    if image is None or not hasattr(image, "save"):
        raise ValueError(f"{source.id}/{source_split}/{key} is missing a decodable image")
    image.convert("RGB").save(image_path, format="PNG")
    return {
        "id": f"{source.id}:{source_split}:{key}",
        "images": [relative_image.as_posix()],
        "system": SYSTEM_PROMPT,
        "prompt": USER_PROMPT,
        "completion": {"receipts": [receipt]},
        "metadata": {
            "source_id": source.id,
            "source_revision": source.revision,
            "source_split": source_split,
            "target_split": target_split,
        },
    }


def _cord_record(row: Mapping[str, Any], split: str, index: int) -> dict[str, Any]:
    raw = row.get("ground_truth")
    try:
        annotation = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        parsed = annotation["gt_parse"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid CORD annotation at {split}:{index}") from error

    menu = parsed.get("menu", [])
    if isinstance(menu, Mapping):
        menu = [menu]
    line_items = [
        {
            "description": _text(item.get("nm")),
            "quantity": _number(item.get("cnt")),
            "amount": _number(item.get("price")),
        }
        for item in menu
        if isinstance(item, Mapping)
    ]
    subtotal = parsed.get("sub_total", {})
    if not isinstance(subtotal, Mapping):
        subtotal = {}
    total = parsed.get("total", {})
    if not isinstance(total, Mapping):
        total = {}
    return _receipt(
        subtotal=_number(subtotal.get("subtotal_price")),
        tax=_number(subtotal.get("tax_price")),
        service_charge=_number(subtotal.get("service_price")),
        total=_number(total.get("total_price")),
        line_items=line_items,
    )


def _sroie_record(row: Mapping[str, Any]) -> dict[str, Any]:
    entities = row.get("entities")
    if not isinstance(entities, Mapping):
        raise ValueError("invalid SROIE entities")
    return _receipt(
        merchant=_text(entities.get("company")),
        address=_text(entities.get("address")),
        transaction_date=_date(entities.get("date")),
        total=_number(entities.get("total")),
    )


def _receipt(
    *,
    merchant: str | None = None,
    address: str | None = None,
    transaction_date: str | None = None,
    subtotal: int | float | None = None,
    tax: int | float | None = None,
    service_charge: int | float | None = None,
    total: int | float | None = None,
    line_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "merchant": merchant,
        "address": address,
        "transaction_date": transaction_date,
        "currency": None,
        "subtotal": subtotal,
        "tax": tax,
        "service_charge": service_charge,
        "total": total,
        "line_items": line_items or [],
    }


def _row_key(source_id: str, split: str, row: Mapping[str, Any]) -> str:
    if source_id == "receipt-sroie":
        value = row.get("key")
    else:
        raw = row.get("ground_truth")
        annotation = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        value = annotation.get("meta", {}).get("image_id")
    if value is None or str(value).strip() == "":
        payload = json.dumps({key: value for key, value in row.items() if key != "image"}, sort_keys=True)
        value = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return str(value)


def _sroie_train_split(key: str) -> str:
    digest = hashlib.sha256(f"20261017:{key}".encode()).digest()
    return "dev" if int.from_bytes(digest[:8], "big") % 5 == 0 else "train"


def _number(value: Any) -> int | float | None:
    if value is None:
        return None
    text = re.sub(r"[^0-9,.-]", "", str(value).strip())
    if not text or text in {"-", ".", ","}:
        return None
    if "," in text and "." in text:
        if text.rfind(".") > text.rfind(","):
            text = text.replace(",", "")
        else:
            text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+", text):
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")
    elif text.count(".") > 1 and re.fullmatch(r"-?\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"invalid numeric receipt label: {value}") from error
    return int(number) if number == number.to_integral() else float(number)


def _date(value: Any) -> str | None:
    text = _text(value)
    if text is None:
        return None
    for pattern in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "-", value).strip(".-")
    return name[:120] or hashlib.sha256(value.encode()).hexdigest()[:16]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty derived split: {path.stem}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument(
        "--sources", type=Path, default=Path("research/adapter-assets/sources.lock.json")
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("research/adapter-assets/skills/receipt-to-expense"),
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(
        materialize_receipt_dataset(
            args.job_dir,
            source_manifest=args.sources,
            contract_dir=args.contract,
            max_rows_per_source_split=args.limit,
        )
    )


if __name__ == "__main__":
    main()
