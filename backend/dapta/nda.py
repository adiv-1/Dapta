"""ContractNLI NDA triage: pinned source, document-level dataset, validation and training rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .data import REQUIRED_SPLITS, freeze_test, read_jsonl, sha256


SKILL_ID = "nda-triage"
DEFAULT_CONTRACT_DIR = Path("research/adapter-assets/skills/nda-triage")
PREPROCESSING_VERSION = "contractnli-document-v1"
VALIDATOR_VERSION = "nda-validator-v1"
CHOICES = ("Entailment", "Contradiction", "NotMentioned")
CONTRACT_NAMES = ("skill_spec", "output_schema", "evaluation_plan")
ROW_KEYS = {"id", "source_id", "document_type", "text", "labels"}
APPROVED = "approved_for_implementation"


class NdaValidationError(ValueError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(f"NDA dataset failed {len(report['errors'])} validation checks")


def load_nda_contract(contract_dir: Path = DEFAULT_CONTRACT_DIR) -> dict[str, Any]:
    contract_dir = contract_dir.resolve()
    artifacts = {
        name: _read_object(contract_dir / f"{name}.json")
        for name in (*CONTRACT_NAMES, "review", "source.lock")
    }
    artifacts["source"] = artifacts.pop("source.lock")
    if artifacts["review"].get("status") != APPROVED:
        raise ValueError("NDA contract is not approved for implementation")
    expected = artifacts["review"].get("artifact_sha256", {})
    for name in CONTRACT_NAMES:
        if expected.get(f"{name}.json") != _canonical_sha256(artifacts[name]):
            raise ValueError(f"NDA contract hash mismatch: {name}.json")

    ids = [item["id"] for item in artifacts["skill_spec"]["hypotheses"]]
    schema = artifacts["output_schema"]
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("NDA output schema must be a closed JSON object")
    if schema.get("required") != ids or list(schema.get("properties", {})) != ids:
        raise ValueError("NDA output schema must require exactly the contract hypotheses, in order")
    if any(schema["properties"][key].get("enum") != list(CHOICES) for key in ids):
        raise ValueError("every NDA hypothesis must use the three ContractNLI labels")
    if artifacts["evaluation_plan"].get("model_based_judging") is not False:
        raise ValueError("NDA objective evaluation must be deterministic")
    artifacts["hypothesis_ids"] = ids
    return artifacts


def fetch_source(cache_dir: Path, source: Mapping[str, Any]) -> Path:
    """Return a verified copy of the pinned archive, downloading it once into cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{source['sha256']}.zip"
    if target.is_file() and sha256(target) == source["sha256"]:
        return target
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".part", delete=False) as handle:
        partial = Path(handle.name)
        request = urllib.request.Request(source["source"], headers={"User-Agent": "dapta-nda/1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            shutil.copyfileobj(response, handle)
    try:
        _verify_archive(partial, source)
    except ValueError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(target)
    return target


def materialize_nda_dataset(
    out_dir: Path,
    *,
    contract_dir: Path = DEFAULT_CONTRACT_DIR,
    source_zip: Path | None = None,
    cache_dir: Path | None = None,
    limit: int | None = None,
    source_lock: Mapping[str, Any] | None = None,
) -> Path:
    """Write train/dev/test.jsonl with one row per NDA and all 17 labels; TEST is hash-frozen."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    contract = load_nda_contract(contract_dir)
    source = dict(source_lock or contract["source"])
    ids = contract["hypothesis_ids"]
    if source_zip is None:
        source_zip = fetch_source(cache_dir or out_dir.parent / "source-cache", source)
    _verify_archive(source_zip, source)

    dataset_dir = out_dir.resolve()
    if dataset_dir.exists():
        raise FileExistsError(dataset_dir)
    dataset_dir.mkdir(parents=True)
    counts: dict[str, int] = {}
    with zipfile.ZipFile(source_zip) as archive:
        for split in REQUIRED_SPLITS:
            member = _member(source, split)
            payload = json.loads(archive.read(member))
            if [key for key in _numeric(payload["labels"])] != ids:
                raise ValueError(f"{member} hypotheses differ from the NDA contract")
            rows = []
            for document in sorted(payload["documents"], key=lambda item: item["id"]):
                annotations = document["annotation_sets"][0]["annotations"]
                if set(annotations) != set(ids):
                    raise ValueError(f"{member} document {document['id']} is missing hypotheses")
                rows.append(
                    {
                        "id": f"contractnli-{document['id']}",
                        "source_id": document["id"],
                        "document_type": document["document_type"],
                        "text": document["text"],
                        "labels": {key: annotations[key]["choice"] for key in ids},
                    }
                )
            # Deterministic prefix by source id, so a limited smoke dataset is reproducible.
            rows = rows[:limit] if limit is not None else rows
            _write_jsonl(dataset_dir / f"{split}.jsonl", rows)
            counts[split] = len(rows)
    freeze_test(dataset_dir)
    manifest = {
        "schema_version": 1,
        "skill_id": SKILL_ID,
        "preprocessing_version": PREPROCESSING_VERSION,
        "source": {
            "id": source["id"],
            "url": source["source"],
            "sha256": source["sha256"],
            "members": source["members"],
        },
        "contract_sha256": contract["review"]["artifact_sha256"],
        "limit_per_split": limit,
        "counts": counts,
        "attribution": source.get("attribution"),
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return dataset_dir


def validate_nda_dataset(
    dataset_dir: Path,
    *,
    contract_dir: Path = DEFAULT_CONTRACT_DIR,
    source_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    label_counts = {choice: 0 for choice in CHOICES}

    def error(check: str, location: str, message: str) -> None:
        errors.append({"check": check, "location": location, "message": message})

    try:
        contract = load_nda_contract(contract_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        error("contract", str(contract_dir), str(exc))
        return _report(False, counts, label_counts, {}, errors, warnings)
    source = dict(source_lock or contract["source"])
    ids = contract["hypothesis_ids"]
    schema = Draft202012Validator(contract["output_schema"])

    manifest: dict[str, Any] = {}
    try:
        manifest = _read_object(dataset_dir / "manifest.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        error("manifest", "manifest.json", str(exc))
    if manifest:
        if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
            error("manifest", "preprocessing_version", "dataset was built by a different preprocessor")
        if manifest.get("source", {}).get("sha256") != source["sha256"]:
            error("source_pin", "manifest.source", "dataset was not built from the pinned archive")
        if manifest.get("contract_sha256") != contract["review"]["artifact_sha256"]:
            error("contract_pin", "manifest.contract_sha256", "dataset was built against another contract")

    seen_ids: dict[Any, str] = {}
    seen_text: dict[str, str] = {}
    for split in REQUIRED_SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        try:
            rows = read_jsonl(path)
        except (OSError, json.JSONDecodeError) as exc:
            error("split", split, str(exc))
            continue
        counts[split] = len(rows)
        if not rows:
            error("split", split, "split is empty")
        split_text: dict[str, str] = {}
        for index, row in enumerate(rows, start=1):
            location = f"{split}:{index}"
            if not isinstance(row, dict) or set(row) != ROW_KEYS:
                error("row_keys", location, f"row keys must be exactly {sorted(ROW_KEYS)}")
                continue
            if not isinstance(row["text"], str) or not row["text"].strip():
                error("text", location, "text must be a non-empty string")
                continue
            labels = row["labels"]
            if not isinstance(labels, dict) or list(labels) != ids:
                error("labels", location, "labels must list the 17 hypotheses in contract order")
                continue
            problems = sorted(item.message for item in schema.iter_errors(labels))
            if problems:
                error("schema", location, "; ".join(problems)[:500])
                continue
            for choice in labels.values():
                label_counts[choice] += 1
            for key, seen, scope in ((row["source_id"], seen_ids, "source_id"), (row["id"], seen_ids, "id")):
                if key in seen:
                    error("duplicate_id", location, f"{scope} {key} also appears in {seen[key]}")
                seen[key] = split
            digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            if digest in seen_text:
                error("split_leakage", location, f"document text also appears in {seen_text[digest]}")
            if digest in split_text:
                # ContractNLI TEST has one identical pair (ids 52 and 97); report, never alter TEST.
                warnings.append(
                    {"check": "duplicate_text", "location": location, "message": f"same text as {split_text[digest]}"}
                )
            split_text[digest] = f"{split}:{row['id']}"
        seen_text.update({digest: where.split(":", 1)[0] for digest, where in split_text.items()})

    limit = manifest.get("limit_per_split") if manifest else None
    for split, expected in source.get("splits", {}).items():
        if split not in counts:
            continue
        if limit is None and counts[split] != expected:
            error("counts", split, f"expected {expected} documents, found {counts[split]}")
        if limit is not None and counts[split] > min(limit, expected):
            error("counts", split, f"limited dataset has {counts[split]} documents, above {limit}")

    hashes: dict[str, Any] = {
        "splits": {
            split: sha256(dataset_dir / f"{split}.jsonl")
            for split in REQUIRED_SPLITS
            if (dataset_dir / f"{split}.jsonl").is_file()
        }
    }
    hash_path = dataset_dir / "test.sha256"
    if not hash_path.is_file():
        error("frozen_test", "test.sha256", "missing frozen TEST hash")
    elif hash_path.read_text(encoding="utf-8").strip() != hashes["splits"].get("test"):
        error("frozen_test", "test.jsonl", "test.jsonl does not match test.sha256")
    hashes["test_frozen"] = hash_path.read_text(encoding="utf-8").strip() if hash_path.is_file() else None
    return _report(not errors, counts, label_counts, hashes, errors, warnings)


def require_valid_nda_dataset(
    dataset_dir: Path,
    *,
    contract_dir: Path = DEFAULT_CONTRACT_DIR,
    report_path: Path | None = None,
) -> dict[str, Any]:
    report = validate_nda_dataset(dataset_dir, contract_dir=contract_dir)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise NdaValidationError(report)
    return report


def to_text_training_rows(
    dataset_dir: Path, split: str, *, contract_dir: Path = DEFAULT_CONTRACT_DIR
) -> list[dict[str, Any]]:
    """Conversational prompt/completion rows with plain-string content and no image column."""
    contract = load_nda_contract(contract_dir)
    prompt = contract["skill_spec"]["runtime_prompt"]
    ids = contract["hypothesis_ids"]
    rows = []
    for row in read_jsonl(dataset_dir / f"{split}.jsonl"):
        answer = json.dumps({key: row["labels"][key] for key in ids}, separators=(",", ":"))
        rows.append(
            {
                "prompt": [{"role": "user", "content": f"{prompt}\n\n{row['text']}"}],
                "completion": [{"role": "assistant", "content": answer}],
            }
        )
    return rows


def _verify_archive(path: Path, source: Mapping[str, Any]) -> None:
    if sha256(path) != source["sha256"]:
        raise ValueError(f"{path} does not match the pinned ContractNLI sha256")
    with zipfile.ZipFile(path) as archive:
        for member, expected in source["members"].items():
            if hashlib.sha256(archive.read(member)).hexdigest() != expected:
                raise ValueError(f"{member} does not match its pinned sha256")


def _member(source: Mapping[str, Any], split: str) -> str:
    matches = [name for name in source["members"] if name.endswith(f"/{split}.json")]
    if len(matches) != 1:
        raise ValueError(f"source lock must pin exactly one {split}.json")
    return matches[0]


def _numeric(labels: Mapping[str, Any]) -> list[str]:
    return sorted(labels, key=lambda key: int(key.split("-")[1]))


def _report(
    passed: bool,
    counts: dict[str, int],
    label_counts: dict[str, int],
    hashes: dict[str, Any],
    errors: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validator_version": VALIDATOR_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "passed": passed,
        "counts": counts,
        "label_counts": label_counts,
        "hashes": hashes,
        "errors": errors,
        "warnings": warnings,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="download, verify and write the NDA dataset")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_DIR)
    build.add_argument("--source-zip", type=Path)
    build.add_argument("--cache", type=Path)
    build.add_argument("--limit", type=int, help="documents per split, for smoke runs")
    check = commands.add_parser("validate", help="write a validation report; exit 1 if it fails")
    check.add_argument("--dataset", type=Path, required=True)
    check.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_DIR)
    check.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        path = materialize_nda_dataset(
            args.out,
            contract_dir=args.contract,
            source_zip=args.source_zip,
            cache_dir=args.cache,
            limit=args.limit,
        )
        print(path)
        return
    try:
        report = require_valid_nda_dataset(
            args.dataset, contract_dir=args.contract, report_path=args.report
        )
    except NdaValidationError as error:
        print(json.dumps(error.report["errors"][:20], indent=2))
        raise SystemExit(1) from error
    print(json.dumps({"passed": report["passed"], "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
