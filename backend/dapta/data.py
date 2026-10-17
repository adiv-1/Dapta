from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_SPLITS = ("train", "dev", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(dataset_dir: Path, *, require_frozen_test: bool = True) -> dict[str, int]:
    dataset_dir = dataset_dir.resolve()
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    seen_cases: dict[str, str] = {}

    for split in REQUIRED_SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        if not path.is_file():
            raise ValueError(f"missing dataset split: {path}")
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"empty dataset split: {split}")
        rows_by_split[split] = rows

        for index, row in enumerate(rows, start=1):
            _validate_row(row, dataset_dir, split, index)
            case_id = row["id"]
            if case_id in seen_ids:
                raise ValueError(f"duplicate id across splits: {case_id}")
            seen_ids.add(case_id)
            fingerprint = _case_fingerprint(row, dataset_dir)
            if fingerprint in seen_cases:
                raise ValueError(f"duplicate case across {seen_cases[fingerprint]} and {split}: {case_id}")
            seen_cases[fingerprint] = split

    if require_frozen_test:
        hash_path = dataset_dir / "test.sha256"
        if not hash_path.is_file():
            raise ValueError(f"missing frozen test hash: {hash_path}")
        expected = hash_path.read_text(encoding="utf-8").strip()
        actual = sha256(dataset_dir / "test.jsonl")
        if expected != actual:
            raise ValueError("test.jsonl does not match test.sha256")

    return {split: len(rows) for split, rows in rows_by_split.items()}


def freeze_test(dataset_dir: Path) -> str:
    test_path = dataset_dir / "test.jsonl"
    if not test_path.is_file():
        raise ValueError(f"missing test split: {test_path}")
    digest = sha256(test_path)
    (dataset_dir / "test.sha256").write_text(f"{digest}\n", encoding="utf-8")
    return digest


def to_training_rows(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    rows = read_jsonl(dataset_dir / f"{split}.jsonl")
    output = []
    for row in rows:
        images = [str((dataset_dir / value).resolve()) for value in row["images"]]
        image_parts = [{"type": "image"} for _ in images]
        completion = row["completion"]
        if not isinstance(completion, str):
            completion = json.dumps(completion, ensure_ascii=False, separators=(",", ":"))
        output.append(
            {
                "images": images,
                "prompt": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": row.get("system", "Follow the requested output schema exactly."),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": image_parts + [{"type": "text", "text": row["prompt"]}],
                    },
                ],
                "completion": [
                    {"role": "assistant", "content": [{"type": "text", "text": completion}]}
                ],
            }
        )
    return output


def _validate_row(row: dict[str, Any], dataset_dir: Path, split: str, index: int) -> None:
    required = {"id", "images", "prompt", "completion"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"{split}:{index} missing {sorted(missing)}")
    if not isinstance(row["id"], str) or not row["id"].strip():
        raise ValueError(f"{split}:{index} has invalid id")
    if not isinstance(row["prompt"], str) or not row["prompt"].strip():
        raise ValueError(f"{split}:{index} has invalid prompt")
    if not isinstance(row["images"], list):
        raise ValueError(f"{split}:{index} images must be a list")
    for image in row["images"]:
        path = (dataset_dir / image).resolve()
        if dataset_dir not in path.parents or not path.is_file():
            raise ValueError(f"{split}:{index} has missing or unsafe image path: {image}")


def _case_fingerprint(row: dict[str, Any], dataset_dir: Path) -> str:
    digest = hashlib.sha256(row["prompt"].strip().encode())
    for image in row["images"]:
        digest.update(bytes.fromhex(sha256((dataset_dir / image).resolve())))
    return digest.hexdigest()
