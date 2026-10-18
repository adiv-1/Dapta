from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SPLITS = ("train", "dev", "test", "ood")
DEFAULT_MINIMUMS = {
    "train": 16,
    "dev": 4,
    "test": 4,
    "ood": 2,
}
SMOKE_PROFILE = "smoke"
SMOKE_MINIMUMS = {
    "train": 4,
    "dev": 2,
    "test": 2,
    "ood": 2,
}
SMOKE_MAX_EXAMPLES = 10
SMOKE_MAX_TRAIN_SECONDS = 120
SMOKE_MAX_ROUNDS = 1
SMOKE_EPOCHS = 1.0


@dataclass(frozen=True)
class DatasetPolicy:
    minimums: dict[str, int]
    maximum_total: int | None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "DatasetPolicy":
        values = os.environ if environ is None else environ
        minimums = {
            split: _positive_int(
                values.get(f"DAPTA_MIN_{split.upper()}_EXAMPLES"),
                default=DEFAULT_MINIMUMS[split],
                name=f"DAPTA_MIN_{split.upper()}_EXAMPLES",
            )
            for split in SPLITS
        }
        raw_cap = values.get("DAPTA_FACTORY_MAX_DATASET_EXAMPLES")
        maximum_total = (
            _positive_int(
                raw_cap,
                default=sum(minimums.values()),
                name="DAPTA_FACTORY_MAX_DATASET_EXAMPLES",
            )
            if raw_cap not in {None, ""}
            else None
        )
        if maximum_total is not None and sum(minimums.values()) > maximum_total:
            raise ValueError(
                "dataset split minimums exceed DAPTA_FACTORY_MAX_DATASET_EXAMPLES"
            )
        return cls(minimums=minimums, maximum_total=maximum_total)

    def generation_instruction(self) -> str:
        counts = ", ".join(
            f"{self.minimums[split]} {split.upper()}" for split in SPLITS
        )
        if self.maximum_total is not None:
            return (
                f"Return exactly {counts} examples and never more than "
                f"{self.maximum_total} examples total."
            )
        return f"Return at least {counts} examples."

    def bound(self, dataset: Mapping[str, object]) -> dict[str, object]:
        """Apply the supervised-demo cap without manufacturing missing examples."""
        if self.maximum_total is None:
            return dict(dataset)
        bounded: dict[str, object] = {}
        for split in SPLITS:
            cases = dataset.get(split)
            bounded[split] = (
                list(cases)[: self.minimums[split]] if isinstance(cases, list) else cases
            )
        return bounded


def smoke_training_limits() -> dict[str, Any]:
    """Hard cap for a public-agent smoke build. Callers cannot raise these numbers."""
    return {
        "profile": SMOKE_PROFILE,
        "max_examples": SMOKE_MAX_EXAMPLES,
        "minimums": dict(SMOKE_MINIMUMS),
        "max_train_examples": SMOKE_MINIMUMS["train"],
        "max_test_examples": SMOKE_MINIMUMS["test"],
        "max_train_seconds": SMOKE_MAX_TRAIN_SECONDS,
        "max_rounds": SMOKE_MAX_ROUNDS,
        "epochs": SMOKE_EPOCHS,
    }


def smoke_dataset_policy() -> DatasetPolicy:
    return DatasetPolicy(minimums=dict(SMOKE_MINIMUMS), maximum_total=SMOKE_MAX_EXAMPLES)


def apply_smoke_training_limits(bundle: dict[str, Any], limits: Mapping[str, Any] | None) -> None:
    if not isinstance(limits, Mapping) or limits.get("profile") != SMOKE_PROFILE:
        return
    canonical = smoke_training_limits()
    bundle["epochs"] = canonical["epochs"]
    bundle["max_train_examples"] = canonical["max_train_examples"]
    bundle["max_test_examples"] = canonical["max_test_examples"]
    bundle["max_training_seconds"] = canonical["max_train_seconds"]


def training_access_path(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    return Path(values.get("DAPTA_TRAINING_ACCESS_PATH") or "artifacts/asi1-training-access.json")


def training_access_open(path: Path | None = None, *, now: float | None = None) -> bool:
    file = training_access_path() if path is None else path
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    if not isinstance(payload, dict) or payload.get("enabled") is not True:
        return False
    try:
        expires_at = float(payload["expires_at"])
    except (KeyError, TypeError, ValueError):
        return False
    current = time.time() if now is None else now
    return current < expires_at


def _positive_int(value: str | None, *, default: int, name: str) -> int:
    try:
        result = int(value) if value not in {None, ""} else default
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result
