import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from dapta.data import read_jsonl, validate_dataset
from dapta.receipts import materialize_receipt_dataset
from dapta.validation import DatasetValidationError, require_valid_receipt_dataset


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "research/adapter-assets/sources.lock.json"
CONTRACT = ROOT / "research/adapter-assets/skills/receipt-to-expense"


class FakeImage:
    def __init__(self, value):
        self.value = value

    def convert(self, mode):
        if mode != "RGB":
            raise ValueError(mode)
        return self

    def save(self, path, format):
        if format != "PNG":
            raise ValueError(format)
        color = tuple(hashlib.sha256(self.value.encode()).digest()[:3])
        Image.new("RGB", (64, 64), color).save(path, format="PNG")


class FakeLoader:
    def __init__(self):
        self.calls = []

    def __call__(self, source, *, revision, split, streaming, cache_dir, token):
        self.calls.append((source, revision, split, streaming, cache_dir, token))
        if source == "naver-clova-ix/cord-v2":
            return [self.cord(split)]
        if split == "test":
            return [self.sroie("sroie-test")]
        return [self.sroie(f"sroie-train-{index}") for index in range(20)]

    @staticmethod
    def cord(split):
        annotation = {
            "gt_parse": {
                "menu": [{"nm": "Lunch", "cnt": "1 x", "price": "75,000"}],
                "sub_total": {
                    "subtotal_price": ["75,000", "75,000"],
                    "tax_price": "7,500",
                },
                "total": {"total_price": "82,500"},
            },
            "meta": {"image_id": f"cord-{split}"},
        }
        return {
            "image": FakeImage(f"cord-{split}"),
            "ground_truth": json.dumps(annotation),
        }

    @staticmethod
    def sroie(key):
        return {
            "image": FakeImage(key),
            "key": key,
            "entities": {
                "company": "Example Store",
                "date": "17/10/2026",
                "address": "1 Main Street",
                "total": "31.00",
            },
        }


class ReceiptDatasetTests(unittest.TestCase):
    def test_remote_worker_guard_blocks_local_downloads(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "remote worker"):
                    materialize_receipt_dataset(
                        Path(temp), source_manifest=SOURCES, contract_dir=CONTRACT
                    )

    def test_pinned_sources_materialize_to_valid_qwen_splits(self):
        loader = FakeLoader()
        with tempfile.TemporaryDirectory() as temp:
            job_dir = Path(temp)
            dataset_dir = materialize_receipt_dataset(
                job_dir,
                source_manifest=SOURCES,
                contract_dir=CONTRACT,
                loader=loader,
                allow_local_test=True,
            )

            counts = validate_dataset(dataset_dir)
            self.assertEqual(sum(counts.values()), 24)
            self.assertGreater(counts["train"], 1)
            self.assertGreater(counts["dev"], 1)
            self.assertEqual(counts["test"], 2)
            self.assertFalse((job_dir / "source-cache").exists())
            self.assertTrue(all(call[3] for call in loader.calls))
            self.assertEqual({call[2] for call in loader.calls}, {"train", "validation", "test"})

            train_rows = read_jsonl(dataset_dir / "train.jsonl")
            cord = next(row for row in train_rows if row["metadata"]["source_id"] == "receipt-cord-v2")
            receipt = cord["completion"]["receipts"][0]
            self.assertEqual(receipt["line_items"][0], {
                "description": "Lunch",
                "quantity": 1,
                "amount": 75000,
            })
            self.assertEqual(receipt["tax"], 7500)
            self.assertEqual(receipt["total"], 82500)

            sroie_rows = [
                row
                for split in ("train", "dev", "test")
                for row in read_jsonl(dataset_dir / f"{split}.jsonl")
                if row["metadata"]["source_id"] == "receipt-sroie"
            ]
            self.assertEqual(sroie_rows[0]["completion"]["receipts"][0]["transaction_date"], "2026-10-17")

            manifest = json.loads((dataset_dir / "manifest.json").read_text())
            self.assertEqual(manifest["counts"], counts)
            self.assertEqual(manifest["frozen_test_sha256"], (dataset_dir / "test.sha256").read_text().strip())

            report = require_valid_receipt_dataset(
                dataset_dir,
                contract_dir=CONTRACT,
                source_manifest=SOURCES,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["checked"]["images"], 24)

    def test_receipt_validation_rejects_schema_image_and_arithmetic_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset_dir = materialize_receipt_dataset(
                Path(temp),
                source_manifest=SOURCES,
                contract_dir=CONTRACT,
                loader=FakeLoader(),
                allow_local_test=True,
            )
            rows = read_jsonl(dataset_dir / "train.jsonl")
            rows[0]["completion"]["unexpected"] = True
            receipt = rows[0]["completion"]["receipts"][0]
            receipt.update({"subtotal": 10, "tax": 2, "service_charge": 1, "total": 99})
            (dataset_dir / "train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            image = dataset_dir / rows[0]["images"][0]
            image.write_bytes(b"not an image")
            report_path = Path(temp) / "validation.json"

            with self.assertRaises(DatasetValidationError) as caught:
                require_valid_receipt_dataset(
                    dataset_dir,
                    contract_dir=CONTRACT,
                    source_manifest=SOURCES,
                    report_path=report_path,
                )

            checks = {item["check"] for item in caught.exception.report["errors"]}
            self.assertTrue({"output_schema", "image_decode", "scalar_arithmetic"} <= checks)
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
