import json
import tempfile
import unittest
from pathlib import Path

from dapta.data import freeze_test, to_training_rows, validate_dataset


class DatasetTests(unittest.TestCase):
    def make_dataset(self) -> Path:
        root = Path(tempfile.mkdtemp())
        for split in ("train", "dev", "test"):
            image = root / f"{split}.png"
            image.write_bytes(split.encode())
            row = {
                "id": f"{split}-1",
                "images": [image.name],
                "prompt": f"Build the {split} expense report.",
                "completion": {"total": 10},
            }
            (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        freeze_test(root)
        return root

    def test_valid_dataset_and_training_shape(self):
        root = self.make_dataset()
        self.assertEqual(validate_dataset(root), {"train": 1, "dev": 1, "test": 1})
        row = to_training_rows(root, "train")[0]
        self.assertEqual(row["prompt"][-1]["role"], "user")
        self.assertEqual(row["prompt"][-1]["content"][0], {"type": "image"})
        self.assertEqual(row["prompt"][-1]["content"][-1]["type"], "text")
        self.assertEqual(row["completion"][0]["role"], "assistant")
        self.assertEqual(row["completion"][0]["content"][0]["type"], "text")

    def test_modified_test_split_is_rejected(self):
        root = self.make_dataset()
        test_path = root / "test.jsonl"
        row = json.loads(test_path.read_text(encoding="utf-8"))
        row["prompt"] = "Tampered after the test split was frozen."
        test_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "test.jsonl"):
            validate_dataset(root)

    def test_cross_split_duplicate_is_rejected(self):
        root = self.make_dataset()
        train = json.loads((root / "train.jsonl").read_text(encoding="utf-8"))
        train["id"] = "dev-1"
        (root / "train.jsonl").write_text(json.dumps(train) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            validate_dataset(root)


if __name__ == "__main__":
    unittest.main()
