import tempfile
import unittest
from pathlib import Path

from dapta.comparisons import ComparisonService, ComparisonStore
from dapta.config import Settings
from dapta.ledger import UsageLedger


class ComparisonScoringTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        path = Path(self.temporary_directory.name) / "dapta.sqlite3"
        self.service = ComparisonService(
            store=ComparisonStore(path),
            ledger=UsageLedger(path),
            settings=Settings.from_env(),
            openrouter=None,
        )
        self.metadata = {
            "required_fields": ["effective_date", "jurisdiction", "parties", "term"],
            "output_schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["effective_date", "jurisdiction", "parties", "term"],
                "properties": {
                    "effective_date": {"type": ["string", "null"]},
                    "jurisdiction": {"type": ["string", "null"]},
                    "parties": {"type": "array", "items": {"type": "string"}},
                    "term": {"type": ["string", "null"]},
                },
            },
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_exact_four_field_output_is_schema_valid(self):
        results = {"model": {"output": {
            "effective_date": "2025-03-07",
            "jurisdiction": "New_York",
            "parties": ["Acme_Inc.", "Beta_LLC"],
            "term": "3_years",
        }}}
        self.service._score_results("NDA", results, self.metadata)
        self.assertTrue(results["model"]["metrics"]["schema_valid"])

    def test_wrapped_or_extra_key_output_is_not_schema_valid(self):
        for output in (
            {"nda": {"effective_date": "2025-03-07"}},
            {
                "effective_date": "2025-03-07",
                "jurisdiction": "New_York",
                "parties": ["Acme_Inc.", "Beta_LLC"],
                "term": "3_years",
                "summary": "extra",
            },
        ):
            with self.subTest(output=output):
                results = {"model": {"output": output}}
                self.service._score_results("NDA", results, self.metadata)
                self.assertFalse(results["model"]["metrics"]["schema_valid"])


if __name__ == "__main__":
    unittest.main()
