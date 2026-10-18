from __future__ import annotations

import unittest

from asi1.validation import validate_output


class ValidationTests(unittest.TestCase):
    def test_validates_receipt_arithmetic_when_fields_exist(self):
        result = validate_output({"subtotal": "$10.00", "tax": "0.80", "total": 10.8})
        self.assertTrue(result["structured"])
        self.assertTrue(result["arithmetic_checked"])
        self.assertTrue(result["arithmetic_valid"])

    def test_does_not_invent_validation_for_other_schemas(self):
        result = validate_output({"effective_date": "2026-10-18"})
        self.assertTrue(result["structured"])
        self.assertFalse(result["arithmetic_checked"])
        self.assertIsNone(result["arithmetic_valid"])


if __name__ == "__main__":
    unittest.main()
