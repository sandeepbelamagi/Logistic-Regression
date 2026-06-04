from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401

from probabilistic_decisioning.bank_marketing import (
    parse_bank_marketing_row,
)
from probabilistic_decisioning.constants import BANK_MARKETING_EXPECTED_FIELD_COUNT


def _build_row(label: str = "yes") -> list[str]:
    return [
        "30",
        "admin.",
        "married",
        "secondary",
        "no",
        "1787",
        "no",
        "no",
        "cellular",
        "19",
        "oct",
        "79",
        "1",
        "-1",
        "0",
        "unknown",
        label,
    ]


class BankMarketingParserTest(unittest.TestCase):
    def test_parse_valid_row(self) -> None:
        record = parse_bank_marketing_row(_build_row(), row_id=7)

        self.assertEqual(record.row_id, 7)
        self.assertEqual(record.label, 1)
        self.assertEqual(record.numeric_features["age"], "30")
        self.assertEqual(record.numeric_features["balance"], "1787")
        self.assertEqual(record.numeric_features["pdays"], "-1")
        self.assertEqual(record.categorical_features["job"], "admin.")
        self.assertEqual(record.categorical_features["poutcome"], "unknown")
        self.assertEqual(record.leakage_prone_features["duration"], "79")

    def test_parse_no_label(self) -> None:
        record = parse_bank_marketing_row(_build_row(label="no"), row_id=3)

        self.assertEqual(record.label, 0)

    def test_invalid_column_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_bank_marketing_row(["1", "2", "3"], row_id=1)

    def test_expected_field_count(self) -> None:
        self.assertEqual(BANK_MARKETING_EXPECTED_FIELD_COUNT, 17)
