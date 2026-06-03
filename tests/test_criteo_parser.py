from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401

from probabilistic_decisioning.constants import CRITEO_EXPECTED_FIELD_COUNT
from probabilistic_decisioning.criteo import parse_criteo_line


def _build_row(label: str = "1") -> str:
    dense_values = [str(index) for index in range(1, 14)]
    categorical_values = [f"cat{index}" for index in range(1, 27)]
    row = [label, *dense_values, *categorical_values]
    assert len(row) == CRITEO_EXPECTED_FIELD_COUNT
    return "\t".join(row)


class CriteoParserTest(unittest.TestCase):
    def test_parse_valid_row(self) -> None:
        record = parse_criteo_line(_build_row(), row_id=7)

        self.assertEqual(record.row_id, 7)
        self.assertEqual(record.label, 1)
        self.assertEqual(record.dense_features["I1"], "1")
        self.assertEqual(record.dense_features["I13"], "13")
        self.assertEqual(record.categorical_features["C1"], "cat1")
        self.assertEqual(record.categorical_features["C26"], "cat26")

    def test_parse_missing_values(self) -> None:
        row = _build_row().split("\t")
        row[2] = ""
        row[20] = ""
        record = parse_criteo_line("\t".join(row), row_id=3)

        self.assertIsNone(record.dense_features["I2"])
        self.assertIsNone(record.categorical_features["C7"])

    def test_invalid_column_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_criteo_line("1\t2\t3", row_id=1)

    def test_invalid_label_raises(self) -> None:
        row = _build_row(label="2")
        with self.assertRaises(ValueError):
            parse_criteo_line(row, row_id=4)
