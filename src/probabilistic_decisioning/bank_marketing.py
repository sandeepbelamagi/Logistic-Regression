"""Parser for the UCI Bank Marketing dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from probabilistic_decisioning.constants import (
    BANK_MARKETING_CATEGORICAL_FEATURE_NAMES,
    BANK_MARKETING_EXPECTED_FIELD_COUNT,
    BANK_MARKETING_INPUT_FIELDS,
    BANK_MARKETING_NUMERIC_FEATURE_NAMES,
)


@dataclass(frozen=True)
class BankMarketingRecord:
    """Single parsed row from the classic Bank Marketing dataset."""

    row_id: int
    label: int
    numeric_features: dict[str, str | None]
    categorical_features: dict[str, str | None]
    leakage_prone_features: dict[str, str | None]


def parse_bank_marketing_row(fields: list[str], row_id: int) -> BankMarketingRecord:
    """Parse one semicolon-delimited Bank Marketing row."""

    normalized_fields = [field.strip() for field in fields]
    if len(normalized_fields) != BANK_MARKETING_EXPECTED_FIELD_COUNT:
        raise ValueError(
            f"Row {row_id} expected {BANK_MARKETING_EXPECTED_FIELD_COUNT} columns, got {len(normalized_fields)}."
        )

    label = _parse_label(normalized_fields[-1], row_id)
    source_values = dict(zip(BANK_MARKETING_INPUT_FIELDS, normalized_fields[:-1], strict=True))
    numeric_features = {
        feature_name: source_values[feature_name] or None
        for feature_name in BANK_MARKETING_NUMERIC_FEATURE_NAMES
    }
    categorical_features = {
        feature_name: source_values[feature_name] or None
        for feature_name in BANK_MARKETING_CATEGORICAL_FEATURE_NAMES
    }
    leakage_prone_features = {
        "duration": source_values["duration"] or None,
    }
    return BankMarketingRecord(
        row_id=row_id,
        label=label,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        leakage_prone_features=leakage_prone_features,
    )


def iter_bank_marketing_records(path: Path) -> Iterator[BankMarketingRecord]:
    """Yield parsed Bank Marketing rows from a CSV file."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        data_row_id = 0
        for raw_fields in reader:
            if not raw_fields or all(not field.strip() for field in raw_fields):
                continue
            if _looks_like_header(raw_fields):
                continue
            yield parse_bank_marketing_row(raw_fields, row_id=data_row_id)
            data_row_id += 1


def count_bank_marketing_rows(path: Path) -> int:
    """Count non-empty data rows in a Bank Marketing CSV file."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        total_rows = 0
        for raw_fields in reader:
            if not raw_fields or all(not field.strip() for field in raw_fields):
                continue
            if _looks_like_header(raw_fields):
                continue
            total_rows += 1
        return total_rows


def _looks_like_header(fields: list[str]) -> bool:
    normalized = [field.strip().lower() for field in fields]
    return normalized == [field.lower() for field in BANK_MARKETING_INPUT_FIELDS + ["y"]]


def _parse_label(raw_label: str, row_id: int) -> int:
    normalized = raw_label.strip().lower()
    if normalized == "yes":
        return 1
    if normalized == "no":
        return 0
    raise ValueError(f"Row {row_id} has invalid binary label: {raw_label!r}.")
