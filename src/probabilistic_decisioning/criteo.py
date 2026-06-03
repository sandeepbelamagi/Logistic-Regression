"""Streaming parser for the Criteo CTR dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from probabilistic_decisioning.constants import (
    CATEGORICAL_FEATURE_NAMES,
    CRITEO_EXPECTED_FIELD_COUNT,
    DENSE_FEATURE_NAMES,
)


@dataclass(frozen=True)
class CriteoRecord:
    """Single parsed row from the Criteo CTR dataset."""

    row_id: int
    label: int
    dense_features: dict[str, str | None]
    categorical_features: dict[str, str | None]


def parse_criteo_line(line: str, row_id: int) -> CriteoRecord:
    """Parse one Criteo TSV row into structured fields."""

    parts = line.rstrip("\r\n").split("\t")
    if len(parts) != CRITEO_EXPECTED_FIELD_COUNT:
        raise ValueError(
            f"Row {row_id} expected {CRITEO_EXPECTED_FIELD_COUNT} columns, got {len(parts)}."
        )

    label = _parse_label(parts[0], row_id)
    dense_values = {
        name: parts[index] or None for index, name in enumerate(DENSE_FEATURE_NAMES, start=1)
    }
    categorical_offset = 1 + len(DENSE_FEATURE_NAMES)
    categorical_values = {
        name: parts[index] or None
        for index, name in enumerate(CATEGORICAL_FEATURE_NAMES, start=categorical_offset)
    }
    return CriteoRecord(
        row_id=row_id,
        label=label,
        dense_features=dense_values,
        categorical_features=categorical_values,
    )


def iter_criteo_records(path: Path) -> Iterator[CriteoRecord]:
    """Yield parsed Criteo records from a TSV file."""

    with path.open("r", encoding="utf-8") as handle:
        for row_id, line in enumerate(handle):
            if not line.strip():
                continue
            yield parse_criteo_line(line, row_id=row_id)


def count_criteo_rows(path: Path) -> int:
    """Count non-empty rows for contiguous splitting."""

    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _parse_label(raw_label: str, row_id: int) -> int:
    if raw_label not in {"0", "1"}:
        raise ValueError(f"Row {row_id} has invalid binary label: {raw_label!r}.")
    return int(raw_label)
