"""Offline artifact builder for the Phase 2 Bank Marketing pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from probabilistic_decisioning.constants import (
    DEFAULT_FEATURE_SET_VERSION,
    DEFAULT_HASH_DIMENSION,
    DEFAULT_START_TIMESTAMP,
)
from probabilistic_decisioning.contracts import (
    ContractContext,
    build_event_timestamp,
    build_raw_contact_event,
    build_subscription_label,
    build_training_row,
)
from probabilistic_decisioning.bank_marketing import (
    count_bank_marketing_rows,
    iter_bank_marketing_records,
)
from probabilistic_decisioning.features import FeatureConfig


@dataclass(frozen=True)
class DatasetBuilderConfig:
    """Configuration for building raw and training artifacts from Bank Marketing."""

    input_path: Path
    output_dir: Path
    hash_dimension: int = DEFAULT_HASH_DIMENSION
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION
    task_context: str = "bank_marketing"
    split_strategy: str = "hash"
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    label_delay_seconds: int = 0
    start_timestamp: str = DEFAULT_START_TIMESTAMP
    seconds_per_row: int = 1
    max_rows: int | None = None
    source_name: str = "bank_full_csv"
    attribution_window_hours: int = 24

    def validate(self) -> None:
        ratio_sum = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError(f"train/validation/test ratios must sum to 1.0, got {ratio_sum}.")
        if self.split_strategy not in {"hash", "contiguous"}:
            raise ValueError("split_strategy must be one of: hash, contiguous.")
        if self.seconds_per_row <= 0:
            raise ValueError("seconds_per_row must be positive.")
        if self.label_delay_seconds < 0:
            raise ValueError("label_delay_seconds must be non-negative.")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive when provided.")


def build_dataset(config: DatasetBuilderConfig) -> Path:
    """Materialize raw and training artifacts from a local Bank Marketing file."""

    config.validate()
    total_rows = _resolve_total_rows(config)
    feature_config = FeatureConfig(hash_dimension=config.hash_dimension)
    contract_context = ContractContext(
        feature_set_version=config.feature_set_version,
        task_context=config.task_context,
        source_name=config.source_name,
        attribution_window_hours=config.attribution_window_hours,
        label_delay_seconds=config.label_delay_seconds,
    )

    raw_dir = config.output_dir / "raw"
    training_dir = config.output_dir / "training"
    metadata_dir = config.output_dir / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    split_stats = {
        "train": {"rows": 0, "positives": 0},
        "validation": {"rows": 0, "positives": 0},
        "test": {"rows": 0, "positives": 0},
    }
    overall_rows = 0
    overall_positives = 0

    raw_event_path = raw_dir / "raw_contact_events.jsonl"
    raw_label_path = raw_dir / "raw_subscription_labels.jsonl"
    train_path = training_dir / "train.jsonl"
    validation_path = training_dir / "validation.jsonl"
    test_path = training_dir / "test.jsonl"

    with (
        raw_event_path.open("w", encoding="utf-8") as raw_event_handle,
        raw_label_path.open("w", encoding="utf-8") as raw_label_handle,
        train_path.open("w", encoding="utf-8") as train_handle,
        validation_path.open("w", encoding="utf-8") as validation_handle,
        test_path.open("w", encoding="utf-8") as test_handle,
    ):
        split_handles = {
            "train": train_handle,
            "validation": validation_handle,
            "test": test_handle,
        }

        for processed_row_count, record in enumerate(iter_bank_marketing_records(config.input_path)):
            if config.max_rows is not None and processed_row_count >= config.max_rows:
                break

            event_id = f"evt_{record.row_id:012d}"
            request_id = f"req_{record.row_id:012d}"
            event_timestamp = build_event_timestamp(
                start_timestamp=config.start_timestamp,
                row_id=record.row_id,
                seconds_per_row=config.seconds_per_row,
            )
            split_name = _assign_split(
                event_id=event_id,
                row_id=processed_row_count,
                total_rows=total_rows,
                config=config,
            )

            raw_event = build_raw_contact_event(
                record=record,
                event_id=event_id,
                request_id=request_id,
                event_timestamp=event_timestamp,
                context=contract_context,
            )
            raw_label = build_subscription_label(
                record=record,
                event_id=event_id,
                event_timestamp=event_timestamp,
                context=contract_context,
            )
            training_row = build_training_row(
                record=record,
                event_id=event_id,
                event_timestamp=event_timestamp,
                feature_config=feature_config,
                context=contract_context,
            )

            _write_jsonl(raw_event_handle, raw_event)
            _write_jsonl(raw_label_handle, raw_label)
            _write_jsonl(split_handles[split_name], training_row)

            split_stats[split_name]["rows"] += 1
            split_stats[split_name]["positives"] += record.label
            overall_rows += 1
            overall_positives += record.label

    summary_path = metadata_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _build_summary(
                config=config,
                overall_rows=overall_rows,
                overall_positives=overall_positives,
                split_stats=split_stats,
            ),
            handle,
            indent=2,
            sort_keys=True,
        )

    return summary_path


def _resolve_total_rows(config: DatasetBuilderConfig) -> int | None:
    if config.split_strategy == "contiguous":
        total_rows = count_bank_marketing_rows(config.input_path)
        if config.max_rows is not None:
            return min(total_rows, config.max_rows)
        return total_rows
    if config.max_rows is not None:
        return config.max_rows
    return None


def _assign_split(
    event_id: str,
    row_id: int,
    total_rows: int | None,
    config: DatasetBuilderConfig,
) -> str:
    if config.split_strategy == "contiguous":
        if total_rows is None or total_rows == 0:
            raise ValueError("contiguous splitting requires a positive total_rows value.")
        fraction = (row_id + 1) / total_rows
    else:
        fraction = _stable_fraction(event_id)

    train_threshold = config.train_ratio
    validation_threshold = config.train_ratio + config.validation_ratio

    if fraction < train_threshold:
        return "train"
    if fraction < validation_threshold:
        return "validation"
    return "test"


def _stable_fraction(value: str) -> float:
    stable_hash = 0
    for character in value:
        stable_hash = (stable_hash * 131 + ord(character)) % 10_000
    return stable_hash / 10_000


def _write_jsonl(handle: TextIO, payload: dict[str, object]) -> None:
    handle.write(json.dumps(payload, separators=(",", ":")))
    handle.write("\n")


def _build_summary(
    config: DatasetBuilderConfig,
    overall_rows: int,
    overall_positives: int,
    split_stats: dict[str, dict[str, int]],
) -> dict[str, object]:
    return {
        "source_path": str(config.input_path),
        "output_dir": str(config.output_dir),
        "source_name": config.source_name,
        "task_context": config.task_context,
        "feature_set_version": config.feature_set_version,
        "hash_dimension": config.hash_dimension,
        "split_strategy": config.split_strategy,
        "label_delay_seconds": config.label_delay_seconds,
        "processed_rows": overall_rows,
        "positive_rate": _safe_rate(overall_positives, overall_rows),
        "splits": {
            split_name: {
                "rows": stats["rows"],
                "positives": stats["positives"],
                "positive_rate": _safe_rate(stats["positives"], stats["rows"]),
            }
            for split_name, stats in split_stats.items()
        },
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
