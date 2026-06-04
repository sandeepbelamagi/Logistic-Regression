"""Data-contract aligned record builders for Phase 2 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from probabilistic_decisioning.constants import DEFAULT_FEATURE_SET_VERSION
from probabilistic_decisioning.bank_marketing import BankMarketingRecord
from probabilistic_decisioning.features import FeatureConfig, build_dense_vector, build_sparse_vector


JSONDict = dict[str, Any]


@dataclass(frozen=True)
class ContractContext:
    """Shared context for generating related contract records."""

    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION
    task_context: str = "bank_marketing"
    source_name: str = "bank_full_csv"
    attribution_window_hours: int = 24
    label_delay_seconds: int = 0


def build_event_timestamp(start_timestamp: str, row_id: int, seconds_per_row: int) -> datetime:
    """Create deterministic pseudo-event time for offline artifacts."""

    base_timestamp = datetime.fromisoformat(start_timestamp)
    if base_timestamp.tzinfo is None:
        base_timestamp = base_timestamp.replace(tzinfo=UTC)
    return base_timestamp + timedelta(seconds=row_id * seconds_per_row)


def build_raw_contact_event(
    record: BankMarketingRecord,
    event_id: str,
    request_id: str,
    event_timestamp: datetime,
    context: ContractContext,
) -> JSONDict:
    """Build a raw contact event aligned with the bank marketing contract."""

    context_features = _build_context_feature_map(record)
    return {
        "event_id": event_id,
        "event_ts": _isoformat(event_timestamp),
        "event_date": event_timestamp.date().isoformat(),
        "request_id": request_id,
        "contact_id": f"contact_{record.row_id:012d}",
        "campaign_id": f"campaign_{record.row_id // 100:06d}",
        "context_features": context_features,
        "source_name": context.source_name,
    }


def build_subscription_label(
    record: BankMarketingRecord,
    event_id: str,
    event_timestamp: datetime,
    context: ContractContext,
) -> JSONDict:
    """Build a subscription label record linked to a contact event."""

    label_timestamp = event_timestamp + timedelta(seconds=context.label_delay_seconds)
    return {
        "label_event_id": f"lbl_{record.row_id:012d}",
        "parent_event_id": event_id,
        "label_ts": _isoformat(label_timestamp),
        "label_date": label_timestamp.date().isoformat(),
        "subscribed": record.label,
        "attribution_window_hours": context.attribution_window_hours,
        "label_delay_seconds": context.label_delay_seconds,
        "source_name": context.source_name,
    }


def build_training_row(
    record: BankMarketingRecord,
    event_id: str,
    event_timestamp: datetime,
    feature_config: FeatureConfig,
    context: ContractContext,
) -> JSONDict:
    """Build a model-ready training row from parsed raw inputs."""

    dense_features = build_dense_vector(record, feature_config)
    sparse_feature_ids, sparse_feature_values = build_sparse_vector(record, feature_config)
    label_timestamp = event_timestamp + timedelta(seconds=context.label_delay_seconds)

    return {
        "training_row_id": f"trn_{record.row_id:012d}",
        "snapshot_date": event_timestamp.date().isoformat(),
        "event_id": event_id,
        "event_ts": _isoformat(event_timestamp),
        "label_ts": _isoformat(label_timestamp),
        "label_available": True,
        "label": record.label,
        "sample_weight": 1.0,
        "dense_features": dense_features,
        "sparse_feature_ids": sparse_feature_ids,
        "sparse_feature_values": sparse_feature_values,
        "feature_set_version": context.feature_set_version,
        "task_context": context.task_context,
    }


def _build_context_feature_map(record: BankMarketingRecord) -> dict[str, str]:
    context_features: dict[str, str] = {}

    for name, value in record.numeric_features.items():
        if value is not None:
            context_features[name] = value
    for name, value in record.categorical_features.items():
        if value is not None:
            context_features[name] = value
    for name, value in record.leakage_prone_features.items():
        if value is not None:
            context_features[name] = value

    return context_features


def _isoformat(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
