"""Feature engineering helpers for Bank Marketing sparse Logistic Regression."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from probabilistic_decisioning.bank_marketing import BankMarketingRecord
from probabilistic_decisioning.constants import (
    BANK_MARKETING_CATEGORICAL_FEATURE_NAMES,
    BANK_MARKETING_NUMERIC_FEATURE_NAMES,
    BANK_MARKETING_LEAKAGE_FEATURE_NAME,
)


@dataclass(frozen=True)
class FeatureConfig:
    """Controls deterministic feature engineering behavior."""

    hash_dimension: int
    include_missing_category_token: bool = True
    dense_log_transform: bool = True
    include_duration_feature: bool = False


def build_dense_vector(record: BankMarketingRecord, config: FeatureConfig) -> list[float]:
    """Convert raw numeric features into model-ready values."""

    dense_values = [
        transform_dense_value(record.numeric_features.get(feature_name), config, feature_name)
        for feature_name in BANK_MARKETING_NUMERIC_FEATURE_NAMES
    ]

    dense_values.append(
        transform_prior_contact_flag(record.numeric_features.get("pdays"))
    )

    if config.include_duration_feature:
        dense_values.append(
            transform_dense_value(
                record.leakage_prone_features.get(BANK_MARKETING_LEAKAGE_FEATURE_NAME),
                config,
                BANK_MARKETING_LEAKAGE_FEATURE_NAME,
            )
        )

    return dense_values


def build_sparse_vector(
    record: BankMarketingRecord, config: FeatureConfig
) -> tuple[list[int], list[float]]:
    """Hash categorical features into a sparse active-feature representation."""

    feature_ids: list[int] = []
    feature_values: list[float] = []

    for feature_name in BANK_MARKETING_CATEGORICAL_FEATURE_NAMES:
        raw_value = record.categorical_features.get(feature_name)
        if raw_value is None and not config.include_missing_category_token:
            continue
        token_value = raw_value if raw_value is not None else "__MISSING__"
        feature_ids.append(hash_feature_token(feature_name, token_value, config.hash_dimension))
        feature_values.append(1.0)

    return feature_ids, feature_values


def transform_dense_value(
    raw_value: str | None, config: FeatureConfig, feature_name: str
) -> float:
    """Apply a stable transform to numeric Bank Marketing features."""

    if raw_value is None:
        return 0.0

    numeric_value = float(raw_value)
    if feature_name == BANK_MARKETING_LEAKAGE_FEATURE_NAME:
        if numeric_value < 0:
            numeric_value = 0.0
    elif feature_name == "balance":
        numeric_value = _signed_log1p(numeric_value)
        return numeric_value
    elif feature_name == "pdays":
        if numeric_value < 0:
            numeric_value = 0.0

    if config.dense_log_transform:
        return math.log1p(numeric_value)
    return numeric_value


def transform_prior_contact_flag(raw_value: str | None) -> float:
    """Return a 0/1 indicator for whether the client was contacted before."""

    if raw_value is None:
        return 0.0
    return 1.0 if float(raw_value) >= 0 else 0.0


def hash_feature_token(feature_name: str, feature_value: str, hash_dimension: int) -> int:
    """Hash one namespaced categorical token into the configured sparse space."""

    if hash_dimension <= 0:
        raise ValueError("hash_dimension must be positive.")

    digest = hashlib.blake2b(
        f"{feature_name}={feature_value}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % hash_dimension


def _signed_log1p(value: float) -> float:
    magnitude = math.log1p(abs(value))
    if value < 0:
        return -magnitude
    return magnitude
