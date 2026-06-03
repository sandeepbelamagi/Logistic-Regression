"""Feature engineering helpers for Criteo-style sparse Logistic Regression."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from probabilistic_decisioning.constants import CATEGORICAL_FEATURE_NAMES, DENSE_FEATURE_NAMES
from probabilistic_decisioning.criteo import CriteoRecord


@dataclass(frozen=True)
class FeatureConfig:
    """Controls deterministic feature engineering behavior."""

    hash_dimension: int
    include_missing_category_token: bool = True
    dense_log_transform: bool = True


def build_dense_vector(record: CriteoRecord, config: FeatureConfig) -> list[float]:
    """Convert raw dense features into model-ready numeric values."""

    return [
        transform_dense_value(record.dense_features.get(feature_name), config)
        for feature_name in DENSE_FEATURE_NAMES
    ]


def build_sparse_vector(record: CriteoRecord, config: FeatureConfig) -> tuple[list[int], list[float]]:
    """Hash categorical features into a sparse active-feature representation."""

    feature_ids: list[int] = []
    feature_values: list[float] = []

    for feature_name in CATEGORICAL_FEATURE_NAMES:
        raw_value = record.categorical_features.get(feature_name)
        if raw_value is None and not config.include_missing_category_token:
            continue
        token_value = raw_value if raw_value is not None else "__MISSING__"
        feature_ids.append(hash_feature_token(feature_name, token_value, config.hash_dimension))
        feature_values.append(1.0)

    return feature_ids, feature_values


def transform_dense_value(raw_value: str | None, config: FeatureConfig) -> float:
    """Apply a stable transform to the 13 dense Criteo features."""

    if raw_value is None:
        return 0.0

    numeric_value = float(raw_value)
    if numeric_value < 0:
        numeric_value = 0.0

    if config.dense_log_transform:
        return math.log1p(numeric_value)
    return numeric_value


def hash_feature_token(feature_name: str, feature_value: str, hash_dimension: int) -> int:
    """Hash one namespaced categorical token into the configured sparse space."""

    if hash_dimension <= 0:
        raise ValueError("hash_dimension must be positive.")

    digest = hashlib.blake2b(
        f"{feature_name}={feature_value}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % hash_dimension
