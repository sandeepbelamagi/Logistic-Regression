from __future__ import annotations

import math
import unittest

from tests import _bootstrap  # noqa: F401

from probabilistic_decisioning.criteo import CriteoRecord
from probabilistic_decisioning.features import (
    FeatureConfig,
    build_dense_vector,
    build_sparse_vector,
    hash_feature_token,
)


class FeatureEngineeringTest(unittest.TestCase):
    def setUp(self) -> None:
        dense_features = {f"I{index}": ("0" if index == 1 else None) for index in range(1, 14)}
        dense_features["I2"] = "9"
        dense_features["I3"] = "-4"
        categorical_features = {f"C{index}": None for index in range(1, 27)}
        categorical_features["C1"] = "alpha"
        categorical_features["C2"] = "beta"
        self.record = CriteoRecord(
            row_id=1,
            label=1,
            dense_features=dense_features,
            categorical_features=categorical_features,
        )
        self.config = FeatureConfig(hash_dimension=1024)

    def test_dense_vector_applies_log1p_and_clamps_negative(self) -> None:
        dense_vector = build_dense_vector(self.record, self.config)

        self.assertEqual(dense_vector[0], 0.0)
        self.assertAlmostEqual(dense_vector[1], math.log1p(9.0))
        self.assertEqual(dense_vector[2], 0.0)
        self.assertEqual(len(dense_vector), 13)

    def test_sparse_vector_keeps_all_categorical_slots(self) -> None:
        feature_ids, feature_values = build_sparse_vector(self.record, self.config)

        self.assertEqual(len(feature_ids), 26)
        self.assertEqual(len(feature_values), 26)
        self.assertTrue(all(value == 1.0 for value in feature_values))
        self.assertTrue(all(0 <= feature_id < 1024 for feature_id in feature_ids))

    def test_hashing_is_stable(self) -> None:
        first = hash_feature_token("C1", "alpha", 4096)
        second = hash_feature_token("C1", "alpha", 4096)
        third = hash_feature_token("C1", "gamma", 4096)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
