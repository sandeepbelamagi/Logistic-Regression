from __future__ import annotations

import math
import unittest

from tests import _bootstrap  # noqa: F401

from probabilistic_decisioning.bank_marketing import BankMarketingRecord
from probabilistic_decisioning.features import (
    FeatureConfig,
    build_dense_vector,
    build_sparse_vector,
    hash_feature_token,
)


class FeatureEngineeringTest(unittest.TestCase):
    def setUp(self) -> None:
        numeric_features = {
            "age": "30",
            "balance": "-20",
            "day": "19",
            "campaign": "3",
            "pdays": "-1",
            "previous": "2",
        }
        categorical_features = {
            "job": "admin.",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "oct",
            "poutcome": "unknown",
        }
        leakage_prone_features = {"duration": "79"}
        self.record = BankMarketingRecord(
            row_id=1,
            label=1,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            leakage_prone_features=leakage_prone_features,
        )
        self.config = FeatureConfig(hash_dimension=1024)

    def test_dense_vector_builds_bank_marketing_features(self) -> None:
        dense_vector = build_dense_vector(self.record, self.config)

        self.assertEqual(len(dense_vector), 7)
        self.assertAlmostEqual(dense_vector[0], math.log1p(30.0))
        self.assertAlmostEqual(dense_vector[1], -math.log1p(20.0))
        self.assertAlmostEqual(dense_vector[2], math.log1p(19.0))
        self.assertAlmostEqual(dense_vector[3], math.log1p(3.0))
        self.assertAlmostEqual(dense_vector[4], 0.0)
        self.assertAlmostEqual(dense_vector[5], math.log1p(2.0))
        self.assertEqual(dense_vector[6], 0.0)

    def test_sparse_vector_keeps_all_categorical_slots(self) -> None:
        feature_ids, feature_values = build_sparse_vector(self.record, self.config)

        self.assertEqual(len(feature_ids), 9)
        self.assertEqual(len(feature_values), 9)
        self.assertTrue(all(value == 1.0 for value in feature_values))
        self.assertTrue(all(0 <= feature_id < 1024 for feature_id in feature_ids))

    def test_hashing_is_stable(self) -> None:
        first = hash_feature_token("job", "admin.", 4096)
        second = hash_feature_token("job", "admin.", 4096)
        third = hash_feature_token("job", "blue-collar", 4096)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
