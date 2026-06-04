from __future__ import annotations

import unittest

from tests import _bootstrap  # noqa: F401

from probabilistic_decisioning.metrics import classification_metric_report


class MetricsTest(unittest.TestCase):
    def test_ranking_and_calibration_metrics_are_reasonable(self) -> None:
        report = classification_metric_report(
            y_true=[0, 1, 0, 1],
            y_score=[0.1, 0.9, 0.2, 0.8],
            n_bins=4,
        )

        self.assertAlmostEqual(report["roc_auc"], 1.0)
        self.assertAlmostEqual(report["pr_auc"], 1.0)
        self.assertLess(report["log_loss"], 0.25)
        self.assertLess(report["ece"], 0.25)
        self.assertEqual(report["example_count"], 4)
        self.assertTrue(report["calibration_bins"])
