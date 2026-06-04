from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401

from probabilistic_decisioning.calibration import (
    evaluate_calibrated_model,
    fit_calibration_artifact,
    fit_isotonic_regressor,
    load_selected_calibrator,
    save_calibration_artifact,
)
from probabilistic_decisioning.dataset_builder import DatasetBuilderConfig, build_dataset
from probabilistic_decisioning.decision_policy import (
    evaluate_decision_policy,
    route_decision,
    save_decision_outcomes,
    save_decision_summary,
)
from probabilistic_decisioning.logistic_regression import (
    LogisticRegressionTrainingConfig,
    load_training_splits,
    train_logistic_regression,
)

WORKSPACE_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"


def _sample_row(label: str, seed: int) -> str:
    values = [
        str(25 + seed),
        "admin." if seed % 2 == 0 else "technician",
        "married" if seed % 3 == 0 else "single",
        "secondary",
        "no",
        str(1000 + seed * 10),
        "yes" if seed % 2 == 0 else "no",
        "no",
        "cellular",
        str(10 + seed),
        "oct",
        str(60 + seed),
        str(1 + seed % 3),
        "-1" if seed % 2 == 0 else "42",
        str(seed % 4),
        "unknown",
        label,
    ]
    return ";".join(values)


class CalibrationAndPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        WORKSPACE_TEMP_ROOT.mkdir(exist_ok=True)

    def _build_phase2_artifacts(self, name: str) -> Path:
        tmp_path = WORKSPACE_TEMP_ROOT / name
        tmp_path.mkdir(parents=True, exist_ok=True)
        input_path = tmp_path / "bank-full.csv"
        input_path.write_text(
            "\n".join(
                [
                    _sample_row("no", 1),
                    _sample_row("no", 2),
                    _sample_row("yes", 3),
                    _sample_row("no", 4),
                    _sample_row("no", 5),
                    _sample_row("yes", 6),
                    _sample_row("no", 7),
                    _sample_row("no", 8),
                    _sample_row("yes", 9),
                    _sample_row("no", 10),
                    _sample_row("no", 11),
                    _sample_row("yes", 12),
                ]
            ),
            encoding="utf-8",
        )

        build_dataset(
            DatasetBuilderConfig(
                input_path=input_path,
                output_dir=tmp_path / "artifacts",
                hash_dimension=256,
                split_strategy="contiguous",
                train_ratio=0.5,
                validation_ratio=0.25,
                test_ratio=0.25,
            )
        )
        return tmp_path / "artifacts"

    def test_isotonic_calibrator_is_monotonic(self) -> None:
        calibrator = fit_isotonic_regressor(
            raw_scores=[0.1, 0.2, 0.3, 0.4],
            labels=[0, 0, 1, 1],
        )
        predictions = [calibrator.predict(score) for score in [0.05, 0.15, 0.25, 0.35, 0.45]]

        self.assertTrue(all(first <= second for first, second in zip(predictions, predictions[1:])))
        self.assertTrue(all(0.0 <= prediction <= 1.0 for prediction in predictions))

    def test_threshold_routing_covers_all_phase4_policies(self) -> None:
        bank_low = route_decision("bank_marketing", 0.05, event_id="evt_1", event_ts="2026-01-01T00:00:00Z")
        bank_mid = route_decision("bank_marketing", 0.10, event_id="evt_2", event_ts="2026-01-01T00:00:01Z")
        bank_high = route_decision("bank_marketing", 0.30, event_id="evt_3", event_ts="2026-01-01T00:00:02Z")
        fraud_approve = route_decision("fraud_policy", 0.10, event_id="evt_4", event_ts="2026-01-01T00:00:03Z")
        fraud_review = route_decision("fraud_policy", 0.20, event_id="evt_5", event_ts="2026-01-01T00:00:04Z")
        fraud_block = route_decision("fraud_policy", 0.80, event_id="evt_6", event_ts="2026-01-01T00:00:05Z")
        loan_approve = route_decision("loan_policy", 0.10, event_id="evt_7", event_ts="2026-01-01T00:00:06Z")
        loan_review = route_decision("loan_policy", 0.15, event_id="evt_8", event_ts="2026-01-01T00:00:07Z")
        loan_decline = route_decision("loan_policy", 0.25, event_id="evt_9", event_ts="2026-01-01T00:00:08Z")

        self.assertEqual(bank_low.action, "suppress")
        self.assertEqual(bank_mid.action, "nurture")
        self.assertEqual(bank_high.action, "prioritize_contact")
        self.assertEqual(fraud_approve.action, "approve")
        self.assertEqual(fraud_review.action, "review")
        self.assertEqual(fraud_block.action, "block")
        self.assertEqual(loan_approve.action, "approve")
        self.assertEqual(loan_review.action, "review")
        self.assertEqual(loan_decline.action, "decline")
        self.assertTrue(fraud_review.manual_review_required)
        self.assertTrue(fraud_block.manual_review_required)
        self.assertTrue(loan_review.manual_review_required)
        self.assertTrue(loan_decline.manual_review_required)

    def test_phase4_pipeline_fits_calibration_and_writes_reports(self) -> None:
        data_dir = self._build_phase2_artifacts("test_phase4_pipeline_fits_calibration_and_writes_reports")
        splits = load_training_splits(data_dir)
        training_result = train_logistic_regression(
            splits["train"],
            splits["validation"],
            splits["test"],
            LogisticRegressionTrainingConfig(
                learning_rate=0.1,
                max_epochs=4,
                l2=0.0001,
                early_stopping_patience=1,
                seed=17,
            ),
            model_version="phase4_unit_test_lr_v1",
            hash_dimension=256,
        )

        artifact = fit_calibration_artifact(
            model=training_result.model,
            validation_examples=splits["validation"],
            candidate_methods=("platt_scaling", "isotonic_regression"),
            selection_metric="validation_ece",
            calibration_version="phase4_unit_test_calibration_v1",
        )
        calibrator = load_selected_calibrator(artifact)
        validation_report = evaluate_calibrated_model(training_result.model, calibrator, splits["validation"])
        test_report = evaluate_calibrated_model(training_result.model, calibrator, splits["test"])

        validation_probabilities = [
            calibrator.predict(training_result.model.score(example))
            for example in splits["validation"]
        ]
        validation_outcomes, validation_summary = evaluate_decision_policy(
            task_context="bank_marketing",
            calibrated_probabilities=validation_probabilities,
            event_ids=[example.event_id for example in splits["validation"]],
            event_timestamps=[example.event_ts for example in splits["validation"]],
            labels=[example.label for example in splits["validation"]],
        )

        temp_output_dir = data_dir / "phase4_outputs"
        calibration_path = save_calibration_artifact(artifact, temp_output_dir / "calibration" / "calibration.json")
        outcomes_path = save_decision_outcomes(
            validation_outcomes,
            temp_output_dir / "decision_outcomes" / "validation_bank_marketing.jsonl",
        )
        summary_path = save_decision_summary(
            validation_summary,
            temp_output_dir / "policies" / "validation_bank_marketing_summary.json",
        )

        self.assertTrue(calibration_path.exists())
        self.assertTrue(outcomes_path.exists())
        self.assertTrue(summary_path.exists())
        self.assertEqual(validation_report["example_count"], len(splits["validation"]))
        self.assertEqual(test_report["example_count"], len(splits["test"]))
        self.assertEqual(validation_summary["decision_count"], len(splits["validation"]))
        self.assertIn(artifact.selected_method, {"platt_scaling", "isotonic_regression"})
        self.assertGreaterEqual(validation_summary["manual_review_rate"], 0.0)
        self.assertLessEqual(validation_summary["manual_review_rate"], 1.0)

        saved_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_payload["selected_method"], artifact.selected_method)
