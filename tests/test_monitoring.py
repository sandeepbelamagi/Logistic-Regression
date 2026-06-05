from __future__ import annotations

import json
import unittest
from pathlib import Path
from uuid import uuid4

from tests import _bootstrap  # noqa: F401

from probabilistic_decisioning.calibration import (
    evaluate_calibrated_model,
    fit_calibration_artifact,
    load_selected_calibrator,
    save_calibration_artifact,
)
from probabilistic_decisioning.dataset_builder import DatasetBuilderConfig, build_dataset
from probabilistic_decisioning.logistic_regression import (
    LogisticRegressionTrainingConfig,
    load_training_splits,
    save_training_artifacts,
    train_logistic_regression,
)
from probabilistic_decisioning.monitoring import (
    MonitoringInputs,
    MonitoringThresholds,
    load_monitoring_thresholds,
    run_monitoring,
)
from probabilistic_decisioning.serving import ServingRequest, build_runtime


WORKSPACE_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"
ROOT = Path(__file__).resolve().parents[1]


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


def _sample_predict_payload(seed: int, task_context: str = "bank_marketing") -> dict[str, object]:
    return {
        "request_id": f"req_{seed}",
        "event_id": f"evt_{seed}",
        "event_ts": "2026-01-01T00:00:00Z",
        "task_context": task_context,
        "realized_label": 1,
        "realized_value": 1.0,
        "features": {
            "age": str(35 + seed),
            "job": "admin." if seed % 2 == 0 else "technician",
            "marital": "married" if seed % 3 == 0 else "single",
            "education": "secondary",
            "default": "no",
            "balance": str(1200 + seed * 15),
            "housing": "yes" if seed % 2 == 0 else "no",
            "loan": "no",
            "contact": "cellular",
            "day": str(12 + seed),
            "month": "oct",
            "duration": str(80 + seed),
            "campaign": str(1 + seed % 3),
            "pdays": "-1" if seed % 2 == 0 else "42",
            "previous": str(seed % 4),
            "poutcome": "unknown",
        },
    }


def _sample_decision_payload(seed: int, calibrated_probability: float, task_context: str) -> dict[str, object]:
    return {
        "request_id": f"req_route_{seed}",
        "event_id": f"evt_route_{seed}",
        "event_ts": "2026-01-01T00:00:00Z",
        "task_context": task_context,
        "calibrated_probability": calibrated_probability,
        "realized_label": 0,
        "realized_value": 0.0,
    }


class MonitoringTest(unittest.TestCase):
    def setUp(self) -> None:
        WORKSPACE_TEMP_ROOT.mkdir(exist_ok=True)

    def _build_monitoring_artifacts(self, name: str) -> dict[str, Path]:
        tmp_path = WORKSPACE_TEMP_ROOT / f"{name}_{uuid4().hex}"
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

        artifacts_dir = tmp_path / "artifacts"
        build_dataset(
            DatasetBuilderConfig(
                input_path=input_path,
                output_dir=artifacts_dir,
                hash_dimension=256,
                split_strategy="contiguous",
                train_ratio=0.5,
                validation_ratio=0.25,
                test_ratio=0.25,
            )
        )
        splits = load_training_splits(artifacts_dir)
        training_result = train_logistic_regression(
            splits["train"],
            splits["validation"],
            splits["test"],
            LogisticRegressionTrainingConfig(
                learning_rate=0.1,
                max_epochs=5,
                l2=0.0001,
                early_stopping_patience=2,
                seed=17,
            ),
            model_version="phase6_unit_test_lr_v1",
            hash_dimension=256,
        )
        model_artifacts = save_training_artifacts(training_result, tmp_path / "model")

        calibration_artifact = fit_calibration_artifact(
            model=training_result.model,
            validation_examples=splits["validation"],
            candidate_methods=("platt_scaling", "isotonic_regression"),
            selection_metric="validation_ece",
            calibration_version="phase6_unit_test_calibration_v1",
        )
        calibration_path = save_calibration_artifact(
            calibration_artifact,
            tmp_path / "calibration" / "calibration.json",
        )

        validation_report = evaluate_calibrated_model(
            training_result.model,
            load_selected_calibrator(calibration_artifact),
            splits["validation"],
        )
        test_report = evaluate_calibrated_model(
            training_result.model,
            load_selected_calibrator(calibration_artifact),
            splits["test"],
        )
        phase4_report = {
            "model_version": training_result.model.model_version,
            "feature_set_version": training_result.model.feature_set_version,
            "task_context": training_result.model.task_context,
            "selected_method": calibration_artifact.selected_method,
            "selection_metric": calibration_artifact.selection_metric,
            "validation_evaluation": validation_report,
            "test_evaluation": test_report,
        }
        phase4_report_path = tmp_path / "phase4" / "policies" / "calibration_and_policy_report.json"
        phase4_report_path.parent.mkdir(parents=True, exist_ok=True)
        phase4_report_path.write_text(json.dumps(phase4_report, indent=2, sort_keys=True), encoding="utf-8")

        runtime = build_runtime(
            model_path=model_artifacts["model_path"],
            calibration_path=calibration_path,
            prediction_log_path=tmp_path / "phase5_logs" / "prediction_log.jsonl",
            decision_log_path=tmp_path / "phase5_logs" / "decision_log.jsonl",
        )
        runtime.predict(ServingRequest.from_payload(_sample_predict_payload(1)))
        runtime.route(
            ServingRequest.from_payload(
                _sample_decision_payload(2, calibrated_probability=0.8, task_context="fraud_policy")
            )
        )

        return {
            "training_data_dir": artifacts_dir,
            "model_path": model_artifacts["model_path"],
            "phase3_metrics_path": model_artifacts["metrics_path"],
            "phase4_report_path": phase4_report_path,
            "prediction_log_path": tmp_path / "phase5_logs" / "prediction_log.jsonl",
            "decision_log_path": tmp_path / "phase5_logs" / "decision_log.jsonl",
        }

    def test_monitoring_report_captures_drift_and_linkage(self) -> None:
        artifacts = self._build_monitoring_artifacts("test_monitoring_report_captures_drift_and_linkage")
        thresholds = load_monitoring_thresholds(ROOT / "configs" / "monitoring.yaml")
        self.assertEqual(thresholds.subgroup_calibration_gap_critical, 0.05)
        self.assertEqual(thresholds.approval_rate_gap_critical, 0.05)

        result = run_monitoring(
            MonitoringInputs(
                model_path=artifacts["model_path"],
                training_data_dir=artifacts["training_data_dir"],
                prediction_log_path=artifacts["prediction_log_path"],
                decision_log_path=artifacts["decision_log_path"],
                phase3_metrics_path=artifacts["phase3_metrics_path"],
                phase4_report_path=artifacts["phase4_report_path"],
                output_dir=artifacts["training_data_dir"] / "monitoring_outputs",
                monitoring_config_path=ROOT / "configs" / "monitoring.yaml",
            ),
            thresholds=thresholds,
        )

        report = result.report
        self.assertTrue(result.report_path.exists())
        self.assertTrue(result.alerts_path.exists())
        self.assertIn("summary", report)
        self.assertIn("feature_drift", report)
        self.assertIn("governance", report)
        self.assertIn("calibration_metrics", report)
        self.assertEqual(report["summary"]["prediction_count"], 1)
        self.assertEqual(report["summary"]["decision_count"], 2)
        self.assertEqual(report["summary"]["prediction_link_rate"], 1.0)
        self.assertEqual(report["governance"]["audit_log_coverage"], 1.0)
        self.assertGreaterEqual(report["feature_drift"]["max_dense_psi"], 0.0)
        self.assertIn("live_primary_calibrated_metrics", report["calibration_metrics"])
        self.assertIn("bank_marketing", report["governance"]["group_metrics"])
        self.assertGreaterEqual(len(report["model"]["dense_feature_names"]), 1)

    def test_monitoring_alerts_with_strict_thresholds(self) -> None:
        artifacts = self._build_monitoring_artifacts("test_monitoring_alerts_with_strict_thresholds")

        result = run_monitoring(
            MonitoringInputs(
                model_path=artifacts["model_path"],
                training_data_dir=artifacts["training_data_dir"],
                prediction_log_path=artifacts["prediction_log_path"],
                decision_log_path=artifacts["decision_log_path"],
                phase3_metrics_path=artifacts["phase3_metrics_path"],
                phase4_report_path=artifacts["phase4_report_path"],
                output_dir=artifacts["training_data_dir"] / "monitoring_outputs_strict",
                monitoring_config_path=ROOT / "configs" / "monitoring.yaml",
            ),
            thresholds=MonitoringThresholds(
                ece_critical=-1.0,
                psi_critical=-1.0,
                prediction_p95_latency_ms_critical=-1.0,
                manual_review_utilization_critical=-1.0,
                subgroup_calibration_gap_critical=-1.0,
                approval_rate_gap_critical=-1.0,
            ),
        )

        self.assertGreaterEqual(len(result.report["alerts"]), 1)
        self.assertFalse(result.report["rollout_readiness"]["ready"])


if __name__ == "__main__":
    unittest.main()
