from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401

from probabilistic_decisioning.dataset_builder import DatasetBuilderConfig, build_dataset
from probabilistic_decisioning.logistic_regression import (
    LogisticRegressionTrainingConfig,
    load_model_artifact,
    load_training_splits,
    run_experiment_suite,
    save_training_artifacts,
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


class LogisticRegressionTrainingTest(unittest.TestCase):
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

    def test_training_pipeline_trains_and_exports_model(self) -> None:
        data_dir = self._build_phase2_artifacts("test_training_pipeline_trains_and_exports_model")
        splits = load_training_splits(data_dir)
        config = LogisticRegressionTrainingConfig(
            learning_rate=0.1,
            max_epochs=6,
            l2=0.0001,
            early_stopping_patience=2,
            seed=21,
        )

        result = train_logistic_regression(
            splits["train"],
            splits["validation"],
            splits["test"],
            config,
            model_version="unit_test_lr_v1",
            hash_dimension=256,
        )

        self.assertEqual(result.dataset_summary["train_rows"], 5)
        self.assertEqual(result.validation_metrics["example_count"], 3)
        self.assertEqual(len(result.model.dense_weights), 7)
        self.assertGreater(result.class_weights["1"], result.class_weights["0"])
        self.assertGreaterEqual(result.validation_metrics["roc_auc"], 0.0)
        self.assertLessEqual(result.validation_metrics["roc_auc"], 1.0)

        output_dir = data_dir / "phase3_outputs"
        artifact_paths = save_training_artifacts(result, output_dir)
        self.assertTrue(artifact_paths["model_path"].exists())
        self.assertTrue(artifact_paths["metrics_path"].exists())
        self.assertTrue(artifact_paths["history_path"].exists())

        loaded_model = load_model_artifact(artifact_paths["model_path"])
        self.assertEqual(loaded_model.model_version, "unit_test_lr_v1")
        first_validation_example = splits["validation"][0]
        self.assertAlmostEqual(
            loaded_model.predict_proba(first_validation_example),
            result.model.predict_proba(first_validation_example),
        )

    def test_experiment_suite_runs_ablation_and_stability_reports(self) -> None:
        data_dir = self._build_phase2_artifacts("test_experiment_suite_runs_ablation_and_stability_reports")
        splits = load_training_splits(data_dir)
        config = LogisticRegressionTrainingConfig(
            learning_rate=0.1,
            max_epochs=4,
            l2=0.0001,
            early_stopping_patience=1,
            seed=7,
        )

        report = run_experiment_suite(
            splits,
            base_config=config,
            model_version="suite_unit_test_lr_v1",
            hash_dimension=256,
            stability_repeats=2,
        )

        self.assertEqual(report["model_version"], "suite_unit_test_lr_v1")
        self.assertIn("baseline", report["ablation_runs"])
        self.assertIn("mse_loss", report["ablation_runs"])
        self.assertIn("oversampling", report["ablation_runs"])
        self.assertIn("no_class_weighting", report["ablation_runs"])
        self.assertIn("no_regularization", report["ablation_runs"])
        self.assertIsNotNone(report["stability_report"])
        self.assertEqual(report["stability_report"]["repeat_count"], 2)
