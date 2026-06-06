from __future__ import annotations

import json
import unittest
from pathlib import Path
from uuid import uuid4

from tests import _bootstrap  # noqa: F401

from probabilistic_decisioning.calibration import (
    fit_calibration_artifact,
    load_selected_calibrator,
    save_calibration_artifact,
)
from probabilistic_decisioning.dataset_builder import DatasetBuilderConfig, build_dataset
from probabilistic_decisioning.hybrid_ranking import (
    HYBRID_RERANKER_DENSE_FEATURE_NAMES,
    HybridRankingConfig,
    HybridRankingInputs,
    run_hybrid_ranking,
)
from probabilistic_decisioning.logistic_regression import (
    LogisticRegressionTrainingConfig,
    load_model_artifact,
    load_training_splits,
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


class HybridRankingTest(unittest.TestCase):
    def setUp(self) -> None:
        WORKSPACE_TEMP_ROOT.mkdir(exist_ok=True)

    def _build_phase_bundle(self, name: str) -> dict[str, Path]:
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
            model_version="phase7_unit_test_lr_v1",
            hash_dimension=256,
        )
        model_paths = save_training_artifacts(training_result, tmp_path / "model")

        calibration_artifact = fit_calibration_artifact(
            model=training_result.model,
            validation_examples=splits["validation"],
            candidate_methods=("platt_scaling", "isotonic_regression"),
            selection_metric="validation_ece",
            calibration_version="phase7_unit_test_calibration_v1",
        )
        calibration_path = save_calibration_artifact(
            calibration_artifact,
            tmp_path / "calibration" / "calibration.json",
        )

        return {
            "training_data_dir": artifacts_dir,
            "model_path": model_paths["model_path"],
            "calibration_path": calibration_path,
            "output_dir": tmp_path / "phase7_outputs",
        }

    def test_hybrid_ranking_pipeline_writes_report_and_ranked_candidates(self) -> None:
        bundle = self._build_phase_bundle("test_hybrid_ranking_pipeline_writes_report_and_ranked_candidates")
        result = run_hybrid_ranking(
            HybridRankingInputs(
                training_data_dir=bundle["training_data_dir"],
                model_path=bundle["model_path"],
                calibration_path=bundle["calibration_path"],
                output_dir=bundle["output_dir"],
            ),
            config=HybridRankingConfig(
                top_k=2,
                exploration_rate=0.5,
                reranker_max_epochs=5,
                reranker_early_stopping_patience=1,
                reranker_seed=19,
            ),
        )

        self.assertTrue(result.report_path.exists())
        self.assertTrue(result.reranker_artifacts["model_path"].exists())
        self.assertTrue(result.ranked_candidate_paths["validation"].exists())
        self.assertIn("reranker", result.report)
        self.assertIn("overall", result.report)
        self.assertIn("rollout_readiness", result.report)
        self.assertEqual(
            result.report["reranker"]["dense_feature_names"],
            list(HYBRID_RERANKER_DENSE_FEATURE_NAMES),
        )
        validation_report = result.report["splits"]["validation"]
        ranking_metrics = validation_report["ranking_metrics"]
        self.assertEqual(ranking_metrics["top_k"], 2)
        self.assertEqual(ranking_metrics["exploration_slots"], 1)
        self.assertEqual(len(ranking_metrics["selected_candidate_ids"]), 2)
        self.assertGreaterEqual(ranking_metrics["final_top_k_positive_rate"], 0.0)
        self.assertLessEqual(ranking_metrics["final_top_k_positive_rate"], 1.0)

        preview_lines = result.ranked_candidate_paths["validation"].read_text(encoding="utf-8").strip().splitlines()
        self.assertGreater(len(preview_lines), 0)
        preview_record = json.loads(preview_lines[0])
        self.assertIn("hybrid_features", preview_record)
        self.assertIn("was_explored", preview_record)

    def test_hybrid_reranker_model_can_be_reloaded(self) -> None:
        bundle = self._build_phase_bundle("test_hybrid_reranker_model_can_be_reloaded")
        result = run_hybrid_ranking(
            HybridRankingInputs(
                training_data_dir=bundle["training_data_dir"],
                model_path=bundle["model_path"],
                calibration_path=bundle["calibration_path"],
                output_dir=bundle["output_dir"],
            )
        )

        reranker_model = load_model_artifact(result.reranker_artifacts["model_path"])
        self.assertEqual(reranker_model.dense_feature_names, HYBRID_RERANKER_DENSE_FEATURE_NAMES)
        self.assertTrue(isinstance(result.report["rollout_readiness"]["ready"], bool))
        self.assertIn("validation_reranker_lift_vs_stage1", result.report["rollout_readiness"])


if __name__ == "__main__":
    unittest.main()
