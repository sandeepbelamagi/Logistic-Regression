from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401

from probabilistic_decisioning.dataset_builder import DatasetBuilderConfig, build_dataset

WORKSPACE_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"


def _sample_row(label: str, seed: int) -> str:
    dense_values = [str(seed + index) for index in range(1, 14)]
    categorical_values = [f"cat{seed}_{index}" for index in range(1, 27)]
    return "\t".join([label, *dense_values, *categorical_values])


class DatasetBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        WORKSPACE_TEMP_ROOT.mkdir(exist_ok=True)

    def test_build_dataset_emits_expected_artifacts(self) -> None:
        tmp_path = WORKSPACE_TEMP_ROOT / "test_build_dataset_emits_expected_artifacts"
        self._reset_dir(tmp_path)
        input_path = tmp_path / "train.txt"
        output_path = tmp_path / "artifacts"
        input_path.write_text(
            "\n".join(
                [
                    _sample_row("1", 10),
                    _sample_row("0", 20),
                    _sample_row("1", 30),
                ]
            ),
            encoding="utf-8",
        )

        summary_path = build_dataset(
            DatasetBuilderConfig(
                input_path=input_path,
                output_dir=output_path,
                hash_dimension=256,
                split_strategy="contiguous",
                train_ratio=0.5,
                validation_ratio=0.25,
                test_ratio=0.25,
            )
        )

        self.assertTrue(summary_path.exists())
        raw_events = (output_path / "raw" / "raw_impression_events.jsonl").read_text(
            encoding="utf-8"
        ).strip().splitlines()
        raw_labels = (output_path / "raw" / "raw_click_labels.jsonl").read_text(
            encoding="utf-8"
        ).strip().splitlines()
        train_rows = (output_path / "training" / "train.jsonl").read_text(
            encoding="utf-8"
        ).strip().splitlines()
        validation_rows = (
            output_path / "training" / "validation.jsonl"
        ).read_text(encoding="utf-8").strip().splitlines()
        test_rows = (output_path / "training" / "test.jsonl").read_text(
            encoding="utf-8"
        ).strip().splitlines()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(len(raw_events), 3)
        self.assertEqual(len(raw_labels), 3)
        self.assertEqual(len(train_rows), 1)
        self.assertEqual(len(validation_rows), 1)
        self.assertEqual(len(test_rows), 1)
        self.assertEqual(summary["processed_rows"], 3)
        self.assertAlmostEqual(summary["positive_rate"], 2 / 3)

        example_training_row = json.loads(train_rows[0])
        self.assertEqual(example_training_row["task_context"], "ctr")
        self.assertEqual(len(example_training_row["dense_features"]), 13)
        self.assertEqual(len(example_training_row["sparse_feature_ids"]), 26)

    def test_ratio_validation_rejects_invalid_config(self) -> None:
        tmp_path = WORKSPACE_TEMP_ROOT / "test_ratio_validation_rejects_invalid_config"
        self._reset_dir(tmp_path)
        input_path = tmp_path / "train.txt"
        input_path.write_text(_sample_row("1", 10), encoding="utf-8")

        with self.assertRaises(ValueError):
            build_dataset(
                DatasetBuilderConfig(
                    input_path=input_path,
                    output_dir=tmp_path / "artifacts",
                    train_ratio=0.7,
                    validation_ratio=0.2,
                    test_ratio=0.2,
                )
            )

    def _reset_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
