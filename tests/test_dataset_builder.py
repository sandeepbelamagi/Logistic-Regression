from __future__ import annotations

import json
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401

from probabilistic_decisioning.dataset_builder import DatasetBuilderConfig, build_dataset

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


class DatasetBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        WORKSPACE_TEMP_ROOT.mkdir(exist_ok=True)

    def test_build_dataset_emits_expected_artifacts(self) -> None:
        tmp_path = WORKSPACE_TEMP_ROOT / "test_build_dataset_emits_expected_artifacts"
        tmp_path.mkdir(parents=True, exist_ok=True)
        input_path = tmp_path / "bank-full.csv"
        output_path = tmp_path / "artifacts"
        input_path.write_text(
            "\n".join(
                [
                    _sample_row("yes", 1),
                    _sample_row("no", 2),
                    _sample_row("yes", 3),
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
        raw_events = (output_path / "raw" / "raw_contact_events.jsonl").read_text(
            encoding="utf-8"
        ).strip().splitlines()
        raw_labels = (output_path / "raw" / "raw_subscription_labels.jsonl").read_text(
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
        self.assertEqual(example_training_row["task_context"], "bank_marketing")
        self.assertEqual(len(example_training_row["dense_features"]), 7)
        self.assertEqual(len(example_training_row["sparse_feature_ids"]), 9)

    def test_ratio_validation_rejects_invalid_config(self) -> None:
        tmp_path = WORKSPACE_TEMP_ROOT / "test_ratio_validation_rejects_invalid_config"
        tmp_path.mkdir(parents=True, exist_ok=True)
        input_path = tmp_path / "bank-full.csv"
        input_path.write_text(_sample_row("yes", 10), encoding="utf-8")

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
