"""CLI entrypoint for the Phase 3 Logistic Regression trainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from probabilistic_decisioning.logistic_regression import (
    LogisticRegressionTrainingConfig,
    load_training_splits,
    run_experiment_suite,
    save_experiment_report,
    save_training_artifacts,
    train_logistic_regression,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 3 Logistic Regression baseline.")
    parser.add_argument("--data-dir", required=True, help="Phase 2 artifact directory containing training splits.")
    parser.add_argument("--output-dir", required=True, help="Directory for model and evaluation artifacts.")
    parser.add_argument("--model-version", default="bank_marketing_lr_v1")
    parser.add_argument("--feature-set-version", default="bank_marketing_v1")
    parser.add_argument("--task-context", default="bank_marketing")
    parser.add_argument("--hash-dimension", type=int, default=1_048_576)
    parser.add_argument("--loss", choices=("cross_entropy", "mse"), default="cross_entropy")
    parser.add_argument("--optimizer", choices=("sgd", "adagrad"), default="adagrad")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--no-class-weighting", action="store_true")
    parser.add_argument("--oversampling", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--run-suite", action="store_true")
    parser.add_argument("--stability-repeats", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = load_training_splits(Path(args.data_dir))
    training_config = LogisticRegressionTrainingConfig(
        loss=args.loss,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        l1=args.l1,
        l2=args.l2,
        class_weighting=not args.no_class_weighting,
        oversampling=args.oversampling,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_suite:
        report = run_experiment_suite(
            splits,
            base_config=training_config,
            model_version=args.model_version,
            feature_set_version=args.feature_set_version,
            task_context=args.task_context,
            hash_dimension=args.hash_dimension,
            stability_repeats=args.stability_repeats,
        )
        report_path = save_experiment_report(report, output_dir)
        print(f"Phase 3 experiment suite complete: {report_path}")
        return 0

    result = train_logistic_regression(
        splits["train"],
        splits["validation"],
        splits["test"],
        training_config,
        model_version=args.model_version,
        feature_set_version=args.feature_set_version,
        task_context=args.task_context,
        hash_dimension=args.hash_dimension,
    )
    artifact_paths = save_training_artifacts(result, output_dir)
    print(json.dumps({key: str(value) for key, value in artifact_paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
