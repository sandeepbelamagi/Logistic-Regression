"""CLI entrypoint for the Phase 4 calibration and decision engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from probabilistic_decisioning.calibration import (
    evaluate_calibrated_model,
    fit_calibration_artifact,
    load_selected_calibrator,
    save_calibration_artifact,
)
from probabilistic_decisioning.decision_policy import (
    evaluate_decision_policy,
    save_decision_outcomes,
    save_decision_summary,
)
from probabilistic_decisioning.logistic_regression import load_model_artifact, load_training_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit calibration and generate decision-policy reports.")
    parser.add_argument("--data-dir", required=True, help="Phase 2 artifact directory containing the splits.")
    parser.add_argument("--model-path", required=True, help="Path to the Phase 3 model.json artifact.")
    parser.add_argument("--output-dir", required=True, help="Directory for calibration and policy artifacts.")
    parser.add_argument("--calibration-version", default="bank_marketing_calibration_v1")
    parser.add_argument(
        "--candidate-methods",
        nargs="+",
        default=["platt_scaling", "isotonic_regression"],
        help="Calibration methods to evaluate.",
    )
    parser.add_argument("--selection-metric", default="validation_ece")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--decision-contexts",
        nargs="+",
        default=["bank_marketing", "fraud_policy", "loan_policy"],
        help="Policy contexts to simulate on the calibrated probabilities.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = load_training_splits(Path(args.data_dir))
    model = load_model_artifact(Path(args.model_path))

    if not splits["validation"]:
        raise ValueError("Phase 4 calibration requires a non-empty validation split.")

    output_dir = Path(args.output_dir)
    calibration_dir = output_dir / "calibration"
    policy_dir = output_dir / "policies"
    decision_dir = output_dir / "decision_outcomes"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)
    decision_dir.mkdir(parents=True, exist_ok=True)

    calibration_artifact = fit_calibration_artifact(
        model=model,
        validation_examples=splits["validation"],
        candidate_methods=args.candidate_methods,
        selection_metric=args.selection_metric,
        calibration_version=args.calibration_version,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )
    calibrator = load_selected_calibrator(calibration_artifact)

    calibration_artifact_path = save_calibration_artifact(
        calibration_artifact,
        calibration_dir / "calibration.json",
    )

    validation_evaluation = evaluate_calibrated_model(model, calibrator, splits["validation"])
    test_evaluation = evaluate_calibrated_model(model, calibrator, splits["test"])

    policy_reports: dict[str, dict[str, object]] = {}
    decision_files: dict[str, str] = {}
    for context in args.decision_contexts:
        context_report: dict[str, object] = {}
        for split_name in ("validation", "test"):
            examples = splits[split_name]
            calibrated_probabilities = [
                calibrator.predict(model.score(example)) for example in examples
            ]
            event_ids = [example.event_id for example in examples]
            event_timestamps = [example.event_ts for example in examples]
            labels = [example.label for example in examples]

            outcomes, summary = evaluate_decision_policy(
                task_context=context,
                calibrated_probabilities=calibrated_probabilities,
                event_ids=event_ids,
                event_timestamps=event_timestamps,
                labels=labels,
            )
            decision_path = decision_dir / f"{split_name}_{context}.jsonl"
            save_decision_outcomes(outcomes, decision_path)
            decision_files[f"{split_name}_{context}"] = str(decision_path)
            context_report[split_name] = summary
        policy_reports[context] = context_report

    report = {
        "model_version": model.model_version,
        "feature_set_version": model.feature_set_version,
        "task_context": model.task_context,
        "calibration_artifact_path": str(calibration_artifact_path),
        "selected_method": calibration_artifact.selected_method,
        "selection_metric": calibration_artifact.selection_metric,
        "validation_evaluation": validation_evaluation,
        "test_evaluation": test_evaluation,
        "policy_reports": policy_reports,
        "decision_files": decision_files,
    }

    report_path = save_decision_summary(report, policy_dir / "calibration_and_policy_report.json")
    print(
        json.dumps(
            {
                "calibration_artifact": str(calibration_artifact_path),
                "report": str(report_path),
                "decision_dir": str(decision_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
