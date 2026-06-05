"""CLI entrypoint for the Phase 6 monitoring and governance job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from probabilistic_decisioning.monitoring import (  # noqa: E402
    MonitoringInputs,
    load_and_write_monitoring_report,
    load_monitoring_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 6 monitoring and governance job.")
    parser.add_argument("--model-path", required=True, help="Path to the Phase 3 model.json artifact.")
    parser.add_argument(
        "--training-data-dir",
        required=True,
        help="Path to the Phase 2 artifact directory containing training splits.",
    )
    parser.add_argument(
        "--prediction-log",
        required=True,
        help="Path to the Phase 5 prediction_log.jsonl artifact.",
    )
    parser.add_argument(
        "--decision-log",
        required=True,
        help="Path to the Phase 5 decision_log.jsonl artifact.",
    )
    parser.add_argument(
        "--phase3-metrics",
        required=True,
        help="Path to the Phase 3 metrics.json artifact.",
    )
    parser.add_argument(
        "--phase4-report",
        required=True,
        help="Path to the Phase 4 calibration_and_policy_report.json artifact.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for monitoring outputs.")
    parser.add_argument(
        "--monitoring-config",
        default=str(ROOT / "configs" / "monitoring.yaml"),
        help="Optional monitoring configuration file with alert thresholds.",
    )
    parser.add_argument("--group-key", default="task_context", help="Decision field used for subgroup analysis.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path)
    training_data_dir = Path(args.training_data_dir)
    prediction_log_path = Path(args.prediction_log)
    decision_log_path = Path(args.decision_log)
    phase3_metrics_path = Path(args.phase3_metrics)
    phase4_report_path = Path(args.phase4_report)
    output_dir = Path(args.output_dir)
    monitoring_config_path = Path(args.monitoring_config) if args.monitoring_config else None

    _validate_inputs(
        model_path=model_path,
        training_data_dir=training_data_dir,
        prediction_log_path=prediction_log_path,
        decision_log_path=decision_log_path,
        phase3_metrics_path=phase3_metrics_path,
        phase4_report_path=phase4_report_path,
    )

    thresholds = load_monitoring_thresholds(monitoring_config_path)
    result = load_and_write_monitoring_report(
        MonitoringInputs(
            model_path=model_path,
            training_data_dir=training_data_dir,
            prediction_log_path=prediction_log_path,
            decision_log_path=decision_log_path,
            phase3_metrics_path=phase3_metrics_path,
            phase4_report_path=phase4_report_path,
            output_dir=output_dir,
            monitoring_config_path=monitoring_config_path,
            group_key=args.group_key,
        ),
        thresholds=thresholds,
    )
    print(
        json.dumps(
            {
                "report_path": str(result.report_path),
                "alerts_path": str(result.alerts_path),
                "rollout_ready": bool(result.report.get("rollout_readiness", {}).get("ready")),
                "alert_count": len(result.report.get("alerts", [])),
            },
            indent=2,
        )
    )
    return 0


def _validate_inputs(
    *,
    model_path: Path,
    training_data_dir: Path,
    prediction_log_path: Path,
    decision_log_path: Path,
    phase3_metrics_path: Path,
    phase4_report_path: Path,
) -> None:
    required_split_paths = [
        training_data_dir / "training" / "train.jsonl",
        training_data_dir / "training" / "validation.jsonl",
        training_data_dir / "training" / "test.jsonl",
    ]
    missing_paths = [path for path in required_split_paths if not path.exists()]
    if missing_paths:
        missing_display = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Phase 6 --training-data-dir must point to the Phase 2 artifact directory "
            f"containing training splits. Missing: {missing_display}"
        )
    for path, label in [
        (model_path, "model"),
        (prediction_log_path, "prediction log"),
        (decision_log_path, "decision log"),
        (phase3_metrics_path, "Phase 3 metrics"),
        (phase4_report_path, "Phase 4 report"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Phase 6 requires an existing {label} artifact. Missing: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
