"""Run the Phase 6 monitoring job against sample or full artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from probabilistic_decisioning.monitoring import MonitoringInputs, run_monitoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 6 monitoring job.")
    parser.add_argument("--mode", choices=("sample", "full"), required=True)
    parser.add_argument("--monitoring-config", default=str(ROOT / "configs" / "monitoring.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--group-key", default="task_context")
    parser.add_argument("--training-data-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--phase3-metrics")
    parser.add_argument("--phase4-report")
    parser.add_argument("--prediction-log")
    parser.add_argument("--decision-log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = _resolve_bundle(args.mode)
    training_data_dir = Path(args.training_data_dir) if args.training_data_dir else bundle["training_data_dir"]
    model_path = Path(args.model_path) if args.model_path else bundle["model_path"]
    phase3_metrics_path = Path(args.phase3_metrics) if args.phase3_metrics else bundle["phase3_metrics_path"]
    phase4_report_path = Path(args.phase4_report) if args.phase4_report else bundle["phase4_report_path"]
    prediction_log_path = Path(args.prediction_log) if args.prediction_log else bundle["prediction_log_path"]
    decision_log_path = Path(args.decision_log) if args.decision_log else bundle["decision_log_path"]
    output_dir = Path(args.output_dir) if args.output_dir else bundle["output_dir"]
    result = run_monitoring(
        MonitoringInputs(
            model_path=model_path,
            training_data_dir=training_data_dir,
            prediction_log_path=prediction_log_path,
            decision_log_path=decision_log_path,
            phase3_metrics_path=phase3_metrics_path,
            phase4_report_path=phase4_report_path,
            output_dir=output_dir,
            monitoring_config_path=Path(args.monitoring_config),
            group_key=args.group_key,
        )
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "report_path": str(result.report_path),
                "alerts_path": str(result.alerts_path),
                "rollout_ready": bool(result.report.get("rollout_readiness", {}).get("ready")),
                "alert_count": len(result.report.get("alerts", [])),
            },
            indent=2,
        )
    )
    return 0


def _resolve_bundle(mode: str) -> dict[str, Path]:
    if mode == "sample":
        return {
            "training_data_dir": ROOT / "artifacts" / "bank_marketing_smoke_cv",
            "model_path": ROOT / "artifacts" / "bank_marketing_lr_cv" / "model.json",
            "phase3_metrics_path": ROOT / "artifacts" / "bank_marketing_lr_cv" / "metrics.json",
            "phase4_report_path": ROOT
            / "artifacts"
            / "bank_marketing_phase4_cv"
            / "policies"
            / "calibration_and_policy_report.json",
            "prediction_log_path": ROOT
            / "artifacts"
            / "bank_marketing_phase5_logs_sample"
            / "prediction_log.jsonl",
            "decision_log_path": ROOT
            / "artifacts"
            / "bank_marketing_phase5_logs_sample"
            / "decision_log.jsonl",
            "output_dir": ROOT / "artifacts" / "bank_marketing_phase6_sample",
        }
    return {
        "training_data_dir": ROOT / "artifacts" / "bank_marketing_full",
        "model_path": ROOT / "artifacts" / "bank_marketing_lr_full" / "model.json",
        "phase3_metrics_path": ROOT / "artifacts" / "bank_marketing_lr_full" / "metrics.json",
        "phase4_report_path": ROOT
        / "artifacts"
        / "bank_marketing_phase4_full"
        / "policies"
        / "calibration_and_policy_report.json",
        "prediction_log_path": ROOT / "artifacts" / "bank_marketing_phase5_logs_full" / "prediction_log.jsonl",
        "decision_log_path": ROOT / "artifacts" / "bank_marketing_phase5_logs_full" / "decision_log.jsonl",
        "output_dir": ROOT / "artifacts" / "bank_marketing_phase6_full",
    }


if __name__ == "__main__":
    raise SystemExit(main())
