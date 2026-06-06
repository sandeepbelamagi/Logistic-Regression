"""Run the Phase 7 hybrid reranking stack against sample or full artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from probabilistic_decisioning.hybrid_ranking import HybridRankingConfig, HybridRankingInputs, run_hybrid_ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 7 hybrid reranking job.")
    parser.add_argument("--mode", choices=("sample", "full"), required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--calibration-path")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--exploration-rate", type=float, default=0.15)
    parser.add_argument("--reranker-model-version", default="bank_marketing_hybrid_reranker_v1")
    parser.add_argument("--reranker-feature-set-version", default="bank_marketing_hybrid_v1")
    parser.add_argument("--reranker-task-context", default="hybrid_ranking")
    parser.add_argument("--reranker-hash-dimension", type=int, default=262_144)
    parser.add_argument("--reranker-learning-rate", type=float, default=0.05)
    parser.add_argument("--reranker-max-epochs", type=int, default=10)
    parser.add_argument("--reranker-l2", type=float, default=0.0005)
    parser.add_argument("--reranker-seed", type=int, default=31)
    parser.add_argument("--reranker-early-stopping-patience", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = _resolve_bundle(args.mode)
    inputs = HybridRankingInputs(
        training_data_dir=Path(args.data_dir) if args.data_dir else bundle["training_data_dir"],
        model_path=Path(args.model_path) if args.model_path else bundle["model_path"],
        calibration_path=Path(args.calibration_path) if args.calibration_path else bundle["calibration_path"],
        output_dir=Path(args.output_dir) if args.output_dir else bundle["output_dir"],
    )
    config = HybridRankingConfig(
        top_k=args.top_k,
        exploration_rate=args.exploration_rate,
        reranker_model_version=args.reranker_model_version,
        reranker_feature_set_version=args.reranker_feature_set_version,
        reranker_task_context=args.reranker_task_context,
        reranker_hash_dimension=args.reranker_hash_dimension,
        reranker_learning_rate=args.reranker_learning_rate,
        reranker_max_epochs=args.reranker_max_epochs,
        reranker_l2=args.reranker_l2,
        reranker_seed=args.reranker_seed,
        reranker_early_stopping_patience=args.reranker_early_stopping_patience,
    )
    result = run_hybrid_ranking(inputs, config=config)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "report_path": str(result.report_path),
                "reranker_model_path": str(result.reranker_artifacts["model_path"]),
                "rollout_ready": bool(result.report.get("rollout_readiness", {}).get("ready")),
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
            "calibration_path": ROOT / "artifacts" / "bank_marketing_phase4_cv" / "calibration" / "calibration.json",
            "output_dir": ROOT / "artifacts" / "bank_marketing_phase7_sample",
        }
    return {
        "training_data_dir": ROOT / "artifacts" / "bank_marketing_full",
        "model_path": ROOT / "artifacts" / "bank_marketing_lr_full" / "model.json",
        "calibration_path": ROOT / "artifacts" / "bank_marketing_phase4_full" / "calibration" / "calibration.json",
        "output_dir": ROOT / "artifacts" / "bank_marketing_phase7_full",
    }


if __name__ == "__main__":
    raise SystemExit(main())
