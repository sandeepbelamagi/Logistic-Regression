"""CLI entrypoint for the Phase 7 hybrid reranking pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from probabilistic_decisioning.hybrid_ranking import (  # noqa: E402
    HybridRankingConfig,
    HybridRankingInputs,
    run_hybrid_ranking,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 7 hybrid reranking and exploration job.")
    parser.add_argument("--data-dir", required=True, help="Phase 2 artifact directory containing the splits.")
    parser.add_argument("--model-path", required=True, help="Path to the Phase 3 model.json artifact.")
    parser.add_argument("--calibration-path", required=True, help="Path to the Phase 4 calibration.json artifact.")
    parser.add_argument("--output-dir", required=True, help="Directory for the Phase 7 artifacts.")
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
    _validate_inputs(
        data_dir=Path(args.data_dir),
        model_path=Path(args.model_path),
        calibration_path=Path(args.calibration_path),
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
    result = run_hybrid_ranking(
        HybridRankingInputs(
            training_data_dir=Path(args.data_dir),
            model_path=Path(args.model_path),
            calibration_path=Path(args.calibration_path),
            output_dir=Path(args.output_dir),
        ),
        config=config,
    )
    print(
        json.dumps(
            {
                "report_path": str(result.report_path),
                "reranker_model_path": str(result.reranker_artifacts["model_path"]),
                "ranked_candidate_paths": {key: str(value) for key, value in result.ranked_candidate_paths.items()},
                "rollout_ready": bool(result.report.get("rollout_readiness", {}).get("ready")),
            },
            indent=2,
        )
    )
    return 0


def _validate_inputs(*, data_dir: Path, model_path: Path, calibration_path: Path) -> None:
    required_split_paths = [
        data_dir / "training" / "train.jsonl",
        data_dir / "training" / "validation.jsonl",
        data_dir / "training" / "test.jsonl",
    ]
    missing_paths = [path for path in required_split_paths if not path.exists()]
    if missing_paths:
        missing_display = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Phase 7 --data-dir must point to the Phase 2 artifact directory containing training splits. "
            f"Missing: {missing_display}"
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Phase 7 requires an existing Phase 3 model.json artifact. Missing: {model_path}")
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Phase 7 requires an existing Phase 4 calibration.json artifact. Missing: {calibration_path}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
