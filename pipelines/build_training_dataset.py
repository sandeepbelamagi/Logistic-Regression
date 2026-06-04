"""CLI entrypoint for the Phase 2 offline dataset builder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from probabilistic_decisioning.dataset_builder import DatasetBuilderConfig, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 2 raw and training artifacts.")
    parser.add_argument("--input", required=True, help="Path to a local Bank Marketing CSV file.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated artifacts.")
    parser.add_argument("--hash-dimension", type=int, default=1_048_576)
    parser.add_argument("--feature-set-version", default="bank_marketing_v1")
    parser.add_argument("--task-context", default="bank_marketing")
    parser.add_argument("--split-strategy", choices=("hash", "contiguous"), default="hash")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--label-delay-seconds", type=int, default=0)
    parser.add_argument("--start-timestamp", default="2026-01-01T00:00:00+00:00")
    parser.add_argument("--seconds-per-row", type=int, default=1)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--source-name", default="bank_full_csv")
    parser.add_argument("--attribution-window-hours", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DatasetBuilderConfig(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        hash_dimension=args.hash_dimension,
        feature_set_version=args.feature_set_version,
        task_context=args.task_context,
        split_strategy=args.split_strategy,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        label_delay_seconds=args.label_delay_seconds,
        start_timestamp=args.start_timestamp,
        seconds_per_row=args.seconds_per_row,
        max_rows=args.max_rows,
        source_name=args.source_name,
        attribution_window_hours=args.attribution_window_hours,
    )
    summary_path = build_dataset(config)
    print(f"Dataset build complete: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
