"""CLI entrypoint for the Phase 5 online serving API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from probabilistic_decisioning.serving import build_runtime, create_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the Phase 5 online serving API.")
    parser.add_argument("--model-path", required=True, help="Path to the Phase 3 model.json artifact.")
    parser.add_argument("--calibration-path", required=True, help="Path to the Phase 4 calibration.json artifact.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-dir", help="Optional directory for prediction and decision JSONL logs.")
    parser.add_argument("--prediction-log-name", default="prediction_log.jsonl")
    parser.add_argument("--decision-log-name", default="decision_log.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir) if args.log_dir else None
    prediction_log_path = None
    decision_log_path = None
    if log_dir is not None:
        prediction_log_path = log_dir / args.prediction_log_name
        decision_log_path = log_dir / args.decision_log_name

    runtime = build_runtime(
        model_path=Path(args.model_path),
        calibration_path=Path(args.calibration_path),
        prediction_log_path=prediction_log_path,
        decision_log_path=decision_log_path,
    )
    server = create_server(args.host, args.port, runtime)
    print(
        f"Serving Phase 5 API at http://{args.host}:{args.port} "
        f"(model={runtime.bundle.model_version}, calibration={runtime.bundle.calibration_artifact.calibration_version})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
