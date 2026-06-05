"""Run the Phase 5 serving stack against sample or full artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from probabilistic_decisioning.serving import build_runtime, create_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 5 local server and smoke calls.")
    parser.add_argument("--mode", choices=("sample", "full"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--calibration-path", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-timeout-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = build_runtime(
        model_path=Path(args.model_path),
        calibration_path=Path(args.calibration_path),
        prediction_log_path=Path(args.log_dir) / "prediction_log.jsonl",
        decision_log_path=Path(args.log_dir) / "decision_log.jsonl",
    )
    server = create_server(args.host, args.port, runtime)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    actual_host, actual_port = server.server_address
    connect_host = "127.0.0.1" if actual_host in {"0.0.0.0", "", "::"} else actual_host
    base_url = f"http://{connect_host}:{actual_port}"

    try:
        _wait_for_health(base_url, args.startup_timeout_seconds)
        print(f"MODE: {args.mode}")
        print("HEALTH")
        _print_json(_get_json(f"{base_url}/health"))

        predict_body = _sample_predict_payload()
        decision_body = _sample_decision_payload()

        print("\nFEATURE LOOKUP")
        _print_json(_post_json(f"{base_url}/v1/features/lookup", predict_body))

        print("\nPREDICT")
        _print_json(_post_json(f"{base_url}/v1/predict", predict_body))

        print("\nDECISION")
        _print_json(_post_json(f"{base_url}/v1/decision", decision_body))

        print("\nLATEST LOGS")
        _print_tail(Path(args.log_dir) / "prediction_log.jsonl")
        _print_tail(Path(args.log_dir) / "decision_log.jsonl")
        return 0
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


def _wait_for_health(base_url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            health = _get_json(f"{base_url}/health")
            if health.get("status") == "ok":
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1)
    if last_error is None:
        raise RuntimeError("Phase 5 server did not become ready.")
    raise RuntimeError("Phase 5 server did not become ready.") from last_error


def _sample_predict_payload() -> dict[str, Any]:
    return {
        "request_id": "req_1",
        "event_id": "evt_1",
        "event_ts": "2026-01-01T00:00:00Z",
        "task_context": "bank_marketing",
        "features": {
            "age": "42",
            "job": "admin.",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "balance": "1200",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "day": "15",
            "month": "oct",
            "duration": "80",
            "campaign": "2",
            "pdays": "-1",
            "previous": "0",
            "poutcome": "unknown",
        },
    }


def _sample_decision_payload() -> dict[str, Any]:
    return {
        "request_id": "req_2",
        "event_id": "evt_2",
        "event_ts": "2026-01-01T00:00:00Z",
        "task_context": "fraud_policy",
        "calibrated_probability": 0.8,
    }


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _print_tail(path: Path, tail_lines: int = 2) -> None:
    if not path.exists():
        print(f"{path} not found")
        return
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        print(f"{path} is empty")
        return
    for line in lines[-tail_lines:]:
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
