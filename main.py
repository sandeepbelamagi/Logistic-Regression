"""Single-entry orchestrator for all project phases and the dashboard."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


@dataclass(frozen=True)
class RunPaths:
    mode: str
    input_path: Path
    phase2_dir: Path
    phase3_dir: Path
    phase4_dir: Path
    phase5_dir: Path
    phase6_dir: Path
    phase7_dir: Path

    @property
    def model_path(self) -> Path:
        return self.phase3_dir / "model.json"

    @property
    def metrics_path(self) -> Path:
        return self.phase3_dir / "metrics.json"

    @property
    def phase4_report_path(self) -> Path:
        return self.phase4_dir / "policies" / "calibration_and_policy_report.json"

    @property
    def calibration_path(self) -> Path:
        return self.phase4_dir / "calibration" / "calibration.json"

    @property
    def prediction_log_path(self) -> Path:
        return self.phase5_dir / "prediction_log.jsonl"

    @property
    def decision_log_path(self) -> Path:
        return self.phase5_dir / "decision_log.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phases 2-7 end-to-end and optionally launch the Streamlit dashboard."
    )
    parser.add_argument("--mode", choices=("sample", "full"), default="full")
    parser.add_argument(
        "--input-path",
        help="Override the raw Bank Marketing CSV path. Defaults to the mode-appropriate dataset.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Root directory for generated artifacts.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip launching the Streamlit dashboard after the pipeline finishes.",
    )
    parser.add_argument("--dashboard-port", type=int, default=8501)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args.mode, Path(args.artifacts_root), Path(args.input_path) if args.input_path else None)

    print(f"Running full pipeline in {args.mode!r} mode", flush=True)
    print(f"Raw input: {paths.input_path}", flush=True)
    print(f"Artifacts root: {Path(args.artifacts_root).resolve()}", flush=True)

    run_phase2(paths)
    run_phase3(paths)
    run_phase4(paths)
    run_phase5(paths)
    run_phase6(paths)
    run_phase7(paths)

    print("\nPipeline complete.", flush=True)
    print("Key outputs:", flush=True)
    print(f"- Phase 2: {paths.phase2_dir}", flush=True)
    print(f"- Phase 3: {paths.phase3_dir}", flush=True)
    print(f"- Phase 4: {paths.phase4_dir}", flush=True)
    print(f"- Phase 5: {paths.phase5_dir}", flush=True)
    print(f"- Phase 6: {paths.phase6_dir}", flush=True)
    print(f"- Phase 7: {paths.phase7_dir}", flush=True)

    if not args.no_dashboard:
        launch_dashboard(args.dashboard_port)
    else:
        print("\nDashboard skipped. To open it later:", flush=True)
        print("streamlit run ui/streamlit_app.py", flush=True)

    return 0


def resolve_paths(mode: str, artifacts_root: Path, input_override: Path | None) -> RunPaths:
    if mode == "sample":
        input_path = input_override or ROOT / "samples" / "bank_full_smoke.csv"
        phase2_dir = artifacts_root / "bank_marketing_smoke_cv"
        phase3_dir = artifacts_root / "bank_marketing_lr_cv"
        phase4_dir = artifacts_root / "bank_marketing_phase4_cv"
        phase5_dir = artifacts_root / "bank_marketing_phase5_logs_sample"
        phase6_dir = artifacts_root / "bank_marketing_phase6_sample"
        phase7_dir = artifacts_root / "bank_marketing_phase7_sample"
    else:
        input_path = input_override or ROOT / "data" / "raw" / "bank_marketing" / "bank-full.csv"
        phase2_dir = artifacts_root / "bank_marketing_full"
        phase3_dir = artifacts_root / "bank_marketing_lr_full"
        phase4_dir = artifacts_root / "bank_marketing_phase4_full"
        phase5_dir = artifacts_root / "bank_marketing_phase5_logs_full"
        phase6_dir = artifacts_root / "bank_marketing_phase6_full"
        phase7_dir = artifacts_root / "bank_marketing_phase7_full"

    return RunPaths(
        mode=mode,
        input_path=input_path,
        phase2_dir=phase2_dir,
        phase3_dir=phase3_dir,
        phase4_dir=phase4_dir,
        phase5_dir=phase5_dir,
        phase6_dir=phase6_dir,
        phase7_dir=phase7_dir,
    )


def run_phase2(paths: RunPaths) -> None:
    command = [
        PYTHON,
        "pipelines/build_training_dataset.py",
        "--input",
        str(paths.input_path),
        "--output-dir",
        str(paths.phase2_dir),
    ]
    if paths.mode == "sample":
        command += [
            "--split-strategy",
            "contiguous",
            "--train-ratio",
            "0.34",
            "--validation-ratio",
            "0.33",
            "--test-ratio",
            "0.33",
            "--max-rows",
            "3",
        ]
    run_command("Phase 2 · dataset build", command)


def run_phase3(paths: RunPaths) -> None:
    command = [
        PYTHON,
        "pipelines/train_logistic_regression.py",
        "--data-dir",
        str(paths.phase2_dir),
        "--output-dir",
        str(paths.phase3_dir),
    ]
    run_command("Phase 3 · model training", command)


def run_phase4(paths: RunPaths) -> None:
    command = [
        PYTHON,
        "pipelines/calibrate_and_route.py",
        "--data-dir",
        str(paths.phase2_dir),
        "--model-path",
        str(paths.model_path),
        "--output-dir",
        str(paths.phase4_dir),
    ]
    run_command("Phase 4 · calibration and policy", command)


def run_phase5(paths: RunPaths) -> None:
    command = [
        PYTHON,
        "scripts/phase5_runner.py",
        "--mode",
        paths.mode,
        "--model-path",
        str(paths.model_path),
        "--calibration-path",
        str(paths.calibration_path),
        "--log-dir",
        str(paths.phase5_dir),
    ]
    run_command("Phase 5 · serving and logs", command)


def run_phase6(paths: RunPaths) -> None:
    command = [
        PYTHON,
        "scripts/phase6_runner.py",
        "--mode",
        paths.mode,
        "--training-data-dir",
        str(paths.phase2_dir),
        "--model-path",
        str(paths.model_path),
        "--phase3-metrics",
        str(paths.metrics_path),
        "--phase4-report",
        str(paths.phase4_report_path),
        "--prediction-log",
        str(paths.prediction_log_path),
        "--decision-log",
        str(paths.decision_log_path),
        "--output-dir",
        str(paths.phase6_dir),
    ]
    run_command("Phase 6 · monitoring", command)


def run_phase7(paths: RunPaths) -> None:
    command = [
        PYTHON,
        "scripts/phase7_runner.py",
        "--mode",
        paths.mode,
        "--data-dir",
        str(paths.phase2_dir),
        "--model-path",
        str(paths.model_path),
        "--calibration-path",
        str(paths.calibration_path),
        "--output-dir",
        str(paths.phase7_dir),
    ]
    run_command("Phase 7 · hybrid ranking", command)


def run_command(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(format_command(command), flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{label} failed with exit code {exc.returncode}.") from exc


def launch_dashboard(port: int) -> None:
    print("\n=== Dashboard ===", flush=True)
    streamlit_command = [
        PYTHON,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "ui" / "streamlit_app.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    if not _streamlit_available():
        print("Streamlit is not installed. Install the dashboard extra first:", flush=True)
        print("pip install -e '.[ui]'", flush=True)
        return

    print(format_command(streamlit_command), flush=True)
    subprocess.Popen(streamlit_command, cwd=ROOT)
    print(f"Dashboard launched at http://127.0.0.1:{port}", flush=True)


def _streamlit_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("streamlit") is not None
    except Exception:
        return False


def format_command(command: list[str]) -> str:
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline(command)
    return shlex.join(command)


if __name__ == "__main__":
    raise SystemExit(main())
