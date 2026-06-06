"""Streamlit dashboard for the Probabilistic Decisioning Platform."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from probabilistic_decisioning.bank_marketing import iter_bank_marketing_records
from probabilistic_decisioning.constants import (
    BANK_MARKETING_CATEGORICAL_FEATURE_NAMES,
    BANK_MARKETING_DENSE_FEATURE_NAMES,
    BANK_MARKETING_NUMERIC_FEATURE_NAMES,
    BANK_MARKETING_LEAKAGE_FEATURE_NAME,
)
from probabilistic_decisioning.features import FeatureConfig, transform_dense_value, transform_prior_contact_flag
from probabilistic_decisioning.logistic_regression import load_model_artifact, load_training_splits


st.set_page_config(
    page_title="Probabilistic Decisioning Dashboard",
    page_icon="📈",
    layout="wide",
)


PALETTE = [
    "#2563eb",
    "#16a34a",
    "#f97316",
    "#7c3aed",
    "#dc2626",
    "#0891b2",
]


def main() -> None:
    st.title("Probabilistic Decisioning Platform")
    st.caption("Phase-by-phase artifact explorer, preprocessing visualizer, and run-quality dashboard.")

    preset, artifacts_root, raw_csv_path = render_sidebar()
    paths = resolve_paths(preset, artifacts_root, raw_csv_path)
    phase_statuses = collect_phase_statuses(paths)
    missing_paths = [str(path) for status in phase_statuses for path in status["missing"]]

    if missing_paths:
        st.warning(
            "Some expected artifacts are missing. The dashboard still renders any available outputs and lists the gaps below."
        )

    render_overview(phase_statuses, paths, missing_paths)

    tabs = st.tabs(
        [
            "Overview",
            "Phase 2 · Data Prep",
            "Phase 3 · Training",
            "Phase 4 · Calibration",
            "Phase 5 · Serving",
            "Phase 6 · Monitoring",
            "Phase 7 · Hybrid Ranking",
        ]
    )

    with tabs[0]:
        render_overview_tab(phase_statuses, paths, missing_paths)
    with tabs[1]:
        render_phase2_tab(paths)
    with tabs[2]:
        render_phase3_tab(paths)
    with tabs[3]:
        render_phase4_tab(paths)
    with tabs[4]:
        render_phase5_tab(paths)
    with tabs[5]:
        render_phase6_tab(paths)
    with tabs[6]:
        render_phase7_tab(paths)


def render_sidebar() -> tuple[str, Path, Path]:
    st.sidebar.header("Dashboard Controls")
    preset = st.sidebar.radio("Artifact preset", ["sample", "full"], index=0)
    artifacts_root = Path(st.sidebar.text_input("Artifacts root", value="artifacts")).expanduser()
    raw_default = (
        "samples/bank_full_smoke.csv"
        if preset == "sample"
        else "data/raw/bank_marketing/bank-full.csv"
    )
    raw_csv_path = Path(st.sidebar.text_input("Raw CSV path", value=raw_default)).expanduser()

    st.sidebar.markdown("### Derived artifact paths")
    for name, path in resolve_paths(preset, artifacts_root, raw_csv_path).items():
        st.sidebar.code(f"{name}: {path}")

    st.sidebar.caption("Switch presets to inspect sample or full outputs.")
    return preset, artifacts_root, raw_csv_path


def resolve_paths(preset: str, artifacts_root: Path, raw_csv_path: Path) -> dict[str, Path]:
    suffix = "sample" if preset == "sample" else "full"
    phase2_dir = artifacts_root / ("bank_marketing_smoke_cv" if preset == "sample" else "bank_marketing_full")
    phase3_dir = artifacts_root / f"bank_marketing_lr_{'cv' if preset == 'sample' else 'full'}"
    phase4_dir = artifacts_root / f"bank_marketing_phase4_{'cv' if preset == 'sample' else 'full'}"
    phase5_dir = artifacts_root / f"bank_marketing_phase5_logs_{suffix}"
    phase6_dir = artifacts_root / f"bank_marketing_phase6_{suffix}"
    phase7_dir = artifacts_root / f"bank_marketing_phase7_{suffix}"
    return {
        "raw_csv_path": raw_csv_path,
        "phase2_dir": phase2_dir,
        "phase3_dir": phase3_dir,
        "phase4_dir": phase4_dir,
        "phase5_dir": phase5_dir,
        "phase6_dir": phase6_dir,
        "phase7_dir": phase7_dir,
    }


@st.cache_data(show_spinner=False)
def read_json(path_str: str) -> object | None:
    path = Path(path_str)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def read_jsonl(path_str: str, limit: int | None = None) -> list[dict[str, object]]:
    path = Path(path_str)
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def collect_phase_statuses(paths: dict[str, Path]) -> list[dict[str, object]]:
    phase_specs: list[tuple[str, list[Path]]] = [
        (
            "Phase 2",
            [
                paths["phase2_dir"] / "metadata" / "dataset_summary.json",
                paths["phase2_dir"] / "training" / "train.jsonl",
                paths["phase2_dir"] / "training" / "validation.jsonl",
                paths["phase2_dir"] / "training" / "test.jsonl",
                paths["phase2_dir"] / "raw" / "raw_contact_events.jsonl",
                paths["phase2_dir"] / "raw" / "raw_subscription_labels.jsonl",
            ],
        ),
        (
            "Phase 3",
            [
                paths["phase3_dir"] / "model.json",
                paths["phase3_dir"] / "metrics.json",
                paths["phase3_dir"] / "history.json",
            ],
        ),
        (
            "Phase 4",
            [
                paths["phase4_dir"] / "calibration" / "calibration.json",
                paths["phase4_dir"] / "policies" / "calibration_and_policy_report.json",
            ],
        ),
        (
            "Phase 5",
            [
                paths["phase5_dir"] / "prediction_log.jsonl",
                paths["phase5_dir"] / "decision_log.jsonl",
            ],
        ),
        (
            "Phase 6",
            [
                paths["phase6_dir"] / "monitoring_report.json",
                paths["phase6_dir"] / "alerts.json",
            ],
        ),
        (
            "Phase 7",
            [
                paths["phase7_dir"] / "hybrid_ranking_report.json",
                paths["phase7_dir"] / "reranker" / "model.json",
                paths["phase7_dir"] / "reranker" / "metrics.json",
                paths["phase7_dir"] / "reranker" / "history.json",
                paths["phase7_dir"] / "rankings" / "train_top_k.jsonl",
                paths["phase7_dir"] / "rankings" / "validation_top_k.jsonl",
                paths["phase7_dir"] / "rankings" / "test_top_k.jsonl",
            ],
        ),
    ]

    statuses: list[dict[str, object]] = []
    for phase_name, required_paths in phase_specs:
        missing = [path for path in required_paths if not path.exists()]
        statuses.append(
            {
                "phase": phase_name,
                "ready": not missing,
                "required_paths": required_paths,
                "missing": missing,
            }
        )
    return statuses


def render_overview(
    phase_statuses: list[dict[str, object]],
    paths: dict[str, Path],
    missing_paths: list[str],
) -> None:
    phase2_summary = load_phase2_artifacts(paths["phase2_dir"])
    phase3_summary = load_phase3_artifacts(paths["phase3_dir"])
    phase4_summary = load_phase4_artifacts(paths["phase4_dir"])
    phase5_summary = load_phase5_artifacts(paths["phase5_dir"])
    phase6_summary = load_phase6_artifacts(paths["phase6_dir"])
    phase7_summary = load_phase7_artifacts(paths["phase7_dir"])

    ready_count = sum(1 for status in phase_statuses if status["ready"])
    total_missing = len(missing_paths)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Phases ready", f"{ready_count}/{len(phase_statuses)}")
    metric_columns[1].metric("Missing files", str(total_missing))
    metric_columns[2].metric(
        "Phase 2 rows",
        _format_int(phase2_summary.get("dataset_summary", {}).get("processed_rows")) if phase2_summary else "—",
    )
    metric_columns[3].metric(
        "Phase 6 rollout",
        "Ready" if phase6_summary and phase6_summary.get("rollout_readiness", {}).get("ready") else "Review",
    )

    st.markdown(
        """
        This dashboard is read-only. It surfaces the artifact outputs from phases 2–7, highlights missing files,
        and adds preprocessing, quality-gate, and compatibility checks that are not obvious from the raw JSON alone.
        """
    )

    if missing_paths:
        with st.expander("Missing artifacts", expanded=True):
            for path in missing_paths:
                st.code(path)

    phase_cards = st.columns(3)
    for index, status in enumerate(phase_statuses):
        with phase_cards[index % len(phase_cards)]:
            ready = bool(status["ready"])
            st.markdown(
                f"""
                <div style="padding:0.75rem;border:1px solid #e5e7eb;border-radius:0.75rem;margin-bottom:0.75rem;">
                    <div style="font-weight:700;font-size:1.02rem;">{status['phase']}</div>
                    <div style="color:{'#16a34a' if ready else '#dc2626'};font-weight:600;">
                        {"Ready" if ready else "Missing artifacts"}
                    </div>
                    <div style="font-size:0.9rem;color:#475569;">
                        {len(status['required_paths'])} expected file(s)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if phase6_summary and phase6_summary.get("rollout_readiness"):
        rollout = phase6_summary["rollout_readiness"]
        st.info(
            f"Phase 6 rollout readiness: {'ready' if rollout.get('ready') else 'not ready'}"
        )
    if phase7_summary and phase7_summary.get("rollout_readiness"):
        rollout = phase7_summary["rollout_readiness"]
        st.info(
            f"Phase 7 rollout readiness: {'ready' if rollout.get('ready') else 'not ready'}"
        )


def render_overview_tab(
    phase_statuses: list[dict[str, object]],
    paths: dict[str, Path],
    missing_paths: list[str],
) -> None:
    st.subheader("End-to-end coverage")
    coverage_rows = [
        {
            "phase": status["phase"],
            "ready": "yes" if status["ready"] else "no",
            "missing_files": len(status["missing"]),
        }
        for status in phase_statuses
    ]
    st.table(coverage_rows)

    st.subheader("What this dashboard adds")
    st.markdown(
        "- raw-vs-engineered preprocessing comparison\n"
        "- artifact presence and compatibility checks\n"
        "- phase-level metric tables and charts\n"
        "- sample previews of logs, policies, alerts, and rankings"
    )

    if missing_paths:
        st.warning("Fix the missing artifacts first if you want every tab to render completely.")

    with st.expander("Regeneration hints"):
        st.markdown(
            """
            - Phase 2: `python pipelines/build_training_dataset.py --input data/raw/bank_marketing/bank-full.csv --output-dir artifacts/bank_marketing_full`
            - Phase 3: `python pipelines/train_logistic_regression.py --data-dir artifacts/bank_marketing_full --output-dir artifacts/bank_marketing_lr_full`
            - Phase 4: `python pipelines/calibrate_and_route.py --data-dir artifacts/bank_marketing_full --model-path artifacts/bank_marketing_lr_full/model.json --output-dir artifacts/bank_marketing_phase4_full`
            - Phase 5: `python pipelines/serve_prediction_api.py --model-path artifacts/bank_marketing_lr_full/model.json --calibration-path artifacts/bank_marketing_phase4_full/calibration/calibration.json --log-dir artifacts/bank_marketing_phase5_logs_full`
            - Phase 6: `python scripts/phase6_runner.py --mode full`
            - Phase 7: `python scripts/phase7_runner.py --mode full`
            """
        )

    st.subheader("Paths")
    st.json({key: str(value) for key, value in paths.items()})


@st.cache_data(show_spinner=False)
def load_raw_profile(csv_path_str: str) -> dict[str, object]:
    csv_path = Path(csv_path_str)
    if not csv_path.exists():
        return {"exists": False, "path": str(csv_path)}

    numeric_values: dict[str, list[float]] = {feature: [] for feature in BANK_MARKETING_NUMERIC_FEATURE_NAMES}
    numeric_values[BANK_MARKETING_LEAKAGE_FEATURE_NAME] = []
    categorical_counts: dict[str, Counter[str]] = {feature: Counter() for feature in BANK_MARKETING_CATEGORICAL_FEATURE_NAMES}
    label_counts: Counter[str] = Counter()
    row_count = 0
    missing_numeric_counts: Counter[str] = Counter()
    missing_categorical_counts: Counter[str] = Counter()

    for record in iter_bank_marketing_records(csv_path):
        row_count += 1
        label_counts["yes" if record.label else "no"] += 1
        for feature_name in BANK_MARKETING_NUMERIC_FEATURE_NAMES:
            raw_value = record.numeric_features.get(feature_name)
            if raw_value in (None, ""):
                missing_numeric_counts[feature_name] += 1
            else:
                numeric_values[feature_name].append(float(raw_value))
        duration_value = record.leakage_prone_features.get(BANK_MARKETING_LEAKAGE_FEATURE_NAME)
        if duration_value in (None, ""):
            missing_numeric_counts[BANK_MARKETING_LEAKAGE_FEATURE_NAME] += 1
        else:
            numeric_values[BANK_MARKETING_LEAKAGE_FEATURE_NAME].append(float(duration_value))
        for feature_name in BANK_MARKETING_CATEGORICAL_FEATURE_NAMES:
            raw_value = record.categorical_features.get(feature_name)
            token = raw_value if raw_value not in (None, "") else "__MISSING__"
            categorical_counts[feature_name][token] += 1
            if token == "__MISSING__":
                missing_categorical_counts[feature_name] += 1

    return {
        "exists": True,
        "path": str(csv_path),
        "row_count": row_count,
        "label_counts": dict(label_counts),
        "numeric_values": numeric_values,
        "categorical_counts": {feature: dict(counter) for feature, counter in categorical_counts.items()},
        "missing_numeric_counts": dict(missing_numeric_counts),
        "missing_categorical_counts": dict(missing_categorical_counts),
    }


def build_numeric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p05": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": float(len(ordered)),
        "mean": float(statistics.mean(ordered)),
        "median": float(statistics.median(ordered)),
        "p05": percentile(ordered, 5.0),
        "p95": percentile(ordered, 95.0),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def histogram(values: Sequence[float], bins: int = 12) -> list[tuple[str, float]]:
    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return []
    if bins <= 0:
        raise ValueError("bins must be positive.")
    minimum = min(numeric_values)
    maximum = max(numeric_values)
    if math.isclose(minimum, maximum):
        return [(f"{minimum:.2f}", float(len(numeric_values)))]
    width = (maximum - minimum) / bins
    counts = [0 for _ in range(bins)]
    for value in numeric_values:
        index = min(int((value - minimum) / width), bins - 1)
        counts[index] += 1
    labels = [
        f"{minimum + index * width:.1f}–{minimum + (index + 1) * width:.1f}"
        for index in range(bins)
    ]
    return list(zip(labels, [float(count) for count in counts], strict=True))


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def render_bar_chart(
    title: str,
    items: Sequence[tuple[str, float]],
    *,
    value_suffix: str = "",
    max_items: int | None = None,
    positive_color: str = "#2563eb",
    negative_color: str = "#dc2626",
) -> None:
    if max_items is not None:
        items = list(items)[:max_items]
    if not items:
        st.caption(f"{title}: no data")
        return
    max_value = max(abs(float(value)) for _, value in items) or 1.0
    st.markdown(f"**{title}**")
    for label, value in items:
        ratio = min(abs(float(value)) / max_value, 1.0)
        bar_color = positive_color if float(value) >= 0 else negative_color
        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns: 220px 1fr 90px;gap:0.5rem;align-items:center;margin:0.35rem 0;">
              <div style="font-size:0.9rem;overflow-wrap:anywhere;">{escape(str(label))}</div>
              <div style="background:#e5e7eb;border-radius:999px;height:12px;overflow:hidden;">
                <div style="width:{ratio * 100:.1f}%;height:12px;background:{bar_color};border-radius:999px;"></div>
              </div>
              <div style="font-family:monospace;text-align:right;">{float(value):+.3f}{value_suffix}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_line_chart(
    title: str,
    series_map: dict[str, Sequence[float]],
    labels: Sequence[str],
    *,
    height: int = 300,
) -> None:
    if not series_map:
        st.caption(f"{title}: no data")
        return
    series_items = [(name, [float(value) for value in values], PALETTE[index % len(PALETTE)]) for index, (name, values) in enumerate(series_map.items())]
    all_values = [value for _, values, _ in series_items for value in values]
    if not all_values:
        st.caption(f"{title}: no data")
        return
    min_value = min(all_values)
    max_value = max(all_values)
    if math.isclose(min_value, max_value):
        max_value = min_value + 1.0

    width = 960
    left_pad = 56
    right_pad = 20
    top_pad = 20
    bottom_pad = 50
    plot_width = width - left_pad - right_pad
    plot_height = height - top_pad - bottom_pad
    point_count = max(len(values) for _, values, _ in series_items)
    step_x = plot_width / max(point_count - 1, 1)

    def y_position(value: float) -> float:
        return top_pad + (max_value - value) / (max_value - min_value) * plot_height

    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{escape(title)}">',
        f'<line x1="{left_pad}" y1="{top_pad + plot_height}" x2="{width - right_pad}" y2="{top_pad + plot_height}" stroke="#cbd5e1" stroke-width="1" />',
        f'<line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{top_pad + plot_height}" stroke="#cbd5e1" stroke-width="1" />',
    ]

    for tick_index in range(5):
        y = top_pad + plot_height * (tick_index / 4)
        value = max_value - (max_value - min_value) * (tick_index / 4)
        svg_parts.append(
            f'<line x1="{left_pad}" y1="{y:.1f}" x2="{width - right_pad}" y2="{y:.1f}" stroke="#f1f5f9" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{left_pad - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="#64748b">{value:.3f}</text>'
        )

    if labels:
        for index, label in enumerate(labels[:point_count]):
            x = left_pad + index * step_x if point_count > 1 else left_pad + plot_width / 2
            svg_parts.append(
                f'<text x="{x:.1f}" y="{top_pad + plot_height + 18:.1f}" font-size="11" text-anchor="middle" fill="#64748b">{escape(str(label))}</text>'
            )

    for series_name, values, color in series_items:
        points = []
        for index, value in enumerate(values):
            x = left_pad + index * step_x if point_count > 1 else left_pad + plot_width / 2
            y = y_position(value)
            points.append(f"{x:.1f},{y:.1f}")
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}" />'
        )
        for index, value in enumerate(values):
            x = left_pad + index * step_x if point_count > 1 else left_pad + plot_width / 2
            y = y_position(value)
            svg_parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" stroke="#ffffff" stroke-width="1.5" />'
            )

    legend_items = []
    for index, (series_name, _, color) in enumerate(series_items):
        legend_items.append(
            f'<span style="display:inline-flex;align-items:center;gap:0.35rem;margin-right:0.9rem;">'
            f'<span style="width:0.7rem;height:0.7rem;border-radius:999px;background:{color};display:inline-block;"></span>'
            f'{escape(series_name)}</span>'
        )

    svg_parts.append("</svg>")
    st.markdown(f"**{title}**")
    st.markdown(
        f"""
        <div style="font-size:0.9rem;color:#64748b;margin-bottom:0.25rem;">{' '.join(legend_items)}</div>
        {''.join(svg_parts)}
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_phase2_artifacts(phase2_dir_str: str) -> dict[str, object]:
    phase2_dir = Path(phase2_dir_str)
    dataset_summary_path = phase2_dir / "metadata" / "dataset_summary.json"
    train_path = phase2_dir / "training" / "train.jsonl"
    validation_path = phase2_dir / "training" / "validation.jsonl"
    test_path = phase2_dir / "training" / "test.jsonl"

    if not (train_path.exists() and validation_path.exists() and test_path.exists()):
        return {"exists": False, "phase2_dir": str(phase2_dir)}

    splits = load_training_splits(phase2_dir)
    dataset_summary = read_json(str(dataset_summary_path)) if dataset_summary_path.exists() else {}
    split_summaries = {}
    previews = {}
    for split_name, examples in splits.items():
        split_summaries[split_name] = {
            "rows": len(examples),
            "positives": sum(example.label for example in examples),
            "positive_rate": (sum(example.label for example in examples) / len(examples)) if examples else 0.0,
            "dense_feature_count": len(examples[0].dense_features) if examples else 0,
            "sparse_feature_count": len(examples[0].sparse_feature_ids) if examples else 0,
        }
        previews[split_name] = [asdict(example) for example in examples[:3]]

    return {
        "exists": True,
        "phase2_dir": str(phase2_dir),
        "dataset_summary": dataset_summary,
        "splits": split_summaries,
        "examples": splits,
        "previews": previews,
    }


@st.cache_data(show_spinner=False)
def load_phase3_artifacts(phase3_dir_str: str) -> dict[str, object]:
    phase3_dir = Path(phase3_dir_str)
    model_path = phase3_dir / "model.json"
    metrics_path = phase3_dir / "metrics.json"
    history_path = phase3_dir / "history.json"
    experiment_path = phase3_dir / "experiment_report.json"

    if not (model_path.exists() and metrics_path.exists() and history_path.exists()):
        return {"exists": False, "phase3_dir": str(phase3_dir)}

    model = load_model_artifact(model_path)
    metrics = read_json(str(metrics_path)) or {}
    history = read_json(str(history_path)) or []
    experiment_report = read_json(str(experiment_path)) if experiment_path.exists() else None

    metric_rows = []
    for split_name in ("train", "validation", "test"):
        split_metrics = metrics.get(f"{split_name}_metrics", {})
        metric_rows.append(
            {
                "split": split_name,
                "log_loss": split_metrics.get("log_loss"),
                "roc_auc": split_metrics.get("roc_auc"),
                "pr_auc": split_metrics.get("pr_auc"),
                "brier_score": split_metrics.get("brier_score"),
                "ece": split_metrics.get("ece"),
                "mce": split_metrics.get("mce"),
                "positive_rate": split_metrics.get("positive_rate"),
                "mean_prediction": split_metrics.get("mean_prediction"),
            }
        )

    return {
        "exists": True,
        "phase3_dir": str(phase3_dir),
        "model": model,
        "metrics": metrics,
        "history": history,
        "experiment_report": experiment_report,
        "metric_rows": metric_rows,
        "dense_coefficients": model.top_dense_coefficients(10),
        "sparse_coefficients": model.top_sparse_coefficients(10),
    }


@st.cache_data(show_spinner=False)
def load_phase4_artifacts(phase4_dir_str: str) -> dict[str, object]:
    phase4_dir = Path(phase4_dir_str)
    calibration_path = phase4_dir / "calibration" / "calibration.json"
    report_path = phase4_dir / "policies" / "calibration_and_policy_report.json"
    decision_dir = phase4_dir / "decision_outcomes"

    if not (calibration_path.exists() and report_path.exists()):
        return {"exists": False, "phase4_dir": str(phase4_dir)}

    calibration = read_json(str(calibration_path)) or {}
    report = read_json(str(report_path)) or {}
    policy_reports = report.get("policy_reports", {})
    decision_previews = {}
    for decision_path in sorted(decision_dir.glob("*.jsonl")):
        decision_previews[decision_path.name] = read_jsonl(str(decision_path), limit=3)

    method_rows = []
    raw_validation_metrics = calibration.get("raw_validation_metrics", {})
    selected_validation_metrics = calibration.get("selected_validation_metrics", {})
    method_rows.append(_metric_row("raw", raw_validation_metrics))
    method_rows.append(_metric_row(f"selected:{calibration.get('selected_method')}", selected_validation_metrics))
    for method_name, method_payload in calibration.get("method_reports", {}).items():
        method_rows.append(_metric_row(method_name, method_payload.get("calibrated_metrics", {})))

    return {
        "exists": True,
        "phase4_dir": str(phase4_dir),
        "calibration": calibration,
        "report": report,
        "policy_reports": policy_reports,
        "decision_previews": decision_previews,
        "method_rows": method_rows,
        "raw_validation_metrics": raw_validation_metrics,
        "selected_validation_metrics": selected_validation_metrics,
    }


@st.cache_data(show_spinner=False)
def load_phase5_artifacts(phase5_dir_str: str) -> dict[str, object]:
    phase5_dir = Path(phase5_dir_str)
    prediction_path = phase5_dir / "prediction_log.jsonl"
    decision_path = phase5_dir / "decision_log.jsonl"

    if not (prediction_path.exists() and decision_path.exists()):
        return {"exists": False, "phase5_dir": str(phase5_dir)}

    prediction_logs = read_jsonl(str(prediction_path))
    decision_logs = read_jsonl(str(decision_path))
    prediction_ids = {str(record.get("prediction_id")) for record in prediction_logs if record.get("prediction_id") is not None}
    decision_ids = {str(record.get("prediction_id")) for record in decision_logs if record.get("prediction_id") is not None}
    linked_ids = prediction_ids.intersection(decision_ids)

    latencies = [float(record.get("latency_ms", 0.0)) for record in prediction_logs if record.get("latency_ms") is not None]
    latency_breakdown_fields = [
        "feature_lookup_latency_ms",
        "scoring_latency_ms",
        "calibration_latency_ms",
        "decision_latency_ms",
    ]
    latency_breakdown = {}
    for field_name in latency_breakdown_fields:
        field_values = [float(record.get(field_name, 0.0)) for record in prediction_logs if record.get(field_name) is not None]
        latency_breakdown[field_name] = {
            "p50": percentile(field_values, 50.0),
            "p95": percentile(field_values, 95.0),
            "p99": percentile(field_values, 99.0),
        }

    action_counts = Counter(str(record.get("action", "unknown")) for record in decision_logs)
    prediction_link_rate = len(linked_ids) / len(prediction_ids) if prediction_ids else 0.0
    decision_link_rate = len(linked_ids) / len(decision_ids) if decision_ids else 0.0

    return {
        "exists": True,
        "phase5_dir": str(phase5_dir),
        "prediction_logs": prediction_logs,
        "decision_logs": decision_logs,
        "prediction_count": len(prediction_logs),
        "decision_count": len(decision_logs),
        "latency_summary": {
            "p50": percentile(latencies, 50.0),
            "p95": percentile(latencies, 95.0),
            "p99": percentile(latencies, 99.0),
        },
        "latency_breakdown": latency_breakdown,
        "action_counts": dict(action_counts),
        "prediction_link_rate": prediction_link_rate,
        "decision_link_rate": decision_link_rate,
        "sample_prediction": prediction_logs[0] if prediction_logs else None,
        "sample_decision": decision_logs[0] if decision_logs else None,
        "sample_feature_vector": prediction_logs[0].get("feature_vector") if prediction_logs else None,
    }


@st.cache_data(show_spinner=False)
def load_phase6_artifacts(phase6_dir_str: str) -> dict[str, object]:
    phase6_dir = Path(phase6_dir_str)
    report_path = phase6_dir / "monitoring_report.json"
    alerts_path = phase6_dir / "alerts.json"

    if not (report_path.exists() and alerts_path.exists()):
        return {"exists": False, "phase6_dir": str(phase6_dir)}

    report = read_json(str(report_path)) or {}
    alerts = read_json(str(alerts_path)) or []
    feature_drift = report.get("feature_drift", {})
    governance = report.get("governance", {})
    summary = report.get("summary", {})
    calibration_metrics = report.get("calibration_metrics", {})

    dense_psi = feature_drift.get("dense_feature_psi", {})
    dense_psi_rows = sorted(dense_psi.items(), key=lambda item: float(item[1]), reverse=True)
    group_metrics = governance.get("group_metrics", {})

    return {
        "exists": True,
        "phase6_dir": str(phase6_dir),
        "report": report,
        "alerts": alerts,
        "summary": summary,
        "feature_drift": feature_drift,
        "governance": governance,
        "calibration_metrics": calibration_metrics,
        "dense_psi_rows": dense_psi_rows,
        "group_rows": [
            {"group": group_name, **group_payload} for group_name, group_payload in group_metrics.items()
        ],
        "rollout_readiness": report.get("rollout_readiness", {}),
    }


@st.cache_data(show_spinner=False)
def load_phase7_artifacts(phase7_dir_str: str) -> dict[str, object]:
    phase7_dir = Path(phase7_dir_str)
    report_path = phase7_dir / "hybrid_ranking_report.json"
    ranking_dir = phase7_dir / "rankings"
    reranker_history_path = phase7_dir / "reranker" / "history.json"
    reranker_model_path = phase7_dir / "reranker" / "model.json"
    reranker_metrics_path = phase7_dir / "reranker" / "metrics.json"

    if not report_path.exists():
        return {"exists": False, "phase7_dir": str(phase7_dir)}

    report = read_json(str(report_path)) or {}
    reranker_history = read_json(str(reranker_history_path)) if reranker_history_path.exists() else []
    reranker_model = read_json(str(reranker_model_path)) if reranker_model_path.exists() else None
    reranker_metrics = read_json(str(reranker_metrics_path)) if reranker_metrics_path.exists() else None
    rankings = {
        split_name: read_jsonl(str(ranking_dir / f"{split_name}_top_k.jsonl"), limit=10)
        for split_name in ("train", "validation", "test")
        if (ranking_dir / f"{split_name}_top_k.jsonl").exists()
    }

    split_rows = []
    for split_name, split_report in report.get("splits", {}).items():
        metrics = split_report.get("ranking_metrics", {})
        split_rows.append(
            {
                "split": split_name,
                "candidate_count": split_report.get("candidate_count"),
                "top_k": split_report.get("top_k"),
                "stage1_pos_rate": metrics.get("stage1_top_k_positive_rate"),
                "reranker_pos_rate": metrics.get("reranker_top_k_positive_rate"),
                "final_pos_rate": metrics.get("final_top_k_positive_rate"),
                "stage1_ndcg": metrics.get("stage1_ndcg_at_k"),
                "reranker_ndcg": metrics.get("reranker_ndcg_at_k"),
                "final_ndcg": metrics.get("final_ndcg_at_k"),
                "lift_vs_stage1": metrics.get("lift_vs_stage1"),
                "lift_vs_reranker": metrics.get("lift_vs_reranker"),
                "exploration_slots": metrics.get("exploration_slots"),
            }
        )

    return {
        "exists": True,
        "phase7_dir": str(phase7_dir),
        "report": report,
        "reranker_history": reranker_history,
        "reranker_model": reranker_model,
        "reranker_metrics": reranker_metrics,
        "rankings": rankings,
        "split_rows": split_rows,
        "rollout_readiness": report.get("rollout_readiness", {}),
        "overall": report.get("overall", {}),
        "configuration": report.get("configuration", {}),
    }


def _metric_row(name: str, metrics: dict[str, object]) -> dict[str, object]:
    return {
        "name": name,
        "log_loss": metrics.get("log_loss"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "brier_score": metrics.get("brier_score"),
        "ece": metrics.get("ece"),
        "mce": metrics.get("mce"),
    }


def render_phase2_tab(paths: dict[str, Path]) -> None:
    st.subheader("Phase 2 outputs")
    phase2 = load_phase2_artifacts(str(paths["phase2_dir"]))
    raw_profile = load_raw_profile(str(paths["raw_csv_path"]))

    if not phase2.get("exists"):
        st.error(f"Phase 2 artifacts are missing under `{paths['phase2_dir']}`.")
        st.code(
            "python pipelines/build_training_dataset.py --input "
            f"{paths['raw_csv_path']} --output-dir {paths['phase2_dir']}"
        )
        return

    dataset_summary = phase2.get("dataset_summary", {})
    splits = phase2.get("splits", {})

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", _format_int(dataset_summary.get("processed_rows") or splits.get("train", {}).get("rows")))
    metric_columns[1].metric("Positive rate", _format_percent(dataset_summary.get("positive_rate")))
    metric_columns[2].metric("Dense features", str(len(BANK_MARKETING_DENSE_FEATURE_NAMES)))
    metric_columns[3].metric("Sparse fields", str(len(BANK_MARKETING_CATEGORICAL_FEATURE_NAMES)))

    st.markdown("### Split summary")
    st.table(
        [
            {
                "split": split_name,
                "rows": split_summary.get("rows"),
                "positives": split_summary.get("positives"),
                "positive_rate": split_summary.get("positive_rate"),
                "dense_feature_count": split_summary.get("dense_feature_count"),
                "sparse_feature_count": split_summary.get("sparse_feature_count"),
            }
            for split_name, split_summary in splits.items()
        ]
    )

    split_chart_items = [(split_name, float(split_summary.get("rows", 0))) for split_name, split_summary in splits.items()]
    render_bar_chart("Split row counts", split_chart_items)

    raw_contact_path = paths["phase2_dir"] / "raw" / "raw_contact_events.jsonl"
    raw_label_path = paths["phase2_dir"] / "raw" / "raw_subscription_labels.jsonl"
    if raw_contact_path.exists() or raw_label_path.exists():
        st.markdown("### Raw Phase 2 contracts")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Contact events")
            if raw_contact_path.exists():
                st.table(read_jsonl(str(raw_contact_path), limit=3))
            else:
                st.caption(f"Missing: `{raw_contact_path}`")
        with right:
            st.markdown("#### Subscription labels")
            if raw_label_path.exists():
                st.table(read_jsonl(str(raw_label_path), limit=3))
            else:
                st.caption(f"Missing: `{raw_label_path}`")

    raw_exists = raw_profile.get("exists")
    st.markdown("### Raw data profile")
    if not raw_exists:
        st.warning(f"Raw file not found: `{paths['raw_csv_path']}`")
    else:
        raw_metric_columns = st.columns(4)
        raw_metric_columns[0].metric("Raw rows", _format_int(raw_profile.get("row_count")))
        raw_metric_columns[1].metric("Label rate", _format_percent(_safe_rate(raw_profile.get("label_counts", {}).get("yes"), raw_profile.get("row_count"))))
        raw_metric_columns[2].metric("Numeric fields", str(len(BANK_MARKETING_NUMERIC_FEATURE_NAMES) + 1))
        raw_metric_columns[3].metric("Categorical fields", str(len(BANK_MARKETING_CATEGORICAL_FEATURE_NAMES)))

        categorical_feature = st.selectbox(
            "Categorical feature",
            BANK_MARKETING_CATEGORICAL_FEATURE_NAMES,
            key="phase2_categorical_feature",
        )
        categorical_counts = raw_profile.get("categorical_counts", {}).get(categorical_feature, {})
        top_categories = sorted(categorical_counts.items(), key=lambda item: item[1], reverse=True)[:10]
        render_bar_chart(
            f"{categorical_feature} top categories",
            [(name, float(count)) for name, count in top_categories],
            value_suffix=" rows",
            max_items=10,
        )

        numeric_feature_choices = BANK_MARKETING_NUMERIC_FEATURE_NAMES + [BANK_MARKETING_LEAKAGE_FEATURE_NAME]
        numeric_feature = st.selectbox(
            "Numeric feature",
            numeric_feature_choices,
            index=0,
            key="phase2_numeric_feature",
        )
        numeric_values = raw_profile.get("numeric_values", {}).get(numeric_feature, [])
        if numeric_feature == BANK_MARKETING_LEAKAGE_FEATURE_NAME:
            st.warning(
                "The `duration` field is leakage-prone and is preserved only for audit visibility, not as a default model feature."
            )

        raw_hist = histogram(numeric_values, bins=12)
        render_bar_chart(
            f"Raw distribution: {numeric_feature}",
            raw_hist,
            value_suffix="",
            max_items=12,
        )

        transformed_values = transform_numeric_series(numeric_feature, numeric_values)
        transformed_hist = histogram(transformed_values, bins=12)
        render_bar_chart(
            f"Engineered distribution: {numeric_feature}",
            transformed_hist,
            value_suffix="",
            max_items=12,
        )

        if numeric_feature == "pdays":
            prior_contact_values = [transform_prior_contact_flag(str(value) if value is not None else None) for value in numeric_values]
            render_bar_chart(
                "Derived prior_contact_flag",
                histogram(prior_contact_values, bins=2),
                value_suffix="",
                max_items=2,
            )

        numeric_summary = build_numeric_summary(numeric_values)
        transformed_summary = build_numeric_summary(transformed_values)
        st.table(
            [
                {"view": "raw", **numeric_summary},
                {"view": "engineered", **transformed_summary},
            ]
        )

    st.markdown("### Training row preview")
    preview_split = st.selectbox(
        "Preview split",
        ["train", "validation", "test"],
        key="phase2_preview_split",
    )
    preview_rows = phase2.get("previews", {}).get(preview_split, [])
    st.table(preview_rows)


def render_phase3_tab(paths: dict[str, Path]) -> None:
    st.subheader("Phase 3 outputs")
    phase3 = load_phase3_artifacts(str(paths["phase3_dir"]))
    if not phase3.get("exists"):
        st.error(f"Phase 3 artifacts are missing under `{paths['phase3_dir']}`.")
        st.code(
            "python pipelines/train_logistic_regression.py --data-dir "
            f"{paths['phase2_dir']} --output-dir {paths['phase3_dir']}"
        )
        return

    model = phase3["model"]
    metrics = phase3["metrics"]
    history = phase3["history"]

    metric_columns = st.columns(4)
    metric_columns[0].metric("Model version", model.model_version)
    metric_columns[1].metric("Validation ROC-AUC", _format_float(metrics.get("validation_metrics", {}).get("roc_auc")))
    metric_columns[2].metric("Validation ECE", _format_float(metrics.get("validation_metrics", {}).get("ece")))
    metric_columns[3].metric("Test log loss", _format_float(metrics.get("test_metrics", {}).get("log_loss")))

    st.markdown("### Metrics by split")
    st.table(phase3["metric_rows"])

    validation_bins = metrics.get("validation_metrics", {}).get("calibration_bins", [])
    if validation_bins:
        labels = [f"{row['lower_bound']:.1f}–{row['upper_bound']:.1f}" for row in validation_bins]
        render_line_chart(
            "Validation reliability curve",
            {
                "avg_prediction": [row["average_prediction"] for row in validation_bins],
                "observed_rate": [row["observed_rate"] for row in validation_bins],
            },
            labels,
        )

    if history:
        labels = [str(row["epoch"]) for row in history]
        render_line_chart(
            "Training history",
            {
                "train_loss": [row["train_loss"] for row in history],
                "validation_log_loss": [row["validation_log_loss"] for row in history],
            },
            labels,
        )

    st.markdown("### Top dense coefficients")
    st.table(phase3["dense_coefficients"])
    render_bar_chart(
        "Dense coefficient magnitude",
        [(row["feature"], float(row["absolute_weight"])) for row in phase3["dense_coefficients"]],
        value_suffix="",
    )

    st.markdown("### Top sparse coefficients")
    sparse_rows = [
        {"feature_id": row["feature_id"], "weight": row["weight"], "absolute_weight": row["absolute_weight"]}
        for row in phase3["sparse_coefficients"]
    ]
    st.table(sparse_rows)
    render_bar_chart(
        "Sparse coefficient magnitude",
        [(f"id {row['feature_id']}", float(row["absolute_weight"])) for row in phase3["sparse_coefficients"]],
        value_suffix="",
    )

    with st.expander("Model artifact"):
        st.json(model.to_dict())

    if phase3.get("experiment_report"):
        with st.expander("Experiment suite report"):
            st.json(phase3["experiment_report"])


def render_phase4_tab(paths: dict[str, Path]) -> None:
    st.subheader("Phase 4 outputs")
    phase4 = load_phase4_artifacts(str(paths["phase4_dir"]))
    if not phase4.get("exists"):
        st.error(f"Phase 4 artifacts are missing under `{paths['phase4_dir']}`.")
        st.code(
            "python pipelines/calibrate_and_route.py --data-dir "
            f"{paths['phase2_dir']} --model-path {paths['phase3_dir'] / 'model.json'} --output-dir {paths['phase4_dir']}"
        )
        return

    calibration = phase4["calibration"]
    report = phase4["report"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected method", str(calibration.get("selected_method")))
    metric_columns[1].metric("Selection metric", str(calibration.get("selection_metric")))
    metric_columns[2].metric("Validation ECE", _format_float(report.get("validation_evaluation", {}).get("calibrated_metrics", {}).get("ece")))
    metric_columns[3].metric("Test ECE", _format_float(report.get("test_evaluation", {}).get("calibrated_metrics", {}).get("ece")))

    st.markdown("### Calibration methods")
    st.table(phase4["method_rows"])

    raw_validation_bins = phase4["raw_validation_metrics"].get("calibration_bins", [])
    selected_validation_bins = phase4["selected_validation_metrics"].get("calibration_bins", [])
    if raw_validation_bins and selected_validation_bins:
        labels = [f"{row['lower_bound']:.1f}–{row['upper_bound']:.1f}" for row in raw_validation_bins]
        render_line_chart(
            "Raw vs selected validation calibration",
            {
                "raw_avg_prediction": [row["average_prediction"] for row in raw_validation_bins],
                "selected_avg_prediction": [row["average_prediction"] for row in selected_validation_bins],
                "observed_rate": [row["observed_rate"] for row in selected_validation_bins],
            },
            labels,
        )

    context_names = list(phase4["policy_reports"].keys())
    if context_names:
        selected_context = st.selectbox("Policy context", context_names, key="phase4_policy_context")
        context_report = phase4["policy_reports"][selected_context]
        left, right = st.columns(2)
        with left:
            st.markdown("#### Validation policy")
            st.table([{"split": "validation", **context_report.get("validation", {})}])
        with right:
            st.markdown("#### Test policy")
            st.table([{"split": "test", **context_report.get("test", {})}])

        validation_path_name = f"validation_{selected_context}.jsonl"
        test_path_name = f"test_{selected_context}.jsonl"
        preview_options = [name for name in phase4["decision_previews"] if name in {validation_path_name, test_path_name}]
        if preview_options:
            preview_choice = st.selectbox("Decision preview file", preview_options, key="phase4_decision_preview")
            st.table(phase4["decision_previews"][preview_choice])

    with st.expander("Calibration artifact"):
        st.json(calibration)
    with st.expander("Policy report"):
        st.json(report)


def render_phase5_tab(paths: dict[str, Path]) -> None:
    st.subheader("Phase 5 outputs")
    phase5 = load_phase5_artifacts(str(paths["phase5_dir"]))
    if not phase5.get("exists"):
        st.error(f"Phase 5 logs are missing under `{paths['phase5_dir']}`.")
        st.code(
            "python pipelines/serve_prediction_api.py --model-path "
            f"{paths['phase3_dir'] / 'model.json'} --calibration-path {paths['phase4_dir'] / 'calibration' / 'calibration.json'} --log-dir {paths['phase5_dir']}"
        )
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Predictions", _format_int(phase5["prediction_count"]))
    metric_columns[1].metric("Decisions", _format_int(phase5["decision_count"]))
    metric_columns[2].metric("Prediction link rate", _format_percent(phase5["prediction_link_rate"]))
    metric_columns[3].metric("Decision link rate", _format_percent(phase5["decision_link_rate"]))

    latency_columns = st.columns(3)
    latency_columns[0].metric("Latency p50", f"{phase5['latency_summary']['p50']:.2f} ms")
    latency_columns[1].metric("Latency p95", f"{phase5['latency_summary']['p95']:.2f} ms")
    latency_columns[2].metric("Latency p99", f"{phase5['latency_summary']['p99']:.2f} ms")

    render_bar_chart(
        "Decision actions",
        sorted(phase5["action_counts"].items(), key=lambda item: item[1], reverse=True),
        value_suffix=" decisions",
    )

    render_bar_chart(
        "Prediction latency breakdown (p95)",
        [(name, float(values["p95"])) for name, values in phase5["latency_breakdown"].items()],
        value_suffix=" ms",
    )

    sample_prediction = phase5.get("sample_prediction")
    sample_decision = phase5.get("sample_decision")
    if sample_prediction or sample_decision:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Sample prediction log")
            if sample_prediction:
                st.json(sample_prediction)
        with right:
            st.markdown("#### Sample decision log")
            if sample_decision:
                st.json(sample_decision)

    if phase5.get("sample_feature_vector"):
        with st.expander("Sample engineered feature vector"):
            st.json(phase5["sample_feature_vector"])


def render_phase6_tab(paths: dict[str, Path]) -> None:
    st.subheader("Phase 6 outputs")
    phase6 = load_phase6_artifacts(str(paths["phase6_dir"]))
    if not phase6.get("exists"):
        st.error(f"Phase 6 monitoring artifacts are missing under `{paths['phase6_dir']}`.")
        st.code(
            "python scripts/phase6_runner.py --mode "
            f"{'sample' if 'sample' in str(paths['phase6_dir']) else 'full'}"
        )
        return

    summary = phase6["summary"]
    rollout = phase6["rollout_readiness"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Prediction p95", f"{summary.get('prediction_p95_latency_ms', 0.0):.2f} ms")
    metric_columns[1].metric("Manual review", _format_percent(summary.get("manual_review_rate")))
    metric_columns[2].metric("Audit coverage", _format_percent(summary.get("prediction_link_rate")))
    metric_columns[3].metric("Rollout", "Ready" if rollout.get("ready") else "Blocked")

    if rollout.get("reasons"):
        st.warning(" | ".join(str(reason) for reason in rollout["reasons"]))

    st.markdown("### Monitoring summary")
    st.table([summary])

    if phase6["alerts"]:
        st.markdown("### Alerts")
        st.table(phase6["alerts"])

    if phase6["dense_psi_rows"]:
        render_bar_chart(
            "Dense feature PSI",
            [(feature, float(value)) for feature, value in phase6["dense_psi_rows"]],
            value_suffix="",
        )

    calibration_metrics = phase6["calibration_metrics"]
    if calibration_metrics.get("baseline_validation_calibrated_metrics") or calibration_metrics.get("live_primary_calibrated_metrics"):
        with st.expander("Calibration snapshot"):
            st.json(calibration_metrics)

    if phase6["group_rows"]:
        st.markdown("### Group metrics")
        st.table(phase6["group_rows"])

    with st.expander("Monitoring report"):
        st.json(phase6["report"])


def render_phase7_tab(paths: dict[str, Path]) -> None:
    st.subheader("Phase 7 outputs")
    phase7 = load_phase7_artifacts(str(paths["phase7_dir"]))
    if not phase7.get("exists"):
        st.error(f"Phase 7 artifacts are missing under `{paths['phase7_dir']}`.")
        st.code(
            "python scripts/phase7_runner.py --mode "
            f"{'sample' if 'sample' in str(paths['phase7_dir']) else 'full'}"
        )
        return

    rollout = phase7["rollout_readiness"]
    overall = phase7["overall"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Validation lift", _format_float(rollout.get("validation_reranker_lift_vs_stage1")))
    metric_columns[1].metric("Test lift", _format_float(rollout.get("test_reranker_lift_vs_stage1")))
    metric_columns[2].metric("Average final NDCG", _format_float(overall.get("average_final_ndcg_at_k")))
    metric_columns[3].metric("Rollout", "Ready" if rollout.get("ready") else "Review")

    if rollout.get("reasons"):
        st.warning(" | ".join(str(reason) for reason in rollout["reasons"]))

    st.markdown("### Ranking summary")
    st.table(phase7["split_rows"])

    render_bar_chart(
        "Average ranking quality",
        [
            ("stage1 positive rate", float(overall.get("average_stage1_positive_rate", 0.0))),
            ("reranker positive rate", float(overall.get("average_reranker_positive_rate", 0.0))),
            ("final positive rate", float(overall.get("average_final_positive_rate", 0.0))),
            ("stage1 ndcg", float(overall.get("average_stage1_ndcg_at_k", 0.0))),
            ("reranker ndcg", float(overall.get("average_reranker_ndcg_at_k", 0.0))),
            ("final ndcg", float(overall.get("average_final_ndcg_at_k", 0.0))),
        ],
        value_suffix="",
    )

    if phase7["reranker_history"]:
        labels = [str(row["epoch"]) for row in phase7["reranker_history"]]
        render_line_chart(
            "Reranker training history",
            {
                "train_loss": [row["train_loss"] for row in phase7["reranker_history"]],
                "validation_log_loss": [row["validation_log_loss"] for row in phase7["reranker_history"]],
            },
            labels,
        )

    with st.expander("Reranker artifacts"):
        st.json(
            {
                "model": phase7.get("reranker_model"),
                "metrics": phase7.get("reranker_metrics"),
            }
        )

    ranking_options = sorted(phase7["rankings"].keys())
    if ranking_options:
        split_choice = st.selectbox("Ranking preview split", ranking_options, key="phase7_ranking_split")
        st.table(phase7["rankings"][split_choice])

    with st.expander("Hybrid ranking report"):
        st.json(phase7["report"])


def transform_numeric_series(feature_name: str, numeric_values: Sequence[float]) -> list[float]:
    if feature_name == "prior_contact_flag":
        return [transform_prior_contact_flag(str(value) if value is not None else None) for value in numeric_values]
    config = FeatureConfig(hash_dimension=1)
    return [transform_dense_value(str(value) if value is not None else None, config, feature_name) for value in numeric_values]


def _safe_rate(numerator: object, denominator: object) -> float:
    numerator_value = float(numerator or 0.0)
    denominator_value = float(denominator or 0.0)
    if denominator_value == 0.0:
        return 0.0
    return numerator_value / denominator_value


def _format_float(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_int(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
