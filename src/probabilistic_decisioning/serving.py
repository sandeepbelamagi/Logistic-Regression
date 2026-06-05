"""Online serving helpers for Phase 5."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from time import perf_counter
from socket import gaierror
from typing import Any

from probabilistic_decisioning.bank_marketing import BankMarketingRecord
from probabilistic_decisioning.calibration import (
    CalibrationArtifact,
    IsotonicRegressionCalibrator,
    PlattScalingCalibrator,
    load_calibration_artifact,
    load_selected_calibrator,
)
from probabilistic_decisioning.constants import (
    BANK_MARKETING_CATEGORICAL_FEATURE_NAMES,
    BANK_MARKETING_DENSE_FEATURE_NAMES,
    BANK_MARKETING_INPUT_FIELDS,
    BANK_MARKETING_LEAKAGE_FEATURE_NAME,
    BANK_MARKETING_NUMERIC_FEATURE_NAMES,
)
from probabilistic_decisioning.decision_policy import DecisionOutcome, route_decision
from probabilistic_decisioning.features import FeatureConfig, build_dense_vector, build_sparse_vector
from probabilistic_decisioning.logistic_regression import LogisticRegressionModel, load_model_artifact


Calibrator = PlattScalingCalibrator | IsotonicRegressionCalibrator


@dataclass(frozen=True)
class ServingRequest:
    """Request payload accepted by the online serving layer."""

    request_id: str
    event_id: str
    event_ts: str
    task_context: str = "bank_marketing"
    features: dict[str, str | None] = field(default_factory=dict)
    raw_score: float | None = None
    raw_probability: float | None = None
    calibrated_probability: float | None = None
    realized_label: int | None = None
    realized_value: float | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ServingRequest":
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")

        request_id = _require_text(payload, "request_id")
        event_id = _require_text(payload, "event_id")
        event_ts = _normalize_timestamp(_require_text(payload, "event_ts"))
        task_context = str(payload.get("task_context", "bank_marketing")).strip().lower()
        features = _extract_feature_payload(payload)

        return cls(
            request_id=request_id,
            event_id=event_id,
            event_ts=event_ts,
            task_context=task_context,
            features=features,
            raw_score=_optional_float(payload.get("raw_score")),
            raw_probability=_optional_float(payload.get("raw_probability")),
            calibrated_probability=_optional_float(payload.get("calibrated_probability")),
            realized_label=_optional_int(payload.get("realized_label")),
            realized_value=_optional_float(payload.get("realized_value")),
        )

    def has_raw_features(self) -> bool:
        return any(value is not None for value in self.features.values())


@dataclass(frozen=True)
class FeatureVector:
    """Engineered sparse and dense features for a single online request."""

    request_id: str
    event_id: str
    event_ts: str
    task_context: str
    feature_set_version: str
    hash_dimension: int
    dense_feature_names: tuple[str, ...]
    dense_features: tuple[float, ...]
    sparse_feature_ids: tuple[int, ...]
    sparse_feature_values: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "event_id": self.event_id,
            "event_ts": self.event_ts,
            "task_context": self.task_context,
            "feature_set_version": self.feature_set_version,
            "hash_dimension": self.hash_dimension,
            "dense_feature_names": list(self.dense_feature_names),
            "dense_features": list(self.dense_features),
            "sparse_feature_ids": list(self.sparse_feature_ids),
            "sparse_feature_values": list(self.sparse_feature_values),
            "dense_feature_count": len(self.dense_features),
            "sparse_feature_count": len(self.sparse_feature_ids),
        }


@dataclass(frozen=True)
class PredictionLogRecord:
    """Prediction log payload aligned with `data_contracts/prediction_log.yaml`."""

    prediction_id: str
    prediction_date: str
    prediction_ts: str
    request_id: str
    event_id: str | None
    model_version: str
    calibration_version: str | None
    feature_set_version: str
    feature_vector: dict[str, object]
    raw_score: float
    calibrated_score: float
    decision_policy_version: str | None
    latency_ms: float
    task_context: str

    def to_contract_dict(self) -> dict[str, object]:
        return {
            "prediction_id": self.prediction_id,
            "prediction_date": self.prediction_date,
            "prediction_ts": self.prediction_ts,
            "request_id": self.request_id,
            "event_id": self.event_id,
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "feature_set_version": self.feature_set_version,
            "feature_vector": self.feature_vector,
            "raw_score": self.raw_score,
            "calibrated_score": self.calibrated_score,
            "decision_policy_version": self.decision_policy_version,
            "latency_ms": self.latency_ms,
            "task_context": self.task_context,
        }


@dataclass(frozen=True)
class ServingResponse:
    """Unified HTTP response for prediction and decision endpoints."""

    request_id: str
    event_id: str
    event_ts: str
    task_context: str
    prediction_id: str
    prediction_ts: str
    model_version: str
    calibration_version: str | None
    feature_set_version: str
    feature_vector: dict[str, object] | None
    raw_score: float | None
    raw_probability: float | None
    calibrated_probability: float | None
    decision: dict[str, object] | None
    prediction_log: dict[str, object] | None
    latency_ms: float
    feature_lookup_latency_ms: float
    scoring_latency_ms: float
    calibration_latency_ms: float
    decision_latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "event_id": self.event_id,
            "event_ts": self.event_ts,
            "task_context": self.task_context,
            "prediction_id": self.prediction_id,
            "prediction_ts": self.prediction_ts,
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "feature_set_version": self.feature_set_version,
            "feature_vector": self.feature_vector,
            "raw_score": self.raw_score,
            "raw_probability": self.raw_probability,
            "calibrated_probability": self.calibrated_probability,
            "decision": self.decision,
            "prediction_log": self.prediction_log,
            "latency_ms": self.latency_ms,
            "feature_lookup_latency_ms": self.feature_lookup_latency_ms,
            "scoring_latency_ms": self.scoring_latency_ms,
            "calibration_latency_ms": self.calibration_latency_ms,
            "decision_latency_ms": self.decision_latency_ms,
        }


@dataclass(frozen=True)
class ServingBundle:
    """Loaded model, calibration, and feature-engineering assets."""

    model: LogisticRegressionModel
    calibration_artifact: CalibrationArtifact
    calibrator: Calibrator
    feature_config: FeatureConfig

    @property
    def model_version(self) -> str:
        return self.model.model_version

    @property
    def feature_set_version(self) -> str:
        return self.model.feature_set_version

    @property
    def task_context(self) -> str:
        return self.model.task_context

    @classmethod
    def load(cls, model_path: Path, calibration_path: Path) -> "ServingBundle":
        model = load_model_artifact(model_path)
        calibration_artifact = load_calibration_artifact(calibration_path)

        if calibration_artifact.model_version != model.model_version:
            raise ValueError(
                "Calibration artifact model_version does not match model.json. "
                f"Model={model.model_version!r}, calibration={calibration_artifact.model_version!r}."
            )
        if calibration_artifact.feature_set_version != model.feature_set_version:
            raise ValueError(
                "Calibration artifact feature_set_version does not match model.json. "
                f"Model={model.feature_set_version!r}, calibration={calibration_artifact.feature_set_version!r}."
            )
        if calibration_artifact.task_context != model.task_context:
            raise ValueError(
                "Calibration artifact task_context does not match model.json. "
                f"Model={model.task_context!r}, calibration={calibration_artifact.task_context!r}."
            )

        return cls(
            model=model,
            calibration_artifact=calibration_artifact,
            calibrator=load_selected_calibrator(calibration_artifact),
            feature_config=FeatureConfig(hash_dimension=model.hash_dimension),
        )


class OnlineServingRuntime:
    """In-process implementation of the Phase 5 serving path."""

    def __init__(
        self,
        bundle: ServingBundle,
        prediction_log_path: Path | None = None,
        decision_log_path: Path | None = None,
    ) -> None:
        self.bundle = bundle
        self.prediction_log_path = prediction_log_path
        self.decision_log_path = decision_log_path
        self._prediction_log_lock = Lock()
        self._decision_log_lock = Lock()

    def lookup_features(self, request: ServingRequest) -> FeatureVector:
        record = _request_to_record(request)
        dense_features = tuple(build_dense_vector(record, self.bundle.feature_config))
        sparse_feature_ids, sparse_feature_values = build_sparse_vector(record, self.bundle.feature_config)
        return FeatureVector(
            request_id=request.request_id,
            event_id=request.event_id,
            event_ts=request.event_ts,
            task_context=request.task_context,
            feature_set_version=self.bundle.feature_set_version,
            hash_dimension=self.bundle.model.hash_dimension,
            dense_feature_names=_resolve_dense_feature_names(self.bundle.feature_config),
            dense_features=dense_features,
            sparse_feature_ids=tuple(sparse_feature_ids),
            sparse_feature_values=tuple(sparse_feature_values),
        )

    def predict(self, request: ServingRequest, calibrated_probability_override: float | None = None) -> ServingResponse:
        start_time = perf_counter()
        feature_lookup_start = perf_counter()
        feature_vector = self.lookup_features(request)
        feature_lookup_latency_ms = _elapsed_ms(feature_lookup_start)

        scoring_start = perf_counter()
        raw_score = self.bundle.model.score_features(
            feature_vector.dense_features,
            feature_vector.sparse_feature_ids,
            feature_vector.sparse_feature_values,
        )
        raw_probability = self.bundle.model.predict_proba_features(
            feature_vector.dense_features,
            feature_vector.sparse_feature_ids,
            feature_vector.sparse_feature_values,
        )
        scoring_latency_ms = _elapsed_ms(scoring_start)

        calibration_start = perf_counter()
        calibrated_probability = (
            calibrated_probability_override
            if calibrated_probability_override is not None
            else self.bundle.calibrator.predict(raw_score)
        )
        calibration_latency_ms = _elapsed_ms(calibration_start)

        decision_start = perf_counter()
        decision = route_decision(
            task_context=request.task_context,
            calibrated_probability=calibrated_probability,
            event_id=request.event_id,
            event_ts=request.event_ts,
            prediction_id=f"prediction_{request.event_id}",
            raw_probability=raw_probability,
            realized_label=request.realized_label,
            realized_value=request.realized_value,
        )
        decision_latency_ms = _elapsed_ms(decision_start)

        prediction_timestamp = datetime.now(tz=UTC)
        total_latency_ms = _elapsed_ms(start_time)
        prediction_log = PredictionLogRecord(
            prediction_id=f"prediction_{request.event_id}",
            prediction_date=prediction_timestamp.date().isoformat(),
            prediction_ts=_isoformat(prediction_timestamp),
            request_id=request.request_id,
            event_id=request.event_id,
            model_version=self.bundle.model_version,
            calibration_version=self.bundle.calibration_artifact.calibration_version,
            feature_set_version=self.bundle.feature_set_version,
            feature_vector=feature_vector.to_dict(),
            raw_score=raw_score,
            calibrated_score=calibrated_probability,
            decision_policy_version=decision.decision_policy_version,
            latency_ms=total_latency_ms,
            task_context=request.task_context,
        )
        response = ServingResponse(
            request_id=request.request_id,
            event_id=request.event_id,
            event_ts=request.event_ts,
            task_context=request.task_context,
            prediction_id=prediction_log.prediction_id,
            prediction_ts=prediction_log.prediction_ts,
            model_version=self.bundle.model_version,
            calibration_version=self.bundle.calibration_artifact.calibration_version,
            feature_set_version=self.bundle.feature_set_version,
            feature_vector=feature_vector.to_dict(),
            raw_score=raw_score,
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            decision=decision.to_dict(),
            prediction_log=prediction_log.to_contract_dict(),
            latency_ms=total_latency_ms,
            feature_lookup_latency_ms=feature_lookup_latency_ms,
            scoring_latency_ms=scoring_latency_ms,
            calibration_latency_ms=calibration_latency_ms,
            decision_latency_ms=decision_latency_ms,
        )
        self._log_prediction(prediction_log)
        self._log_decision(decision)
        return response

    def route(self, request: ServingRequest) -> ServingResponse:
        if request.has_raw_features():
            return self.predict(request, calibrated_probability_override=request.calibrated_probability)
        if request.calibrated_probability is None:
            raise ValueError("Decision routing requires either raw features or calibrated_probability.")

        start_time = perf_counter()
        decision = route_decision(
            task_context=request.task_context,
            calibrated_probability=request.calibrated_probability,
            event_id=request.event_id,
            event_ts=request.event_ts,
            prediction_id=f"prediction_{request.event_id}",
            raw_probability=request.raw_probability,
            realized_label=request.realized_label,
            realized_value=request.realized_value,
        )
        decision_latency_ms = _elapsed_ms(start_time)
        total_latency_ms = decision_latency_ms
        prediction_timestamp = datetime.now(tz=UTC)
        response = ServingResponse(
            request_id=request.request_id,
            event_id=request.event_id,
            event_ts=request.event_ts,
            task_context=request.task_context,
            prediction_id=f"prediction_{request.event_id}",
            prediction_ts=_isoformat(prediction_timestamp),
            model_version=self.bundle.model_version,
            calibration_version=self.bundle.calibration_artifact.calibration_version,
            feature_set_version=self.bundle.feature_set_version,
            feature_vector=None,
            raw_score=request.raw_score,
            raw_probability=request.raw_probability,
            calibrated_probability=request.calibrated_probability,
            decision=decision.to_dict(),
            prediction_log=None,
            latency_ms=total_latency_ms,
            feature_lookup_latency_ms=0.0,
            scoring_latency_ms=0.0,
            calibration_latency_ms=0.0,
            decision_latency_ms=decision_latency_ms,
        )
        self._log_decision(decision)
        return response

    def _log_prediction(self, prediction_log: PredictionLogRecord) -> None:
        if self.prediction_log_path is None:
            return
        _append_jsonl(self.prediction_log_path, prediction_log.to_contract_dict(), self._prediction_log_lock)

    def _log_decision(self, decision: DecisionOutcome) -> None:
        if self.decision_log_path is None:
            return
        _append_jsonl(self.decision_log_path, decision.to_contract_dict(), self._decision_log_lock)


class JsonServingHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server with a bound serving runtime."""

    def __init__(self, server_address: tuple[str, int], runtime: OnlineServingRuntime) -> None:
        self.runtime = runtime
        handler = _make_handler(runtime)
        super().__init__(server_address, handler)


def create_server(
    host: str,
    port: int,
    runtime: OnlineServingRuntime,
) -> JsonServingHTTPServer:
    """Create a threaded HTTP server for local development."""

    try:
        return JsonServingHTTPServer((host, port), runtime)
    except gaierror:
        if host in {"", "0.0.0.0"}:
            raise
        return JsonServingHTTPServer(("", port), runtime)


def _make_handler(runtime: OnlineServingRuntime):
    class ServingRequestHandler(BaseHTTPRequestHandler):
        server_version = "ProbabilisticDecisioning/Phase5"

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/health", "/ready"}:
                self._write_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "model_version": runtime.bundle.model_version,
                        "feature_set_version": runtime.bundle.feature_set_version,
                        "task_context": runtime.bundle.task_context,
                        "calibration_version": runtime.bundle.calibration_artifact.calibration_version,
                        "prediction_logging_enabled": runtime.prediction_log_path is not None,
                    },
                )
                return
            self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json()
                request = ServingRequest.from_payload(payload)
                if self.path == "/v1/features/lookup":
                    feature_vector = runtime.lookup_features(request)
                    self._write_json(HTTPStatus.OK, feature_vector.to_dict())
                    return
                if self.path == "/v1/predict":
                    response = runtime.predict(request)
                    self._write_json(HTTPStatus.OK, response.to_dict())
                    return
                if self.path == "/v1/decision":
                    response = runtime.route(request)
                    self._write_json(HTTPStatus.OK, response.to_dict())
                    return
                self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {self.path}"})
            except Exception as exc:  # noqa: BLE001
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _read_json(self) -> Mapping[str, Any]:
            content_length_header = self.headers.get("Content-Length")
            if content_length_header is None:
                raise ValueError("Missing Content-Length header.")
            try:
                content_length = int(content_length_header)
            except ValueError as exc:
                raise ValueError("Invalid Content-Length header.") from exc
            raw_body = self.rfile.read(content_length)
            if not raw_body:
                raise ValueError("Request body cannot be empty.")
            return json.loads(raw_body.decode("utf-8"))

        def _write_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            response_body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    return ServingRequestHandler


def load_serving_bundle(model_path: Path, calibration_path: Path) -> ServingBundle:
    """Convenience loader used by the CLI and tests."""

    return ServingBundle.load(model_path, calibration_path)


def build_runtime(
    model_path: Path,
    calibration_path: Path,
    prediction_log_path: Path | None = None,
    decision_log_path: Path | None = None,
) -> OnlineServingRuntime:
    """Load the Phase 3 and Phase 4 artifacts into an online runtime."""

    bundle = load_serving_bundle(model_path, calibration_path)
    return OnlineServingRuntime(
        bundle=bundle,
        prediction_log_path=prediction_log_path,
        decision_log_path=decision_log_path,
    )


def _request_to_record(request: ServingRequest) -> BankMarketingRecord:
    feature_payload = request.features
    numeric_features = {
        feature_name: feature_payload.get(feature_name)
        for feature_name in BANK_MARKETING_NUMERIC_FEATURE_NAMES
    }
    categorical_features = {
        feature_name: feature_payload.get(feature_name)
        for feature_name in BANK_MARKETING_CATEGORICAL_FEATURE_NAMES
    }
    leakage_prone_features = {
        BANK_MARKETING_LEAKAGE_FEATURE_NAME: feature_payload.get(BANK_MARKETING_LEAKAGE_FEATURE_NAME)
    }
    return BankMarketingRecord(
        row_id=0,
        label=request.realized_label if request.realized_label is not None else 0,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        leakage_prone_features=leakage_prone_features,
    )


def _extract_feature_payload(payload: Mapping[str, Any]) -> dict[str, str | None]:
    nested_features = payload.get("features")
    if isinstance(nested_features, Mapping):
        source_payload = nested_features
    else:
        source_payload = payload

    feature_payload: dict[str, str | None] = {}
    for feature_name in BANK_MARKETING_INPUT_FIELDS:
        raw_value = source_payload.get(feature_name)
        feature_payload[feature_name] = None if raw_value is None else str(raw_value)
    return feature_payload


def _resolve_dense_feature_names(feature_config: FeatureConfig) -> tuple[str, ...]:
    names = list(BANK_MARKETING_DENSE_FEATURE_NAMES)
    if feature_config.include_duration_feature:
        names.append(BANK_MARKETING_LEAKAGE_FEATURE_NAME)
    return tuple(names)


def _require_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"Missing required field: {key}.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Field {key} cannot be empty.")
    return text


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _normalize_timestamp(raw_timestamp: str) -> str:
    timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _isoformat(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _elapsed_ms(start_time: float) -> float:
    return (perf_counter() - start_time) * 1000.0


def _append_jsonl(path: Path, payload: Mapping[str, Any], lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")))
            handle.write("\n")
