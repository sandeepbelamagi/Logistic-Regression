from __future__ import annotations

import http.client
import json
import threading
import time
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401

from probabilistic_decisioning.calibration import fit_calibration_artifact, save_calibration_artifact
from probabilistic_decisioning.dataset_builder import DatasetBuilderConfig, build_dataset
from probabilistic_decisioning.logistic_regression import (
    LogisticRegressionTrainingConfig,
    load_training_splits,
    save_training_artifacts,
    train_logistic_regression,
)
from probabilistic_decisioning.serving import (
    ServingRequest,
    build_runtime,
    create_server,
)

WORKSPACE_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"


def _sample_row(label: str, seed: int) -> str:
    values = [
        str(25 + seed),
        "admin." if seed % 2 == 0 else "technician",
        "married" if seed % 3 == 0 else "single",
        "secondary",
        "no",
        str(1000 + seed * 10),
        "yes" if seed % 2 == 0 else "no",
        "no",
        "cellular",
        str(10 + seed),
        "oct",
        str(60 + seed),
        str(1 + seed % 3),
        "-1" if seed % 2 == 0 else "42",
        str(seed % 4),
        "unknown",
        label,
    ]
    return ";".join(values)


def _sample_request_payload(seed: int, task_context: str = "bank_marketing") -> dict[str, object]:
    return {
        "request_id": f"req_{seed}",
        "event_id": f"evt_{seed}",
        "event_ts": "2026-01-01T00:00:00Z",
        "task_context": task_context,
        "features": {
            "age": str(35 + seed),
            "job": "admin." if seed % 2 == 0 else "technician",
            "marital": "married" if seed % 3 == 0 else "single",
            "education": "secondary",
            "default": "no",
            "balance": str(1200 + seed * 15),
            "housing": "yes" if seed % 2 == 0 else "no",
            "loan": "no",
            "contact": "cellular",
            "day": str(12 + seed),
            "month": "oct",
            "duration": str(80 + seed),
            "campaign": str(1 + seed % 3),
            "pdays": "-1" if seed % 2 == 0 else "42",
            "previous": str(seed % 4),
            "poutcome": "unknown",
        },
    }


class ServingRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        WORKSPACE_TEMP_ROOT.mkdir(exist_ok=True)

    def _build_serving_bundle(self, name: str):
        tmp_path = WORKSPACE_TEMP_ROOT / name
        tmp_path.mkdir(parents=True, exist_ok=True)
        input_path = tmp_path / "bank-full.csv"
        input_path.write_text(
            "\n".join(
                [
                    _sample_row("no", 1),
                    _sample_row("no", 2),
                    _sample_row("yes", 3),
                    _sample_row("no", 4),
                    _sample_row("no", 5),
                    _sample_row("yes", 6),
                    _sample_row("no", 7),
                    _sample_row("no", 8),
                    _sample_row("yes", 9),
                    _sample_row("no", 10),
                    _sample_row("no", 11),
                    _sample_row("yes", 12),
                ]
            ),
            encoding="utf-8",
        )

        artifacts_dir = tmp_path / "artifacts"
        build_dataset(
            DatasetBuilderConfig(
                input_path=input_path,
                output_dir=artifacts_dir,
                hash_dimension=256,
                split_strategy="contiguous",
                train_ratio=0.5,
                validation_ratio=0.25,
                test_ratio=0.25,
            )
        )
        splits = load_training_splits(artifacts_dir)
        training_result = train_logistic_regression(
            splits["train"],
            splits["validation"],
            splits["test"],
            LogisticRegressionTrainingConfig(
                learning_rate=0.1,
                max_epochs=5,
                l2=0.0001,
                early_stopping_patience=2,
                seed=17,
            ),
            model_version="phase5_unit_test_lr_v1",
            hash_dimension=256,
        )
        training_paths = save_training_artifacts(training_result, tmp_path / "model")

        calibration_artifact = fit_calibration_artifact(
            model=training_result.model,
            validation_examples=splits["validation"],
            candidate_methods=("platt_scaling", "isotonic_regression"),
            selection_metric="validation_ece",
            calibration_version="phase5_unit_test_calibration_v1",
        )
        calibration_path = save_calibration_artifact(
            calibration_artifact,
            tmp_path / "calibration" / "calibration.json",
        )
        return tmp_path, training_paths["model_path"], calibration_path

    def test_runtime_predicts_routes_and_logs(self) -> None:
        base_dir, model_path, calibration_path = self._build_serving_bundle("test_runtime_predicts_routes_and_logs")
        prediction_log_path = base_dir / "prediction_log.jsonl"
        decision_log_path = base_dir / "decision_log.jsonl"
        runtime = build_runtime(
            model_path=model_path,
            calibration_path=calibration_path,
            prediction_log_path=prediction_log_path,
            decision_log_path=decision_log_path,
        )

        request = ServingRequest.from_payload(_sample_request_payload(3))
        feature_vector = runtime.lookup_features(request)
        self.assertEqual(feature_vector.feature_set_version, "bank_marketing_v1")
        self.assertEqual(len(feature_vector.dense_features), 7)
        self.assertEqual(len(feature_vector.sparse_feature_ids), 9)

        response = runtime.predict(request)
        self.assertIsNotNone(response.raw_score)
        self.assertIsNotNone(response.raw_probability)
        self.assertIsNotNone(response.calibrated_probability)
        self.assertIn(response.decision["action"], {"suppress", "nurture", "prioritize_contact"})
        self.assertTrue(prediction_log_path.exists())
        self.assertTrue(decision_log_path.exists())

        prediction_payload = json.loads(prediction_log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        decision_payload = json.loads(decision_log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        self.assertEqual(prediction_payload["prediction_id"], response.prediction_id)
        self.assertEqual(prediction_payload["request_id"], request.request_id)
        self.assertGreaterEqual(prediction_payload["latency_ms"], 0.0)
        self.assertEqual(decision_payload["decision_id"], f"decision_{request.event_id}")

    def test_runtime_routes_without_raw_features(self) -> None:
        base_dir, model_path, calibration_path = self._build_serving_bundle("test_runtime_routes_without_raw_features")
        runtime = build_runtime(model_path=model_path, calibration_path=calibration_path)

        request = ServingRequest.from_payload(
            {
                "request_id": "req_route_only",
                "event_id": "evt_route_only",
                "event_ts": "2026-01-01T00:00:00Z",
                "task_context": "fraud_policy",
                "calibrated_probability": 0.8,
                "raw_probability": 0.7,
            }
        )
        response = runtime.route(request)
        self.assertIsNone(response.feature_vector)
        self.assertEqual(response.decision["action"], "block")
        self.assertIsNone(response.prediction_log)
        self.assertEqual(response.task_context, "fraud_policy")

    def test_http_server_handles_predict_and_decision_requests(self) -> None:
        _base_dir, model_path, calibration_path = self._build_serving_bundle(
            "test_http_server_handles_predict_and_decision_requests"
        )
        runtime = build_runtime(model_path=model_path, calibration_path=calibration_path)
        server = create_server("127.0.0.1", 0, runtime)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.05)
        host, port = server.server_address

        try:
            health_status, health_payload = self._post_json_response(host, port, "GET", "/health")
            self.assertEqual(health_status, 200)
            self.assertEqual(health_payload["status"], "ok")

            lookup_status, lookup_payload = self._post_json_response(
                host,
                port,
                "POST",
                "/v1/features/lookup",
                _sample_request_payload(5),
            )
            self.assertEqual(lookup_status, 200)
            self.assertIn("dense_features", lookup_payload)

            predict_status, predict_payload = self._post_json_response(
                host,
                port,
                "POST",
                "/v1/predict",
                _sample_request_payload(6),
            )
            self.assertEqual(predict_status, 200)
            self.assertIsNotNone(predict_payload["calibrated_probability"])
            self.assertIn(predict_payload["decision"]["action"], {"suppress", "nurture", "prioritize_contact"})

            decision_status, decision_payload = self._post_json_response(
                host,
                port,
                "POST",
                "/v1/decision",
                {
                    "request_id": "req_decision",
                    "event_id": "evt_decision",
                    "event_ts": "2026-01-01T00:00:00Z",
                    "task_context": "fraud_policy",
                    "calibrated_probability": 0.8,
                },
            )
            self.assertEqual(decision_status, 200)
            self.assertEqual(decision_payload["decision"]["action"], "block")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)

    def _post_json_response(
        self,
        host: str,
        port: int,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, object]]:
        connection = http.client.HTTPConnection(host, port, timeout=10)
        headers: dict[str, str] = {}
        body = None
        if payload is not None:
            body = json.dumps(payload)
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw_body = response.read().decode("utf-8")
            parsed_body = json.loads(raw_body) if raw_body else {}
            return response.status, parsed_body
        finally:
            connection.close()
