# Pipelines

Planned pipeline modules:

- ingestion
- validation
- feature generation
- training dataset assembly
- model training
- calibration refresh
- batch scoring
- monitoring backfills

Phase 2 delivers the first executable pipeline:

- `build_training_dataset.py` builds raw and training artifacts from local Bank Marketing CSV files

Phase 5 adds the first executable serving pipeline:

- `serve_prediction_api.py` starts the local feature, prediction, and decision HTTP server

Phase 6 adds the monitoring pipeline:

- `run_monitoring.py` aggregates model, drift, calibration, and governance metrics
