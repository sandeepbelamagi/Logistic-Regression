# Prediction API

Implemented responsibility:

- load deployed Logistic Regression weights
- compute raw score and calibrated probability
- emit prediction logs with latency metadata

Start the local runtime with:

```bash
python pipelines/serve_prediction_api.py --model-path artifacts/bank_marketing_lr_cv/model.json --calibration-path artifacts/bank_marketing_phase4_cv/calibration/calibration.json --log-dir artifacts/bank_marketing_phase5_logs
```
