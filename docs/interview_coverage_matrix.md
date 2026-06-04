# Interview Coverage Matrix

| Topic from notes | Required depth | Project implementation | Evidence artifact | Planned phase |
| --- | --- | --- | --- | --- |
| Core Logistic Regression intuition | probability modeling, sigmoid, decision boundary | baseline sparse Logistic Regression scorer | architecture and model docs | 1 to 3 |
| Cross-entropy vs MSE | math, optimization, practical implications | controlled experiment comparing convergence and calibration | experiment report | 3 |
| Fraud accuracy failure | imbalance, thresholding, business cost | policy simulator with approve, review, block routing | threshold config and KPI dashboard spec | 1 to 4 |
| Class weighting vs oversampling | optimization and overfitting tradeoff | ablation suite on minority-sensitive policy tasks | experiment report | 3 |
| Bank Marketing calibration drift | delayed labels, business impact, threshold tuning | recalibration service and drift monitors | calibration config and KPI framework | 1 to 4 |
| Why probabilistic models miscalibrate | misspecification, shift, bias, feedback loops | calibration layer plus monitoring for shift and skew | architecture doc and monitoring config | 1 and 6 |
| Loan approval failure after deployment | delayed labels, selection bias, fairness | delayed-outcome simulator and cohort monitors | data contracts and governance KPIs | 1 to 6 |
| Multicollinearity | stability and interpretability | coefficient stability study with regularization analysis | experiment report | 3 |
| Logistic Regression at internet scale | sparse features, hashing, low latency | Bank Marketing sparse LR system with hashed features | architecture doc and model config | 1 to 5 |
| Calibration vs discrimination | statistical difference and business impact | split dashboards for ranking and probability quality | KPI framework | 1 and 6 |
| Hybrid campaign ranking system | LR plus deep reranking and exploration | stage-1 linear scorer, stage-2 deep reranker, stage-3 calibration | architecture doc | 1 and 7 |

## Phase Intent

- **Phase 1** establishes architecture, KPIs, schemas, and scaffold.
- **Phase 2** prepares data and feature definitions.
- **Phase 3** delivers Logistic Regression training and experiment logic.
- **Phase 4** delivers calibration and decision policy logic.
- **Phase 5** delivers serving interfaces and latency-oriented design.
- **Phase 6** delivers monitoring, governance, and drift controls.
- **Phase 7** delivers the hybrid extension.
