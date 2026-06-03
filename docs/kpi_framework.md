# KPI Framework

## Measurement Philosophy

The platform separates four layers of measurement:

1. model discrimination
2. model calibration
3. policy and business outcomes
4. system reliability and governance

This separation is mandatory because a model can rank well but calibrate poorly, and a calibrated model can still drive bad business outcomes if thresholds or feedback loops are wrong.

## North-Star KPIs

### CTR Context

- expected value proxy from `bid * calibrated_ctr`
- calibrated CTR error by bucket
- revenue proxy uplift versus baseline policy

### Fraud Context

- fraud capture rate
- false positive cost
- manual-review queue utilization

### Loan Context

- risk-adjusted profit
- cohort default rate
- approval quality by segment

## Model KPIs

### Discrimination

- `ROC-AUC`
- `PR-AUC`
- top-decile lift

### Calibration

- log loss
- `ECE`
- maximum calibration error
- reliability curve slope and intercept

### Stability

- coefficient drift
- sign-flip rate for important coefficients
- sensitivity to regularization changes

## Policy KPIs

### Thresholding

- action rate by bucket: approve, review, block
- threshold sensitivity by cost matrix
- marginal gain from threshold movement

### Operations

- analyst queue depth
- analyst SLA breach rate
- review-to-confirmed-positive yield

### Fairness

- subgroup approval rate
- subgroup false negative rate
- subgroup calibration gap

## Data KPIs

- feature freshness lag
- feature null rate
- Population Stability Index
- label-delay distribution
- data validation pass rate

## System KPIs

- prediction API `p50`, `p95`, `p99` latency
- error rate
- availability
- online feature timeout rate
- prediction logging completeness

## Governance KPIs

- audit log coverage
- reproducibility rate for training runs
- rollback readiness for active model versions
- explanation generation coverage for regulated decisions

## KPI Ownership

- **ML engineering** owns training quality, calibration quality, and skew
- **platform engineering** owns latency, availability, and logging
- **risk or business owners** own threshold policy and cost matrices
- **governance owners** own fairness, audit, and compliance reviews

## Alert Thresholds

These thresholds are initial defaults and will be tuned after baseline experiments.

- `ECE > 0.03` triggers calibration investigation
- `PSI > 0.2` triggers drift review
- `p95 latency > 10ms` triggers serving incident
- manual-review queue utilization `> 85%` triggers threshold review
- subgroup calibration gap `> 0.05` triggers fairness investigation

## Reporting Cadence

- near-real-time system metrics
- daily model and data health summaries
- weekly threshold and business policy review
- monthly fairness and governance review
