# Problem Statement

## Summary

This project builds a unified machine learning platform where Logistic Regression is treated as a production probability engine rather than a classroom model.

The platform must demonstrate:

- mathematically correct probability modeling
- production-grade calibration
- decision thresholding under asymmetric costs
- robustness to delayed labels, drift, and feedback loops
- internet-scale sparse feature training and low-latency serving

## Why This Project

The interview notes require a single project that can explain:

- why cross-entropy is used instead of MSE
- why accuracy fails under class imbalance
- how thresholding changes business outcomes
- why calibration matters in campaign and risk systems
- how feedback loops and selection bias break naive deployment
- why Logistic Regression still matters at internet scale

A bank marketing propensity platform is the strongest backbone because it naturally covers:

- sparse categorical features
- calibrated probabilities
- campaign prioritization and contact economics
- online learning and drift
- strict latency requirements

Fraud and loan workflows are implemented as decision-policy extensions on top of the same calibrated probability interface.

## Primary Dataset Choice

### Primary

- **UCI Bank Marketing `bank-full.csv`**

### Reason

- smaller, production-realistic tabular feature space
- strong fit for Logistic Regression and feature hashing
- realistic calibration, thresholding, and campaign ranking problems
- easy to run locally without large-scale infrastructure

### Secondary Evaluation Modes

- **fraud-style policy simulator** for class imbalance and action routing
- **loan-style delayed-outcome simulator** for selection bias, delayed labels, and fairness analysis

## Product Objective

Deliver a platform that can answer both modeling and systems questions:

- can the model rank and calibrate probabilities well
- can the business apply thresholds safely
- can the system detect drift, bias, and delayed-failure patterns
- can the platform serve predictions within strict latency limits

## Success Criteria

### Technical

- reproducible offline training pipeline
- calibration-aware evaluation
- low-latency online scoring design
- versioned contracts and configs

### Business

- bank marketing use case supports expected-contact ranking
- fraud policy supports cost-sensitive routing
- loan policy supports long-horizon risk controls

### Interview Readiness

The project must create concrete artifacts for:

- optimization reasoning
- calibration analysis
- threshold design
- drift diagnosis
- policy feedback-loop redesign
- internet-scale architecture tradeoffs

## In Scope

- architecture and MLOps design
- data contracts and pipeline plan
- sparse Logistic Regression training plan
- calibration framework
- decision policy engine design
- hybrid reranking extension plan

## Out of Scope for Phase 1

- full model implementation
- distributed infrastructure setup
- production deployment manifests
- deep learning model training

## Key Risks

### Data Risk

- public dataset may not expose all downstream business labels directly
- mitigated by using policy simulators with explicit cost models

### Modeling Risk

- raw probabilistic output may be poorly calibrated under shift
- mitigated by calibration layer and drift-aware retraining

### System Risk

- training-serving skew can invalidate online metrics
- mitigated by shared feature definitions and logged prediction contracts

### Governance Risk

- policy optimization can improve short-term metrics while harming long-term outcomes
- mitigated by delayed-label monitoring, cohort analysis, and auditability
