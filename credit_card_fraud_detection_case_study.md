# Machine Learning Pipeline for Extreme Imbalance Credit Card Fraud Detection

**Author:** Annus Jawad

**Role:** Data Scientist / Machine Learning Engineer Trainee

**Date:** September 28, 2026

**Environment:** Jupyter / VS Code, Python 3.14.4

**Libraries Used:** `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn` (`SMOTE`), `xgboost`, `matplotlib`, `seaborn`

## Executive Summary

Financial fraud detection presents a canonical extreme class imbalance challenge: fraudulent transactions constitute a tiny fraction of total volume, making conventional accuracy metrics meaningless.

This project implements an end-to-end Machine Learning production workflow on the ULB Kaggle Credit Card Fraud dataset (284,807 transactions; **0.173% fraud rate**). Key engineering highlights include:

1. **Strict Data Leakage Prevention:** Isolated train/test splitting before feature transformation or resampling to maintain strict evaluation boundaries.

2. **Imbalance Remediation Protocols:** Evaluated synthetic oversampling (**SMOTE**) against cost-sensitive loss adjustments (**Class Weighting**).

3. **Model Benchmarking:** Evaluated baseline Logistic Regression against Random Forest and XGBoost ensemble architectures on an untouched test partition.

4. **Precision-Recall Optimization:** Analyzed decision threshold tuning to balance financial loss (False Negatives) against operational triage costs (False Positives).

## Problem Statement & Metrics Strategy

In highly imbalanced regimes, a trivial baseline predicting "Legitimate" across all inputs yields **99.828% classification accuracy** while missing 100% of fraudulent activity.

$$
\text{Fraud Ratio} = \frac{492}{284,807} \approx 0.00173 \quad (0.173\%)
$$

### Primary Evaluation Metrics

* **PR-AUC (Area Under Precision-Recall Curve):** Primary optimization metric. Unlike ROC-AUC, PR-AUC isolates true positive rates relative to positive predictions without distortion from massive True Negative counts.

* **Recall (Sensitivity):** Fraction of actual fraud cases successfully intercepted.

* **Precision:** Ratio of flagged alerts that represent true fraudulent activity.

* $F_1$**-Score:** Harmonic mean of Precision and Recall ($F_1 = 2 \cdot \frac{P \cdot R}{P + R}$).

## End-to-End Pipeline Architecture

```
Raw Transaction Stream (284,807 rows)
           │
           ▼
Stratified Train/Test Split (80/20) ────► Holdout Test Set (56,962 rows; 98 Frauds)
           │                                 │ (Untouched until evaluation)
           ▼                                 │
Robust Scaling (Amount & Time)               │
  • Fitted on Train only                     │
           │                                 │
           ▼                                 │
Train Set Imbalance Handling                 │
  • Strategy A: SMOTE (Target = 10%)         │
  • Strategy B: Balanced Class Weights       │
           │                                 │
           ▼                                 │
Model Training & Optimization                │
  • Logistic Regression                      │
  • Random Forest Ensembles                  │
  • Gradient Boosted Trees (XGBoost)         │
           │                                 │
           └──────────────────┬──────────────┘
                              ▼
                 Comparative Test Evaluation

```

## Data Architecture & Exploratory Analysis

* **Dataset Volume:** 284,807 instances $\times$ 31 numeric features.

* **Feature Composition:**

  * `V1`–`V28`: PCA-transformed, anonymized continuous features.

  * `Amount`: Unscaled transaction value in Euros ($€$).

  * `Time`: Seconds elapsed from initial dataset observation.

  * `Class`: Target indicator ($0 = \text{Legitimate}$, $1 = \text{Fraud}$).

### Descriptive Transaction Statistics

| Transaction Class | Count | Percentage | Mean (€) | Median (€) | 75th Percentile (€) | Max (€) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Legitimate (**$0$**)** | 284,315 | 99.827% | 88.29 | 22.00 | 77.05 | 25,691.16 | 
| **Fraudulent (**$1$**)** | 492 | 0.173% | 122.21 | 9.25 | 105.89 | 2,125.87 | 
| **Total Cohort** | 284,807 | 100.00% | 88.35 | 22.00 | 77.16 | 25,691.16 | 

## Data Preprocessing & Leakage Mitigation

### 1. Stratified Partitioning

To prevent sample distribution skew across rare events, an $80/20$ stratified train/test split was executed prior to downstream scaling or oversampling:

$$
\text{Train Partition} = 227,845 \text{ rows } (394 \text{ frauds}); \quad \text{Test Partition} = 56,962 \text{ rows } (98 \text{ frauds})
$$

### 2. Robust Feature Scaling

Due to severe right-skewness in transaction `Amount`, features were transformed using `RobustScaler`, centering on the median and scaling via the Interquartile Range ($\text{IQR}$):

$$
X_{\text{scaled}} = \frac{X - Q_2(X_{\text{train}})}{Q_3(X_{\text{train}}) - Q_1(X_{\text{train}})}
$$

The transformation parameters were learned exclusively on $X_{\text{train}}$ and applied to $X_{\text{test}}$ without re-fitting.

### 3. Imbalance Handling Protocols

* **SMOTE Resampling:** Applied to the training set only to adjust the minority-to-majority class ratio to $1:10$ ($10\%$ ratio), growing training fraud samples from 394 to 22,745 instances.

* **Cost-Sensitive Class Weighting:** Parameterized loss penalties inversely proportional to class frequencies ($w_j = \frac{N}{k \cdot n_j}$), weighting fraud misclassifications $\approx 578\times$ heavier than legitimate instances.

## Model Benchmark & Experimental Results

All models were evaluated on the held-out test partition ($56,962$ transactions, $98$ ground-truth fraud instances) at a standard decision threshold ($t = 0.5$).

### Comparative Performance Matrix

| Model Architecture | Resampling / Weighting | True Positives (TP) | False Negatives (FN) | False Positives (FP) | Precision | Recall | $F_1$-Score | PR-AUC | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Logistic Regression** | SMOTE ($10\%$) | 87 | 11 | 160 | 0.352 | **0.888** | 0.504 | 0.752 | 
| **Logistic Regression** | Class Weighted (`balanced`) | 90 | 8 | 1,388 | 0.061 | **0.918** | 0.114 | 0.717 | 
| **Random Forest** | Class Weighted (`subsample`) | 72 | 26 | **3** | **0.960** | 0.735 | 0.832 | 0.860 | 
| **Random Forest** | SMOTE ($10\%$) | 83 | 15 | **10** | 0.892 | 0.847 | **0.869** | 0.879 | 
| **XGBoost** | SMOTE ($10\%$) | 83 | 15 | 14 | 0.856 | 0.847 | 0.851 | **0.883** | 

## Decision Threshold Tuning Analysis

Using the optimal XGBoost model, probability decision thresholds ($t$) were systematically varied across $X_{\text{test}}$ to map operational trade-offs:

| Decision Threshold ($t$) | True Positives (TP) | False Negatives (FN) | False Positives (FP) | Precision | Recall | Primary Operational Focus | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| $0.05$ | 88 | 10 | 87 | 0.503 | 0.898 | Maximum fraud interception | 
| $0.10$ | 88 | 10 | 55 | 0.615 | 0.898 | High recall, reduced false alerts | 
| $0.30$ | 86 | 12 | 24 | 0.782 | 0.878 | Balanced operational operating point | 
| $0.50$ **(Default)** | 83 | 15 | 14 | 0.856 | 0.847 | Standard baseline threshold | 
| $0.70$ | 81 | 17 | 8 | 0.910 | 0.827 | High precision, low alert volume | 
| $0.99$ | 74 | 24 | 2 | 0.974 | 0.755 | Strict auto-blocking regime | 

## Feature Importance Attribution

Feature attributions extracted from ensemble tree models identified primary fraud indicators across the latent PCA space:

| Rank | XGBoost Feature | Importance Score | Random Forest Feature | Importance Score | 
| ----- | ----- | ----- | ----- | ----- | 
| **1** | **`V14`** | **0.643** | **`V14`** | **0.212** | 
| **2** | `V4` | 0.043 | `V10` | 0.132 | 
| **3** | `V17` | 0.043 | `V17` | 0.115 | 
| **4** | `V10` | 0.029 | `V12` | 0.108 | 
| **5** | `V8` | 0.019 | `V16` | 0.076 | 
| **6** | `V12` | 0.018 | `V11` | 0.054 | 
| **7** | `V13` | 0.013 | `V3` | 0.053 | 
| **8** | `V1` | 0.013 | `V4` | 0.038 | 
| **9** | `V3` | 0.012 | `V9` | 0.032 | 
| **10** | `Time` | 0.012 | `V2` | 0.021 | 

> **Key Analytical Observation:** Both tree-based architectures converged on `V14` as the single most critical predictor. XGBoost allocated $64.3\%$ of total split importance to `V14`, highlighting heavy reliance on specific principal component variance.

## Strategic Engineering Insights

1. **Model Architecture Comparison:** Non-linear ensemble architectures (Random Forest, XGBoost) significantly outperformed linear baselines, elevating PR-AUC from $\approx 0.75$ to $\approx 0.88$. XGBoost achieved equivalent performance to Random Forest while demonstrating an $8\times$ reduction in computational training latency ($3\text{s}$ vs $25\text{s}$).

2. **Resampling Dynamics:** Oversampling via SMOTE maintained balanced precision-recall characteristics, whereas aggressive class weighting (`balanced`) caused severe precision degradation in linear models (dropping to $0.061$).

3. **Operational Threshold Tuning:** Operating points must be configured based on asymmetry in business costs:

   * **Low Threshold (**$t = 0.10$**):** Optimal when the cost of missed fraud exceeds operational investigator triage capacity costs.

   * **High Threshold (**$t = 0.70+$**):** Optimal when preventing customer friction (false declines) is prioritized over minor fraud leakage.

## Engineering Limitations & Future Extensions

* **Test Partition Size Constraint:** Held-out testing relied on 98 positive fraud events; single instance variations introduce $\approx 1\%$ variance in recall statistics. Repeated K-Fold cross-validation is recommended for production validation.

* **Temporal Dynamics:** Dataset features represent a static two-day historical window. Production systems require time-based split validation ($TemporalSplit$) to simulate model drift against evolving fraud tactics.

* **Feature Anonymization Constraints:** Feature attributions (`V1`–`V28`) reflect abstract PCA components, limiting domain-specific interpretability without raw feature mapping.