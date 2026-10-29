# Credit Card Fraud Detection Pipeline for Extreme Class Imbalance

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-111111?style=flat&logo=xgboost)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline engineered to detect fraudulent transactions in an extremely imbalanced dataset (**0.173% positive class**). This project emphasizes strict data-leakage prevention, comparative imbalance handling (SMOTE vs. Cost-Sensitive Class Weighting), ensemble model benchmarking, and operational decision threshold tuning.

---

## Key Highlights & Performance

* **Dataset Volume:** 284,807 European credit card transactions (492 fraudulent).
* **PR-AUC Optimization:** Reached a top PR-AUC of **0.883** using XGBoost on a strict holdout test set.
* **Leakage Prevention:** Implemented a stratified split prior to scaling or oversampling, maintaining an untouched test set ($N=56,962$, 98 frauds).
* **Operational Threshold Tuning:** Demonstrated how probability threshold adjustments control the trade-off between direct financial losses (False Negatives) and operational investigation costs (False Positives).

### Comparative Model Summary

| Model Architecture | Resampling / Weighting | Precision | Recall | $F_1$-Score | **PR-AUC** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | SMOTE ($10\%$) | 0.352 | 0.888 | 0.504 | 0.752 |
| **Logistic Regression** | Class Weighted (`balanced`) | 0.061 | **0.918** | 0.114 | 0.717 |
| **Random Forest** | Class Weighted (`subsample`) | **0.960** | 0.735 | 0.832 | 0.860 |
| **Random Forest** | SMOTE ($10\%$) | 0.892 | 0.847 | **0.869** | 0.879 |
| **XGBoost** | SMOTE ($10\%$) | 0.856 | 0.847 | 0.851 | **0.883** |

---

## Pipeline Architecture

```
Raw Data Stream (284,807 rows)
      │
      ▼
Stratified Train/Test Split (80/20) ──────► Untouched Holdout Test Set
      │                                       (56,962 rows | 98 Frauds)
      ▼                                                   │
Robust Scaling (Amount & Time)                            │
 (Fit on Train, Transform on Test)                        │
      │                                                   │
      ▼                                                   │
Train Set Imbalance Remediation                           │
 ├── SMOTE Oversampling (10% target)                      │
 └── Cost-Sensitive Loss Weighting                        │
      │                                                   │
      ▼                                                   │
Model Training & Benchmarking                             │
 ├── Logistic Regression Baseline                         │
 ├── Random Forest Classifier                             │
 └── XGBoost Gradient Boosting                            │
      │                                                   │
      └─────────────────────────┬─────────────────────────┘
                                ▼
                  Model Evaluation & Threshold Tuning
```

---

## Repository Structure

```
.
├── README.md                      # Project documentation and summary
├── fraud_detection_case_study.md  # Detailed technical case study & report
├── data/
│   └── creditcard.csv             # Kaggle dataset (git-ignored)
├── notebooks/
│   └── fraud_detection_lab.ipynb  # End-to-end executable notebook
└── requirements.txt               # Project dependencies
```

---

## Quickstart & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Set up a virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the Dataset:**
   Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it inside the `data/` directory.

4. **Run the Analysis:**
   Launch VS Code or Jupyter Lab:
   ```bash
   jupyter lab
   ```
   Open `notebooks/fraud_detection_lab.ipynb` and execute all cells.

---

## Key Insights

1. **Accuracy is Misleading:** A model predicting $0$ (Legitimate) for every transaction achieves **99.828% accuracy** while missing 100% of fraud. PR-AUC and Precision-Recall metrics are mandatory for evaluating extreme imbalance.
2. **SMOTE vs. Class Weighting:** Oversampling via SMOTE maintained a strong balance between Precision and Recall. Aggressive class weighting (`balanced`) caused severe precision decay in linear models ($0.061$ precision due to $1,388$ false alarms).
3. **Primary Feature Importance:** `V14` emerged as the single most critical predictor across both tree architectures, driving **64.3%** of split importance in XGBoost.

---

## Author & Contact

* **Author:** Annus Jawad
* **Role:** Machine Learning Engineer / Data Scientist Trainee
* **LinkedIn:** [Your LinkedIn Link]
* **GitHub:** [Your GitHub Link]