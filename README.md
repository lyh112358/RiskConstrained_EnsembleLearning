# Kinematics-Inspired Behavioral Dynamics and Risk-Constrained Ensemble Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![KSEM 2026](https://img.shields.io/badge/Conference-KSEM_2026-brightgreen)](https://ksem.org/)

Official implementation for the KSEM 2026 accepted paper: **"Kinematics-Inspired Behavioral Dynamics and Risk-Constrained Ensemble Learning for Student Performance Prediction in Blended Environments"**.

---

## Overview
This repository provides a robust machine learning framework designed to solve two critical challenges in real-world tabular data mining:
- **Extreme Class Imbalance**: Handling datasets where failure/risk rates are < 5%.
- **Temporal Concept Drift**: Addressing non-stationary data distributions across different academic cohorts.

By integrating **Temporal Motivation Theory (TMT)** and **physical kinematics**, we engineered novel behavioral dynamics features. Furthermore, we designed a **Recall-Constrained Heterogeneous Ensemble** architecture using Bayesian optimization to prevent the model from falling into the "Accuracy Trap."

---

## Key Innovations

### 1. Kinematics-Inspired Feature Engineering
Beyond conventional static features (e.g., total scores, online duration), we extract higher-order dynamic patterns:
*   **Learning Acceleration (LAcc)**: Quantifies the temporal "rate of detachment" via second-order temporal difference.
*   **Relative Procrastination Index (RPI)**: Measures cohort-referenced procrastination habits, robust to task difficulty and release timing.

### 2. Risk-Constrained Bayesian Optimization
We embedded an asymmetric recall penalty directly into the objective function of Optuna's Tree-structured Parzen Estimator (TPE) sampler.

```python
# Pseudo-code for our asymmetric penalty mechanism
if current_recall < R_min:
    penalty = lambda_factor * (R_min - current_recall)
    objective_score = base_metric - penalty
```
*This strictly forces the optimizer to explore hyperparameter sub-spaces that guarantee sensitivity to the minority class (at-risk students), avoiding the precision paradox.*

### 3. Heterogeneous Stacking Ensemble
A robust fusion of gradient boosting (XGBoost, LightGBM), randomized trees (ExtraTrees), and probabilistic models. This dual-layer architecture mitigates variance and bias simultaneously, providing endogenous robustness against monotonic decay.

---

## Repository Structure
```text
Risk-Constrained-Behavioral-Dynamics/
├── data_preprocessing.py          # Penalty-based imputation and temporal feature engineering
├── blending_optimization.py       # Optuna-based Bayesian hyperparameter tuning with Recall constraints
├── dual_fusion_optimization.py    # Stacking heterogeneous base models & meta-learner calibration
├── lstm_baseline.py               # Cost-sensitive LSTM sequence modeling baseline
├── shap_analysis.py               # Global interpretability and feature importance analysis
├── requirements.txt
└── README.md
```

---

## Experimental Results
Cross-cohort validation on multidimensional educational data (2020–2025 cohorts) demonstrated exceptional robustness against concept drift and extreme imbalance (Imbalance Ratio up to 45.0:1):

| Metric | Proposed Framework (Ours) | XGBoost (Cost-Sensitive) | Deep Baseline (LSTM) |
| :--- | :---: | :---: | :---: |
| **Recall (At-risk)** | **82.35%** | 81.18% | 69.41% |
| **Specificity** | 71.82% | 67.17% | 71.19% |
| **G-mean** | **0.7691** | 0.7384 | 0.7029 |
| **Balanced-ACC** | **0.7709** | 0.7417 | 0.7030 |

Supported by SHAP analysis, early dynamic indicators (RPI & LAcc) exhibit significantly higher predictive power than late-stage static performance metrics.

---

## Installation & Usage

### 1. Setup Environment
```bash
git clone https://github.com/lyh112358/Risk-Constrained-Behavioral-Dynamics.git
cd Risk-Constrained-Behavioral-Dynamics
pip install -r requirements.txt
```

### 2. Execute Pipeline
```bash
# Step 1: Feature Extraction & Data Governance
python data_preprocessing.py

# Step 2: Constrained Optimization for Base Models
python blending_optimization.py

# Step 3: Heterogeneous Stacking Ensemble Evaluation
python dual_fusion_optimization.py

# Step 4: Interpretability (Generates SHAP Summary Plots)
python shap_analysis.py
```

---

## Citation
If you find our work, concepts (LAcc/RPI), or this repository useful for your research, please consider citing our paper:

```bibtex
@inproceedings{li2026kinematics,
  title={Kinematics-Inspired Behavioral Dynamics and Risk-Constrained Ensemble Learning for Student Performance Prediction in Blended Environments},
  author={Li, Yinghe},
  booktitle={Proceedings of the International Conference on Knowledge Science, Engineering and Management (KSEM)},
  year={2026}
}
```

---

## Contact & Developer
**Yinghe Li (李英赫)**
*   Undergraduate Student at Tang Aoqing Class, Jilin University
*   Email: [13339388066@163.com](mailto:13339388066@163.com)
*   GitHub: [https://github.com/lyh112358](https://github.com/lyh112358)

*Currently seeking 2026 Summer Internship opportunities in Data Mining / Machine Learning Engineering.*

---

## License
This project is licensed under the MIT License
