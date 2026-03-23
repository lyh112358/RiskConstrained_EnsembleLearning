# Kinematics-Inspired Behavioral Dynamics and Risk-Constrained Ensemble Learning for Student Performance Prediction

## Overview

This repository contains the implementation of a feature extraction and enhancement framework grounded in **Temporal Motivation Theory (TMT)** and psychological prior knowledge, along with a **Recall-Constrained Heterogeneous Ensemble Learning** model for student performance prediction in blended learning environments.

## Problem Statement

Real-world educational data presents a dual challenge:
- **Extreme class imbalance**: Failure rates below 10%
- **Distributional shifts**: Concept drift across academic cohorts

## Key Features

### Temporal Motivation Theory (TMT) Based Feature Engineering

Beyond conventional features, we innovatively construct:

- **Learning Engagement (LE)**: Engagement metrics from learning behavioral data
- **Learning Stability (LS)**: Stability indicators of learning patterns
- **Learning Acceleration (LAcc)**: Dynamics of learning pace
- **Relative Procrastination Index (RPI)**: Cohort-relative procrastination habits

### Recall-Constrained Heterogeneous Ensemble Learning

- **Bayesian Hyperparameter Optimization**: Recall threshold embedded into Optuna's TPE sampler
- **Multi-Strategy Fusion**: Stacking and soft voting at decision level
- **Confusion Matrix-Weighted Optimization**: Dynamically balances prediction coverage and precision
- **False Alarm Control**: Strictly controls false alarm rates while maximizing recall

## Project Structure

```
AcademicEarlyWarning_MachineLearning/
├── dual_fusion_optimization.py    # Two-layer stacking fusion with meta-features
├── blending_optimization.py       # Base model hyperparameter tuning
├── shap_analysis.py               # SHAP-based model interpretation
├── lstm_baseline.py               # LSTM neural network baseline
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- NumPy, Pandas
- Scikit-learn
- XGBoost, LightGBM, CatBoost
- Optuna
- SHAP
- PyTorch (for LSTM model)
- imbalanced-learn (optional, for SMOTE)

## Key Optimization Strategy

### Recall-Constrained Optimization

Our approach embeds a **recall threshold** into the Bayesian hyperparameter optimization process:

```
if recall < min_recall_threshold:
    penalty = (min_recall_threshold - recall) * penalty_factor
    final_score = score - penalty
```

This enforces high sensitivity towards the minority class (failing students) while maximizing overall performance.

### Confusion Matrix-Weighted Strategy

At the decision level, we employ a confusion matrix-weighted optimization to:
- Maximize recall for at-risk students
- Control false alarm rates
- Dynamically balance prediction coverage and precision

## Experimental Results

Cross-cohort validation on "Fundamentals of Programming" course (2020-2025 cohorts):

| Metric | Value |
|--------|-------|
| **Recall (At-risk)** | 82.35% |
| **Specificity (Passing)** | 71.82% |
| **G-mean** | 0.7691 |

## Usage

### 1. Base Model Hyperparameter Optimization

```bash
python blending_optimization.py
```

### 2. Dual-Layer Fusion Optimization

```bash
python dual_fusion_optimization.py
```

### 3. SHAP Analysis

```bash
python shap_analysis.py
```

### 4. LSTM Baseline

```bash
python lstm_baseline.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kinematics_inspired_ensemble_learning,
  title={Kinematics-Inspired Behavioral Dynamics and Risk-Constrained Ensemble Learning for Student Performance Prediction in Blended Environments},
  author={Yinghe Li (Jilin University)},
  year={2025}
}
```

## License

MIT License
