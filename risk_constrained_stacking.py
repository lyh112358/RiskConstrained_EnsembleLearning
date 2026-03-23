"""
Dual-Layer Fusion Model with Recall-Constrained Ensemble Learning

A recall-constrained heterogeneous ensemble learning system for academic early warning
in blended learning environments with extreme class imbalance.

Key Components:
- Temporal Motivation Theory (TMT) based feature engineering (LE, LS, LAcc, RPI)
- Recall threshold embedded into Bayesian hyperparameter optimization (Optuna TPE)
- Multi-strategy fusion: Stacking and soft voting at decision level
- Confusion matrix-weighted optimization for false alarm rate control
- SHAP-based model interpretability analysis

This module implements the two-layer fusion mechanism that:
1. Generates meta-features from base models (XGBoost, LightGBM, ET, etc.)
2. Applies recall-constrained optimization with penalty mechanism
3. Fuses predictions via Stacking or Voting strategies

Reference: "Kinematics-Inspired Behavioral Dynamics and Risk-Constrained Ensemble Learning
for Student Performance Prediction in Blended Environments"

Author: Yinghe Li (Jilin University)
License: MIT
"""

import os
import warnings
import datetime
import logging
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import optuna

from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import logging

logging.getLogger('lightgbm').setLevel(logging.ERROR)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from xgboost.callback import EarlyStopping
from lightgbm import early_stopping, log_evaluation

import shap

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ['LIGHTGBM_VERBOSITY'] = '-1'
optuna.logging.set_verbosity(optuna.logging.ERROR)

use_gpu = False

class GlobalConfig:
    ENABLE_PASSTHROUGH_FEATURES = False

    FUSION_STRATEGY = 'stacking'

    optimization_objective = 'gmean'

    w_specificity_in_pre = 0.5
    w_recall_in_pre = 0.5

    MIN_RECALL_RATE = 0.8
    PENALTY_COEFFICIENT = 0.6

    SHAP_ANALYSIS_ON = False
    SHAP_BASE_SAMPLES = 300
    SHAP_FUSION_SAMPLES = 300

    random_seed = 42

    max_search_combinations = 50
    random_search_rounds = max_search_combinations // 4
    early_stopping_rounds = 300

    TOP_M_RESULTS = 30

    MODEL_SELECTION = {
        'LightGBM': (
            "path/to/your/LightGBM_params.xlsx",
            [0]
        ),
        'HGB': (
            "path/to/your/HGB_params.xlsx",
            [0]
        ),
        'XGBoost': (
            "path/to/your/XGBoost_params.xlsx",
            [0]
        )
    }

    DATA_PATHS = {
        'source1': 'path/to/your/Merged_2020_全特征_updated.csv',
        'source2': 'path/to/your/Merged_2021_全特征_updated.csv',
        'source3': 'path/to/your/Merged_2022_全特征_updated.csv',
        'source4': 'path/to/your/Merged_2023_全特征_updated.csv',
        'source5': 'path/to/your/Merged_2024_全特征_updated.csv',
        'target': 'path/to/your/Merged_2025_全特征_updated.csv'
    }

    FUSION_METHODS = ['XGBoost', 'LightGBM', 'BNB', 'ET', 'HGB', 'LR', 'ENLR', 'Ridge', 'KNN', 'RF', 'DT', 'LDA']

    SEARCH_SPACES = {
        'XGBoost': ({
            'learning_rate': [0.001, 0.5], 'n_estimators': [100, 5000], 'max_depth': [3, 20],
            'min_child_weight': [0.1, 100], 'subsample': [0.4, 1.0], 'colsample_bytree': [0.4, 1.0],
            'gamma': [0, 20], 'reg_lambda': [0.0001, 1000], 'reg_alpha': [0.0001, 1000]
        }, {'tree_method': 'hist', 'device': 'cuda' if use_gpu else 'cpu', 'eval_metric': 'logloss', 'n_jobs': -1}),

        'ET': ({
            'n_estimators': [100, 800], 'max_depth': [5, 25], 'min_samples_split': [2, 20],
            'min_samples_leaf': [2, 10], 'max_features': ['sqrt', 'log2', 0.5, 0.9], 'bootstrap': [True, False]
        }, {'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}),

        'LightGBM': ({
            'learning_rate': [0.001, 0.5], 'n_estimators': [100, 5000], 'num_leaves': [20, 1024],
            'max_depth': [-1, 50], 'min_child_samples': [5, 300], 'feature_fraction': [0.4, 1.0],
            'bagging_fraction': [0.4, 1.0], 'lambda_l1': [1e-8, 100], 'lambda_l2': [1e-8, 100]
        }, {'random_state': 42, 'objective': 'binary', 'device': 'gpu' if use_gpu else 'cpu', 'verbosity': -1, 'n_jobs': -1}),

        'RF': ({
            'n_estimators': [100, 800], 'max_depth': [5, 15], 'min_samples_split': [2, 20],
            'min_samples_leaf': [2, 10], 'max_features': ['sqrt', 'log2', 0.5, 0.8],
        }, {'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}),

        'KNN': ({'n_neighbors': [3, 200], 'weights': ['uniform', 'distance'], 'p': [1, 2]}, {'n_jobs': -1}),

        'HGB': ({
            'learning_rate': [0.001, 0.5], 'max_iter': [100, 5000], 'max_depth': [1, 50, None],
            'min_samples_leaf': [5, 300], 'l2_regularization': [1e-8, 100]
        }, {'random_state': 42, 'early_stopping': True}),

        'LR': ({
            'C': [0.001, 100.0], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga'], 'class_weight': ['balanced', None]
        }, {'random_state': 42, 'n_jobs': -1, 'max_iter': 1000}),

        'ENLR': ({
            'C': [0.001, 100.0], 'l1_ratio': [0.1, 0.9],
        }, {'random_state': 42, 'penalty': 'elasticnet', 'solver': 'saga', 'max_iter': 2000, 'class_weight': 'balanced', 'n_jobs': -1}),

        'GNB': ({'var_smoothing': [1e-15, 1e-1]}, {}),
        'BNB': ({'alpha': [0.0001, 1000.0], 'binarize': [0.0, 1.0]}, {}),

        'DT': ({
            'max_depth': [1, 50, None], 'min_samples_split': [2, 100], 'min_samples_leaf': [1, 50],
            'class_weight': ['balanced', None]
        }, {'random_state': 42}),

        'LDA': ({'solver': ['svd', 'lsqr'], 'tol': [1e-8, 1e-1]}, {}),
        'Ridge': ({'alpha': [0.00001, 10000.0], 'max_iter': [100, 10000], 'class_weight': ['balanced', None]}, {'random_state': 42}),
    }

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    mode_str = "WithPassthrough" if ENABLE_PASSTHROUGH_FEATURES else "MetaOnly"
    target_str = optimization_objective
    if target_str == 'pre':
        target_str = f"pre_spe{w_specificity_in_pre}_rec{w_recall_in_pre}"

    if FUSION_STRATEGY == 'voting':
        search_info = "Voting_NoSearch"
    else:
        search_info = f"{max_search_combinations}次"

    OUTPUT_DIR = fr"./results/dual_fusion_{timestamp}_{FUSION_STRATEGY}_{target_str}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_optimal_threshold(y_true, y_prob):
    target = GlobalConfig.optimization_objective
    min_recall = GlobalConfig.MIN_RECALL_RATE

    thresholds = np.linspace(0.01, 0.99, 200)

    if target == 'auc': return 0.5

    valid_candidates = []
    all_candidates = []

    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        specificity = tn / (tn + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        score = 0
        if target == 'pre':
            score = GlobalConfig.w_specificity_in_pre * specificity + GlobalConfig.w_recall_in_pre * recall
        elif target == 'gmean':
            score = np.sqrt(recall * specificity)
        elif target == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif target == 'acc':
            score = accuracy_score(y_true, y_pred)

        all_candidates.append((recall, t, score))

        if recall >= min_recall:
            valid_candidates.append((score, t))

    if valid_candidates:
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        return valid_candidates[0][1]

    all_candidates.sort(key=lambda x: x[0], reverse=True)
    return all_candidates[0][1]

def evaluate_detailed(y_true, y_prob, model_name="", fixed_threshold=None):
    if fixed_threshold is not None: opt = fixed_threshold
    else: opt = find_optimal_threshold(y_true, y_prob)

    y_pred = (y_prob > opt).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try: auc_s = roc_auc_score(y_true, y_prob)
    except: auc_s = 0.5
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    spec = tn / (tn + fp + 1e-9)
    gmean = np.sqrt(rec * spec)
    bal_acc = (rec + spec) / 2
    weighted_score = GlobalConfig.w_specificity_in_pre * spec + GlobalConfig.w_recall_in_pre * rec
    fpr = fp / (tn + fp + 1e-9)
    fnr = fn / (tp + fn + 1e-9)

    return {
        "Model": model_name, "Optimization_Target": GlobalConfig.optimization_objective,
        "Weighted_Score": round(weighted_score, 4), "G_Mean": round(gmean, 4),
        "ACC": round(acc, 4), "AUC": round(auc_s, 4), "F1": round(f1, 4),
        "Precision": round(pre, 4), "Recall (Fail Capture)": round(rec, 4), "Specificity (Pass Correct)": round(spec, 4),
        "Balanced_ACC": round(bal_acc, 4), "FPR": round(fpr, 4), "FNR": round(fnr, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn), "Threshold": round(opt, 4)
    }

def clean_param(param_name, param_value):
    if pd.isna(param_value): return None
    if str(param_value).lower() in ['none', 'nan']: return None
    int_params = ['n_estimators', 'max_depth', 'num_leaves', 'min_child_weight', 'min_samples_split', 'min_samples_leaf', 'n_neighbors', 'max_iter', 'min_child_samples']
    bool_params = ['fit_intercept', 'early_stopping', 'bootstrap', 'fit_prior', 'probability', 'verbose', 'replace', 'shuffle']
    float_params = ['learning_rate', 'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda', 'alpha', 'l1_ratio']
    try:
        if param_name in int_params: return int(float(param_value))
        if param_name in bool_params:
            if isinstance(param_value, bool): return param_value
            val_str = str(param_value).lower()
            return True if val_str in ['true', 'yes'] else (False if val_str in ['false', 'no'] else bool(float(param_value)))
        if param_name in float_params: return float(param_value)
        return param_value
    except: return None

def load_specific_params(file_path, selected_indices):
    try:
        df = pd.read_excel(file_path)
        exclude = [
            'Model', 'ACC', 'AUC', 'F1', 'Precision', 'Recall', 'Specificity',
            'Balanced_ACC', 'G-Mean', 'G_Mean', 'Weighted_Score',
            'TP', 'FP', 'TN', 'FN', 'Threshold', 'Fail_Miss_Rate',
            'Recall (Fail Capture)', 'Specificity (Pass Correct)',
            'FPR', 'FNR', 'Wrongly_Accused_Rate', 'Correct_Grade0_Rate'
        ]
        param_cols = [c for c in df.columns if c not in exclude and 'Contrib' not in c and 'Weight' not in c]
        result_list = []
        for idx in selected_indices:
            if idx < 0 or idx >= len(df): continue
            row_data = df.iloc[idx]
            model_id = str(row_data['Model']) if 'Model' in df.columns else f"Row{idx}"
            params = {}
            for k in param_cols:
                v = row_data[k]
                if pd.notna(v):
                    cleaned = clean_param(k, v)
                    if cleaned is not None:
                        params[k] = cleaned
            if params:
                result_list.append((model_id, params))
        return result_list
    except Exception as e:
        print(f"Error loading params from {file_path}: {e}")
        return []

def build_base_model(model_type, params):
    try:
        p = params.copy()
        no_rs_models = ['BNB', 'GNB', 'KNN', 'LDA', 'Ridge']
        if model_type in no_rs_models and 'random_state' in p:
            del p['random_state']

        if model_type == 'LightGBM':
            p['verbosity'] = -1
            p['verbose'] = -1
            p['importance_type'] = 'gain'
            return lgb.LGBMClassifier(**p)
        if model_type == 'XGBoost':
            return xgb.XGBClassifier(**p)
        if model_type == 'HGB':
            return HistGradientBoostingClassifier(**p)
        if model_type == 'RF':
            return RandomForestClassifier(**p)
        if model_type == 'ET':
            return ExtraTreesClassifier(**p)
        if model_type == 'BNB':
            return BernoulliNB(**p)
        if model_type == 'GNB':
            return GaussianNB(**p)
        if model_type in ['LR', 'ENLR']:
            return LogisticRegression(**p)
        if model_type == 'KNN':
            return KNeighborsClassifier(**p)
        if model_type == 'DT':
            return DecisionTreeClassifier(**p)
        if model_type == 'LDA':
            return LDA(**p)
        if model_type == 'Ridge':
            return CalibratedClassifierCV(RidgeClassifier(**p), method='sigmoid', cv=3)

        return None
    except Exception as e:
        return None

def calculate_shap_values(model, X_sample, model_type="generic"):
    try:
        shap_values = None

        tree_model_types = ['LightGBM', 'XGBoost', 'CatBoost', 'RF', 'ET', 'DT', 'GBC', 'HGB']
        is_tree_model = model_type in tree_model_types

        if is_tree_model:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]

        elif model_type in ['LR', 'ENLR']:
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)

        elif model_type == 'Ridge':
            try:
                inner_model = model.calibrated_classifiers_[0].estimator
                explainer = shap.LinearExplainer(inner_model, X_sample)
                shap_values = explainer.shap_values(X_sample)
            except:
                f = lambda x: model.predict_proba(x)[:, 1]
                background = shap.kmeans(X_sample, min(20, len(X_sample)))
                explainer = shap.KernelExplainer(f, background)
                shap_values = explainer.shap_values(X_sample)

        elif model_type == 'LDA':
            f = lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x)
            background = shap.kmeans(X_sample, min(20, len(X_sample))) if len(X_sample) > 20 else X_sample
            explainer = shap.KernelExplainer(f, background)
            shap_values = explainer.shap_values(X_sample)

        else:
            background = shap.kmeans(X_sample, min(20, len(X_sample))) if len(X_sample) > 50 else X_sample
            f = lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x)
            explainer = shap.KernelExplainer(f, background)
            shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        return shap_values

    except Exception as e:
        print(f"!!! SHAP ERROR ({model_type}): {str(e)} !!!")
        return np.zeros(X_sample.shape)

class FusionSystem:
    def __init__(self, config):
        self.cfg = config
        self.meta_feature_names = []
        self.fold_ids = None
        self.base_model_global_importances = {}

    def load_data(self):
        print(f"=== Loading Data (Min Recall Constraint: {self.cfg.MIN_RECALL_RATE}) ===")
        dfs = []
        source_keys = ['source1', 'source2', 'source3', 'source4', 'source5']

        for i, key in enumerate(source_keys):
            path = self.cfg.DATA_PATHS.get(key)
            if path and os.path.exists(path):
                df = pd.read_csv(path)
                df['custom_fold_id'] = -1
                skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.cfg.random_seed)
                if 'grade_label' in df.columns:
                    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df['grade_label'])):
                        current_global_fold = (i * 2) + fold_idx
                        df.iloc[val_idx, df.columns.get_loc('custom_fold_id')] = current_global_fold
                else:
                    raise ValueError(f"Missing 'grade_label' in {key}")
                dfs.append(df)
                print(f"Loaded {key}: {len(df)} samples -> Folds {i*2}, {i*2+1}")
            else:
                print(f"Warning: {key} path not found or empty.")

        source = pd.concat(dfs, ignore_index=True)
        target = pd.read_csv(self.cfg.DATA_PATHS['target'])

        exclude = ['SID', 'grade', 'grade_label', 'custom_fold_id','MOOCgrade']
        features = [c for c in source.columns if c not in exclude and 'RPI' not in c and 'LS' not in c and 'LE' not in c]
        features = [c for c in features if c in target.columns]

        X_src = source[features].replace([np.inf, -np.inf], np.nan).values
        y_src = source['grade_label'].values
        self.fold_ids = source['custom_fold_id'].values

        X_tgt = target[features].replace([np.inf, -np.inf], np.nan).values
        y_tgt = target['grade_label'].values

        imputer = SimpleImputer(strategy='mean')
        X_src = imputer.fit_transform(X_src)
        X_tgt = imputer.transform(X_tgt)

        scaler = StandardScaler()
        X_src = scaler.fit_transform(X_src)
        X_tgt = scaler.transform(X_tgt)

        self.X_train_raw, self.y_train = X_src, y_src
        self.X_test_raw, self.y_test = X_tgt, y_tgt
        self.features = features

        print(f"Total Train Shape: {self.X_train_raw.shape}, Test Shape: {self.X_test_raw.shape}")

    def generate_meta_features(self):
        all_models_data = []
        for key, (path, indices) in self.cfg.MODEL_SELECTION.items():
            if not os.path.exists(path): continue

            base_model_type = ''.join([c for c in key if not c.isdigit()])

            loaded_items = load_specific_params(path, indices)
            for m_id, params in loaded_items:
                all_models_data.append((base_model_type, m_id, params))

        if not all_models_data: raise ValueError("No valid model parameters loaded!")

        total_models = len(all_models_data)
        self.meta_train = np.zeros((self.X_train_raw.shape[0], total_models))
        self.meta_test = np.zeros((self.X_test_raw.shape[0], total_models))
        self.meta_feature_names = []

        custom_cv = PredefinedSplit(test_fold=self.fold_ids)

        shap_writer = None
        if self.cfg.SHAP_ANALYSIS_ON:
            shap_output_path = os.path.join(self.cfg.OUTPUT_DIR, f"BaseModels_SHAP_Analysis_{self.cfg.SHAP_BASE_SAMPLES}Rounds.xlsx")
            shap_writer = pd.ExcelWriter(shap_output_path, engine='xlsxwriter')
            print(f"--- Base Model SHAP Analysis Enabled (Rounds: {self.cfg.SHAP_BASE_SAMPLES}) ---")

        print(f"\n=== Generating Meta-Features (10-Fold CV) - {total_models} Base Models ===")
        with tqdm(total=total_models, desc="Processing Base Models") as pbar:
            for col_idx, (model_type, model_id, params) in enumerate(all_models_data):
                feature_name = f"Meta_{model_id}"
                self.meta_feature_names.append(feature_name)

                try:
                    model = build_base_model(model_type, params)
                    if model is None:
                        pbar.update(1)
                        continue

                    oof_pred = np.zeros(self.X_train_raw.shape[0])
                    for train_ix, val_ix in custom_cv.split(self.X_train_raw, self.y_train):
                        X_tr_fold, y_tr_fold = self.X_train_raw[train_ix], self.y_train[train_ix]
                        X_val_fold = self.X_train_raw[val_ix]
                        model.fit(X_tr_fold, y_tr_fold)
                        if hasattr(model, "predict_proba"): val_pred = model.predict_proba(X_val_fold)[:, 1]
                        else: val_pred = model.predict(X_val_fold)
                        oof_pred[val_idx] = val_pred

                    self.meta_train[:, col_idx] = oof_pred

                    model.fit(self.X_train_raw, self.y_train)
                    if hasattr(model, "predict_proba"): test_pred = model.predict_proba(self.X_test_raw)[:, 1]
                    else: test_pred = model.predict(self.X_test_raw)
                    self.meta_test[:, col_idx] = test_pred

                    if self.cfg.SHAP_ANALYSIS_ON:
                        try:
                            bg_data = shap.utils.sample(self.X_train_raw, self.cfg.SHAP_BASE_SAMPLES)
                            shap_vals = calculate_shap_values(model, bg_data, model_type)

                            if shap_vals.ndim == 2:
                                global_imp = np.abs(shap_vals).mean(axis=0)
                            else:
                                global_imp = np.abs(shap_vals)

                            self.base_model_global_importances[feature_name] = global_imp

                            shap_df = pd.DataFrame({
                                'Feature': self.features,
                                'Mean_Abs_SHAP': global_imp
                            }).sort_values('Mean_Abs_SHAP', ascending=False)

                            clean_name = f"{model_id}".replace("Classifier", "")
                            sheet_name = f"{clean_name}_{col_idx}"[:30]
                            shap_df.to_excel(shap_writer, sheet_name=sheet_name, index=False)

                        except Exception as shap_e:
                            print(f"Failed SHAP for {feature_name}: {shap_e}")
                            self.base_model_global_importances[feature_name] = np.zeros(len(self.features))

                except Exception as e:
                    print(f"Error generating {feature_name}: {e}")
                    self.base_model_global_importances[feature_name] = np.zeros(len(self.features))

                pbar.update(1)

        if shap_writer:
            shap_writer.close()
            print(f"Base Model SHAP results saved to: {shap_output_path}")

        valid_idx = ~np.all(self.meta_train == 0, axis=0)
        self.meta_train = self.meta_train[:, valid_idx]
        self.meta_test = self.meta_test[:, valid_idx]

        self.meta_feature_names = [n for i, n in enumerate(self.meta_feature_names) if valid_idx[i]]

        if self.cfg.ENABLE_PASSTHROUGH_FEATURES:
            self.X_train_final = np.hstack([self.X_train_raw, self.meta_train])
            self.X_test_final = np.hstack([self.X_test_raw, self.meta_test])
            self.final_feature_names = self.features + self.meta_feature_names
        else:
            self.X_train_final = self.meta_train
            self.X_test_final = self.meta_test
            self.final_feature_names = self.meta_feature_names

        print(f"Final Input Shape: {self.X_train_final.shape}")

    def _get_optuna_params(self, method, trial):
        if method not in self.cfg.SEARCH_SPACES: return {}, {}
        space_config, fixed_params = self.cfg.SEARCH_SPACES[method]
        params = fixed_params.copy()
        for name, value_range in space_config.items():
            if isinstance(value_range, list):
                is_numeric_list = len(value_range) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value_range)
                is_mixed = any(isinstance(x, str) or x is None for x in value_range)
                if is_numeric_list and not is_mixed:
                    low, high = value_range[0], value_range[1]
                    use_log = (low > 0 and (high / low >= 100))
                    if isinstance(low, int) and isinstance(high, int) and name not in ['learning_rate', 'gamma', 'l2_regularization']:
                         params[name] = trial.suggest_int(name, low, high, log=use_log)
                    else:
                        params[name] = trial.suggest_float(name, low, high, log=use_log)
                else:
                    params[name] = trial.suggest_categorical(name, value_range)
            else:
                 params[name] = trial.suggest_categorical(name, [value_range])
        return params

    def optimize_fusion(self, method):
        if method not in self.cfg.SEARCH_SPACES: return pd.DataFrame(), pd.DataFrame()

        def objective(trial):
            params = self._get_optuna_params(method, trial)
            custom_cv = PredefinedSplit(test_fold=self.fold_ids)

            oof_y_true = []
            oof_y_prob = []

            for train_idx, val_idx in custom_cv.split(self.X_train_final, self.y_train):
                X_tr, y_tr = self.X_train_final[train_idx], self.y_train[train_idx]
                X_val, y_val = self.X_train_final[val_idx], self.y_train[val_idx]

                model = build_base_model(method, params)
                if model is None: return 0
                try:
                    if method == 'XGBoost':
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    elif method == 'LightGBM':
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                                  callbacks=[early_stopping(self.cfg.early_stopping_rounds, verbose=False), log_evaluation(0)])
                    elif method == 'HGB' and params.get('early_stopping'):
                        model.fit(X_tr, y_tr)
                    else:
                        model.fit(X_tr, y_tr)
                except: return 0

                if hasattr(model, "predict_proba"): prob = model.predict_proba(X_val)[:, 1]
                else: prob = model.predict(X_val)

                oof_y_true.append(y_val)
                oof_y_prob.append(prob)

            y_true_all = np.concatenate(oof_y_true)
            y_prob_all = np.concatenate(oof_y_prob)

            opt_t = find_optimal_threshold(y_true_all, y_prob_all)

            y_pred_all = (y_prob_all > opt_t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).ravel()

            global_recall = tp / (tp + fn + 1e-9)
            global_spec = tn / (tn + fp + 1e-9)

            target = self.cfg.optimization_objective
            final_score = 0
            if target == 'gmean': final_score = np.sqrt(global_recall * global_spec)
            elif target == 'pre': final_score = self.cfg.w_specificity_in_pre * global_spec + self.cfg.w_recall_in_pre * global_recall
            elif target == 'acc': final_score = accuracy_score(y_true_all, y_pred_all)
            elif target == 'auc':
                try: final_score = roc_auc_score(y_true_all, y_prob_all)
                except: final_score = 0.5

            trial.set_user_attr("best_threshold", float(opt_t))

            if global_recall < self.cfg.MIN_RECALL_RATE:
                penalty = (self.cfg.MIN_RECALL_RATE - global_recall) * self.cfg.PENALTY_COEFFICIENT
                final_score = final_score - penalty

            return final_score

        sampler = optuna.samplers.TPESampler(n_startup_trials=self.cfg.random_search_rounds, seed=self.cfg.random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        print(f"\nOptimization Start: {method} (Target: {self.cfg.optimization_objective} | Constraint: Recall >= {self.cfg.MIN_RECALL_RATE})")
        with tqdm(total=self.cfg.max_search_combinations, ncols=100) as pbar:
            def callback(study, trial):
                pbar.update(1)
                best_val = study.best_value if study.best_value is not None else -1
                pbar.set_postfix({'best_score': f"{best_val:.4f}"})
            study.optimize(objective, n_trials=self.cfg.max_search_combinations, callbacks=[callback], n_jobs=1)

        results = []
        fusion_shap_data = []

        valid_trials = [t for t in study.trials if t.value is not None and t.value > -0.5]
        top_trials = sorted(valid_trials, key=lambda t: t.value, reverse=True)[:self.cfg.TOP_M_RESULTS]

        if not top_trials:
            return pd.DataFrame(), []

        rank_counter = 1
        for trial in top_trials:
            best_params = {}
            fixed = self.cfg.SEARCH_SPACES[method][1]
            best_params.update(fixed)
            best_params.update(trial.params)

            learned_threshold = trial.user_attrs.get("best_threshold", 0.5)

            final_model = build_base_model(method, best_params)
            final_model.fit(self.X_train_final, self.y_train)

            if hasattr(final_model, "predict_proba"):
                y_prob = final_model.predict_proba(self.X_test_final)[:, 1]
            else:
                y_prob = final_model.predict(self.X_test_final)

            model_name_tag = f"{method}_Stacking_Top{rank_counter}"
            metrics = evaluate_detailed(self.y_test, y_prob, model_name=model_name_tag, fixed_threshold=learned_threshold)
            weights = self._extract_importance(final_model, method)
            metrics.update(weights)
            metrics.update(best_params)
            results.append(metrics)

            if self.cfg.SHAP_ANALYSIS_ON and rank_counter <= 50:
                try:
                    bg_data_fusion = shap.utils.sample(self.X_train_final, self.cfg.SHAP_FUSION_SAMPLES)
                    meta_shap_vals = calculate_shap_values(final_model, bg_data_fusion, method)
                    if meta_shap_vals.ndim == 2: meta_imp = np.abs(meta_shap_vals).mean(axis=0)
                    else: meta_imp = np.abs(meta_shap_vals)

                    weighted_original_imp = np.zeros(len(self.features))
                    for idx, feat_name in enumerate(self.final_feature_names):
                        if feat_name in self.base_model_global_importances:
                            weight = meta_imp[idx]
                            base_imp_vec = self.base_model_global_importances[feat_name]
                            weighted_original_imp += weight * base_imp_vec

                    final_shap_df = pd.DataFrame({
                        'Feature': self.features,
                        'Weighted_SHAP_Importance': weighted_original_imp
                    }).sort_values('Weighted_SHAP_Importance', ascending=False)
                    fusion_shap_data.append((model_name_tag, final_shap_df))
                except Exception as e:
                    print(f"Fusion SHAP Error for {model_name_tag}: {e}")

            rank_counter += 1

        return pd.DataFrame(results), fusion_shap_data

    def _extract_importance(self, model, method):
        imp_dict = {}
        try:
            importances = None
            if hasattr(model, 'feature_importances_'): importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                if method == 'Ridge': importances = np.abs(model.calibrated_classifiers_[0].estimator.coef_[0])
                else: importances = np.abs(model.coef_[0])

            if importances is not None:
                total = np.sum(importances) + 1e-9
                norm_imp = (importances / total) * 100
                tmp_df = pd.DataFrame({'Name': self.final_feature_names, 'Imp': norm_imp}).sort_values('Imp', ascending=False)
                for i in range(min(20, len(tmp_df))):
                    row = tmp_df.iloc[i]
                    imp_dict[f"Top{i+1}_Feature"] = row['Name']
                    imp_dict[f"Top{i+1}_Contrib(%)"] = round(row['Imp'], 4)
        except: pass
        return imp_dict

    def run_voting(self):
        print(f"\n>>> Running Fusion Strategy: Soft Voting")
        if self.meta_train.shape[1] == 0: return
        y_prob_train_voting = np.mean(self.meta_train, axis=1)
        best_threshold_voting = find_optimal_threshold(self.y_train, y_prob_train_voting)
        print(f"Voting: Learned Threshold from Training Data (OOF): {best_threshold_voting:.4f}")

        y_prob_test_voting = np.mean(self.meta_test, axis=1)
        metrics = evaluate_detailed(self.y_test, y_prob_test_voting, model_name="Soft_Voting_Average", fixed_threshold=best_threshold_voting)
        metrics['Included_Models_Count'] = self.meta_test.shape[1]

        df = pd.DataFrame([metrics])
        save_path = os.path.join(self.cfg.OUTPUT_DIR, f"Fusion_Result_Voting.xlsx")
        df.to_excel(save_path, index=False)
        print(f"Voting Finished. Score: {metrics['Weighted_Score']:.4f}")

    def run(self):
        self.load_data()
        self.generate_meta_features()
        if self.cfg.FUSION_STRATEGY == 'voting': self.run_voting()
        else:
            all_dfs = []

            fusion_shap_writer = None
            if self.cfg.SHAP_ANALYSIS_ON:
                p = os.path.join(self.cfg.OUTPUT_DIR, f"Top10_Fusion_Weighted_SHAP_{self.cfg.SHAP_FUSION_SAMPLES}Rounds.xlsx")
                fusion_shap_writer = pd.ExcelWriter(p, engine='xlsxwriter')

            for method in self.cfg.FUSION_METHODS:
                print(f"\n>>> Running Fusion Optimization for: {method}")
                df, shap_data_list = self.optimize_fusion(method)

                if not df.empty:
                    save_path = os.path.join(self.cfg.OUTPUT_DIR, f"Fusion_Result_{method}.xlsx")
                    df.to_excel(save_path, index=False)
                    all_dfs.append(df)

                    if fusion_shap_writer and shap_data_list:
                        for m_name, s_df in shap_data_list:
                            s_name = f"{m_name}"[:30]
                            s_df.to_excel(fusion_shap_writer, sheet_name=s_name, index=False)

            if fusion_shap_writer:
                fusion_shap_writer.close()
                print(f"Fusion Weighted SHAP results saved.")

            if all_dfs:
                final_df = pd.concat(all_dfs, ignore_index=True)
                sort_key = 'G_Mean' if self.cfg.optimization_objective == 'gmean' else 'Weighted_Score'
                if sort_key not in final_df.columns: sort_key = 'ACC'
                final_df = final_df.sort_values(sort_key, ascending=False)
                summary_path = os.path.join(self.cfg.OUTPUT_DIR, f"Final_Fusion_Summary_Top{self.cfg.TOP_M_RESULTS}.xlsx")
                final_df.to_excel(summary_path, index=False)
                print(f"\nAll done! Summary saved to: {summary_path}")

if __name__ == "__main__":
    config = GlobalConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    system = FusionSystem(config)
    system.run()
