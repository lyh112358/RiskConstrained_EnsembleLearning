"""
Base Model Hyperparameter Optimization with Recall-Constrained Learning

A comprehensive machine learning pipeline for training base classifiers with
recall-constrained Bayesian hyperparameter optimization.

Key Components:
- Multi-year transfer learning with year-based cross-validation
- Cost-Sensitive Learning (CSL) for class imbalance handling
- Optuna-based hyperparameter optimization with TPE sampler
- Recall threshold constraint with penalty mechanism
- SHAP-based feature importance and selection

This module implements the base model optimization phase:
1. Trains heterogeneous base models (XGBoost, LightGBM, CatBoost, etc.)
2. Embeds recall threshold into Bayesian optimization objective
3. Applies penalty for violating recall constraints
4. Enables SHAP-based feature selection

Reference: "Kinematics-Inspired Behavioral Dynamics and Risk-Constrained Ensemble Learning
for Student Performance Prediction in Blended Environments"

Author: Yinghe Li (Jilin University)
License: MIT
"""

import os
import warnings
import numpy as np
import pandas as pd
import datetime
import logging
from collections import Counter

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
except ImportError:
    print("Warning: imblearn not installed. SMOTE will be disabled.")
    SMOTE = None

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

use_smote = False
smote_strategy = 'regular'

use_cost_sensitive_learning = True

my_early_stopping_rounds = 300
max_search_combinations = 100
n_random_startup_trials = 30
Seed = 42

num_folds = 10
k_fold_way = 'YearSplit'

optimization_objective = 'gmean'
w_specificity_in_pre = 0.5
w_recall_in_pre = 0.5

min_recall_threshold = 0.8
penalty_factor = 6.0

has_finish_time = "有完成时间"
enable_SHAP_feature_optimization = False
feature_select_num = 180
shap_sample_size_for_not_tree_models = 5

use_gpu = False
test_or_train = 'train'

if use_gpu:
    try:
        from cuml.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from cuml.linear_model import LogisticRegression, Ridge
        from cuml.neighbors import KNeighborsClassifier
        from cuml.svm import SVC
        print("cuML GPU mode enabled")
    except ImportError:
        print("cuML import failed, falling back to CPU")
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
else:
    print("CPU mode")
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
smote_info = f"SMOTE({smote_strategy})" if (use_smote and SMOTE is not None) else "SMOTE_off"
csl_info = "CSL_on" if use_cost_sensitive_learning else "CSL_off"

base_dir = (
    f"./results/base_model_optimization_{timestamp}_{k_fold_way}_{smote_info}_{csl_info}_"
    f"rec{w_recall_in_pre}_spe{w_specificity_in_pre}_{optimization_objective}_{max_search_combinations}trials"
    f"{n_random_startup_trials}random_{my_early_stopping_rounds}earlystop"
)
os.makedirs(base_dir, exist_ok=True)

def find_optimal_threshold(y_true, y_pred_prob):
    thresholds = np.linspace(0.01, 0.99, 200)
    best_threshold, best_score = 0.5, -np.inf

    for t in thresholds:
        y_pred = (y_pred_prob > t).astype(int)
        if optimization_objective == 'acc':
            score = accuracy_score(y_true, y_pred)
        elif optimization_objective == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif optimization_objective == 'pre':
            if len(np.unique(y_pred)) < 2 and len(np.unique(y_true)) > 1: continue
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
            score = w_specificity_in_pre * (tn/(tn+fp+1e-9)) + w_recall_in_pre * (tp/(tp+fn+1e-9))
        elif optimization_objective == 'gmean':
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            rec = tp / (tp + fn + 1e-9)
            spec = tn / (tn + fp + 1e-9)
            score = np.sqrt(rec * spec)
        elif optimization_objective == 'auc':
            return 0.5

        if score > best_score:
            best_score, best_threshold = score, t
    return best_threshold

def evaluate_detailed(y_true, y_pred_prob, fixed_threshold=None, model_name=""):
    opt = fixed_threshold if fixed_threshold is not None else find_optimal_threshold(y_true, y_pred_prob)
    y_pred = (y_pred_prob > opt).astype(int)

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        tn = np.sum((y_true == 0) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    rec = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    fail_miss_rate = fn / (tp + fn + 1e-9)
    g_mean = np.sqrt(rec * spec)

    return {
        "Model": model_name,
        "ACC": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_pred_prob), 4) if len(np.unique(y_true))>1 else 0.5,
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(rec, 4),
        "Specificity": round(spec, 4),
        "G-Mean": round(g_mean, 4),
        "Weighted_Score": round(w_specificity_in_pre * spec + w_recall_in_pre * rec, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Threshold": round(opt, 4),
        "Fail_Miss_Rate": round(fail_miss_rate, 4),
    }

class CalibratedRidgeClassifier:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None
    def fit(self, X, y, sample_weight=None):
        base = RidgeClassifier(**self.params)
        self.model = CalibratedClassifierCV(base, method='sigmoid', cv=3)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
    def predict(self, X): return self.model.predict(X)
    def predict_proba(self, X): return self.model.predict_proba(X)
    def set_params(self, **params): self.params.update(params); return self
    def get_params(self, deep=True): return self.params

def search_top_parameters(model_class, base_config, param_grid, class_type,
                          X_train, y_train, X_test, y_test,
                          groups=None, fold_ids=None,
                          save_dir=None, file_tag="Origin"):

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(n_startup_trials=n_random_startup_trials, seed=Seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def preprocess_fold(X_tr_raw, X_val_raw):
        imputer = SimpleImputer(strategy='mean')
        X_tr_imp = imputer.fit_transform(X_tr_raw)
        X_val_imp = imputer.transform(X_val_raw)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_imp)
        X_val_sc = scaler.transform(X_val_imp)
        return X_tr_sc, X_val_sc

    def objective(trial):
        try:
            cfg = base_config.copy()
            for key, values in param_grid.items():
                if any(isinstance(v, (str, bool)) or v is None for v in values):
                    cfg[key] = trial.suggest_categorical(key, values)
                elif all(isinstance(v, int) for v in values):
                    cfg[key] = trial.suggest_int(key, min(values), max(values))
                elif all(isinstance(v, float) for v in values):
                    cfg[key] = trial.suggest_float(key, min(values), max(values), log=False)
                else:
                    cfg[key] = trial.suggest_categorical(key, values)

            if k_fold_way == 'YearSplit' and fold_ids is not None:
                fold_re = PredefinedSplit(test_fold=fold_ids)
                split_args = (X_train, y_train)
            elif k_fold_way == 'TimeSeriesSplit':
                fold_re = TimeSeriesSplit(n_splits=num_folds)
                split_args = (X_train, y_train)
            else:
                fold_re = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=Seed)
                split_args = (X_train, y_train)

            oof_preds, oof_targets = [], []
            y_train_values = y_train.values if hasattr(y_train, 'values') else y_train

            for train_idx, val_idx in fold_re.split(*split_args):
                X_tr_raw, X_val_raw = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train_values[train_idx], y_train_values[val_idx]
                X_tr, X_val = preprocess_fold(X_tr_raw, X_val_raw)[:2]

                if use_smote and SMOTE is not None:
                    sm = BorderlineSMOTE(random_state=Seed) if smote_strategy=='borderline' else SMOTE(random_state=Seed)
                    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

                n_neg = np.sum(y_tr == 0)
                n_pos = np.sum(y_tr == 1)
                ratio = n_neg / (n_pos + 1e-9)

                if use_cost_sensitive_learning:
                    if class_type in ['RF', 'ET', 'LR', 'DT', 'Ridge', 'HGB', 'SVM', 'MLP', 'ENLR']:
                        if 'class_weight' in model_class().get_params():
                            cfg['class_weight'] = 'balanced'

                    elif class_type in ['XGBoost', 'LightGBM']:
                        cfg['scale_pos_weight'] = ratio

                    elif class_type == 'CatBoost':
                        cfg['auto_class_weights'] = 'Balanced'

                if class_type == 'XGBoost':
                    model = model_class(**cfg, early_stopping_rounds=my_early_stopping_rounds)
                else:
                    model = model_class(**cfg)

                fit_params = {}

                if use_cost_sensitive_learning:
                     sw = compute_sample_weight(class_weight='balanced', y=y_tr)

                     if class_type in ['AdaBoost', 'GBC']:
                         fit_params['sample_weight'] = sw
                     elif class_type == 'CatBoost' and cfg.get('auto_class_weights') is None:
                         fit_params['sample_weight'] = sw

                if class_type == 'XGBoost':
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, **fit_params)
                elif class_type == 'LightGBM':
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                              callbacks=[lgb.early_stopping(stopping_rounds=my_early_stopping_rounds, verbose=False)],
                              **fit_params)
                elif class_type == 'CatBoost':
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, **fit_params)
                else:
                    try: model.fit(X_tr, y_tr, **fit_params)
                    except TypeError: model.fit(X_tr, y_tr)

                y_pred_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
                oof_preds.append(y_pred_prob)
                oof_targets.append(y_val)

            all_oof_preds = np.concatenate(oof_preds)
            all_oof_targets = np.concatenate(oof_targets)

            best_threshold = find_optimal_threshold(all_oof_targets, all_oof_preds)
            trial.set_user_attr("best_threshold", best_threshold)

            y_pred_class = (all_oof_preds > best_threshold).astype(int)

            if len(np.unique(all_oof_targets)) > 1:
                tn, fp, fn, tp = confusion_matrix(all_oof_targets, y_pred_class, labels=[0, 1]).ravel()
                specificity = tn / (tn + fp + 1e-9)
                recall = tp / (tp + fn + 1e-9)
            else:
                specificity, recall = 0, 0

            if optimization_objective == 'acc': score = accuracy_score(all_oof_targets, y_pred_class)
            elif optimization_objective == 'auc': score = roc_auc_score(all_oof_targets, all_oof_preds)
            elif optimization_objective == 'f1': score = f1_score(all_oof_targets, y_pred_class, zero_division=0)
            elif optimization_objective == 'pre': score = w_specificity_in_pre * specificity + w_recall_in_pre * recall
            elif optimization_objective == 'gmean': score = np.sqrt(recall * specificity)

            if recall < min_recall_threshold:
                penalty = (min_recall_threshold - recall) * penalty_factor
                final_score = score - penalty
            else:
                final_score = score

            return final_score

        except Exception:
            return 0.0

    study.optimize(objective, n_trials=max_search_combinations, show_progress_bar=True)

    if test_or_train == 'train':
        top_trials = sorted([t for t in study.trials if t.value is not None],
                           key=lambda x: x.value, reverse=True)[:50]
        results = []

        final_imputer = SimpleImputer(strategy='mean')
        X_train_imp = final_imputer.fit_transform(X_train)
        X_test_imp = final_imputer.transform(X_test)
        final_scaler = StandardScaler()
        X_train_sc = final_scaler.fit_transform(X_train_imp)
        X_test_sc = final_scaler.transform(X_test_imp)

        y_train_final = y_train
        if use_smote and SMOTE is not None:
            sm = BorderlineSMOTE(random_state=Seed) if smote_strategy=='borderline' else SMOTE(random_state=Seed)
            X_train_sc, y_train_final = sm.fit_resample(X_train_sc, y_train)

        n_neg = np.sum(y_train_final == 0)
        n_pos = np.sum(y_train_final == 1)
        final_ratio = n_neg / (n_pos + 1e-9)
        final_sw = compute_sample_weight(class_weight='balanced', y=y_train_final) if use_cost_sensitive_learning else None

        for rank, trial in enumerate(top_trials, 1):
            params = trial.params
            cv_learned_threshold = trial.user_attrs.get("best_threshold", 0.5)

            cfg = base_config.copy()
            cfg.update(params)

            if use_cost_sensitive_learning:
                if class_type in ['RF', 'ET', 'LR', 'DT', 'Ridge', 'HGB', 'SVM', 'MLP', 'ENLR']:
                     if 'class_weight' in model_class().get_params(): cfg['class_weight'] = 'balanced'
                elif class_type in ['XGBoost', 'LightGBM']:
                    cfg['scale_pos_weight'] = final_ratio
                elif class_type == 'CatBoost':
                    cfg['auto_class_weights'] = 'Balanced'

            model = model_class(**cfg)

            fit_params = {}
            if use_cost_sensitive_learning:
                 if class_type in ['AdaBoost', 'GBC']:
                     fit_params['sample_weight'] = final_sw
                 elif class_type == 'CatBoost' and cfg.get('auto_class_weights') is None:
                     fit_params['sample_weight'] = final_sw

            try:
                if class_type == 'CatBoost':
                    model.fit(X_train_sc, y_train_final, verbose=False, **fit_params)
                elif class_type in ['XGBoost', 'LightGBM']:
                    model.fit(X_train_sc, y_train_final, verbose=False, **fit_params) if class_type == 'XGBoost' else model.fit(X_train_sc, y_train_final, **fit_params)
                else:
                    try: model.fit(X_train_sc, y_train_final, **fit_params)
                    except TypeError: model.fit(X_train_sc, y_train_final)

                y_pred = model.predict_proba(X_test_sc)[:, 1]

                metrics = evaluate_detailed(y_test, y_pred, fixed_threshold=cv_learned_threshold,
                                           model_name=f"{model_class.__name__}_Top{rank}")
                metrics.update(params)
                results.append(metrics)
            except Exception as e:
                print(f"Skip Top{rank}: {e}")
                continue

        if results:
            sort_metric_map = {'acc': 'ACC', 'auc': 'AUC', 'f1': 'F1', 'pre': 'Weighted_Score', 'gmean': 'G-Mean'}
            target_col = sort_metric_map.get(optimization_objective, 'ACC')
            df = pd.DataFrame(results).sort_values(by=target_col, ascending=False)
        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if save_dir is None: save_dir = base_dir
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"top50_{model_class.__name__}_{class_type}_{file_tag}_{timestamp}.xlsx")

    if not df.empty:
        df.to_excel(out_path, index=False)
        print(f"Saved ({file_tag}) -> {out_path}")

    return df


class Transfer:
    def __init__(self, source_paths, target_path):
        self.source_paths = source_paths
        self.target_path = target_path
        self.is_preprocessed = False
        self.groups, self.fold_ids, self.col = None, None, None

    def pre_conduct(self):
        if self.is_preprocessed: return
        print(f"Loading training data ({len(self.source_paths)} years)...")

        year_dfs = []
        for path in self.source_paths:
            df = pd.read_csv(path)
            year_dfs.append(df)

        df_target = pd.read_csv(self.target_path)

        print(f"Target year samples: {len(df_target)}")

        if k_fold_way == 'YearSplit':
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=Seed)

            for year_idx, df_year in enumerate(year_dfs):
                df_year['custom_fold_id'] = -1
                for fold_idx, (_, val_idx) in enumerate(skf.split(df_year, df_year['grade_label'])):
                    df_year.iloc[val_idx, df_year.columns.get_loc('custom_fold_id')] = year_idx * 2 + fold_idx

            src = pd.concat(year_dfs, ignore_index=True)
            self.fold_ids = src['custom_fold_id'].values
            print(f"YearSplit mode, total folds: {len(year_dfs) * 2}")
        else:
            src = pd.concat(year_dfs, ignore_index=True)
            print(f"Standard cross-validation ({num_folds} folds)")

        self.groups = src['SID'] if 'SID' in src.columns else None
        exclude_cols = ['SID', 'grade', 'grade_label', 'MOOCgrade', 'custom_fold_id']

        if has_finish_time == "有完成时间":
            self.col = [c for c in src.columns if c not in exclude_cols + ['LE_score_off', 'LS_score_off'] and 'Fixed' not in c and 'LAcc' not in c]
        elif has_finish_time == "无完成时间":
            self.col = [c for c in src.columns if c not in exclude_cols and "完成时间" not in c]
        elif has_finish_time == "只有处理前时间":
            self.col = [c for c in src.columns if c not in exclude_cols and "完成处理后时间" not in c and "提交处理后时间" not in c]
        

        self.col = [c for c in self.col if c in df_target.columns]
        self.source_x, self.source_y = src[self.col].values, src['grade_label']
        self.target_x, self.target_y = df_target[self.col].values, df_target['grade_label']
        self.is_preprocessed = True
        print(f"Features: {len(self.col)}, Train samples: {len(self.source_x)}")
        print(f"Train distribution: {Counter(self.source_y)}, Test distribution: {Counter(self.target_y)}")

    def run_blending_with_proper_separation(self):
        print("=== Training Pipeline ===")
        print("Label: 1=FAIL (positive), 0=PASS (negative)")
        self.pre_conduct()

        self.source_x = pd.DataFrame(self.source_x).replace([np.inf, -np.inf], np.nan).values
        self.target_x = pd.DataFrame(self.target_x).replace([np.inf, -np.inf], np.nan).values
        X_train, y_train = self.source_x, self.source_y
        X_test, y_test = self.target_x, self.target_y

        all_models_config = {
            'XGBoost': {
                'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 3,
                'random_state': 42, 'eval_metric': 'logloss', 'n_jobs': -1
            },
            'LightGBM': {
                'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': -1,
                'random_state': 42, 'verbosity': -1, 'n_jobs': -1
            },
            'DT': {'max_depth': 6, 'random_state': 42},
            'Ridge': {'alpha': 1.0, 'max_iter': 1000, 'random_state': 42},
            'HGB': {'max_iter': 100, 'learning_rate': 0.1, 'random_state': 42},
            'ET': {
                'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 2,
                'random_state': 42, 'n_jobs': -1
            },
        }

        base_models_config = {}
        for name, config in all_models_config.items():
            if use_gpu and name in ['RF', 'ET', 'LR', 'KNN', 'Ridge', 'ENLR']:
                base_models_config[name] = {k: v for k, v in config.items() if k not in ['n_jobs']}
            else:
                base_models_config[name] = config

        all_shap_top10, all_shap_full, results = [], {}, []
        tmp_imputer, tmp_scaler = SimpleImputer(strategy='mean'), StandardScaler()
        X_train_shap = tmp_scaler.fit_transform(tmp_imputer.fit_transform(X_train))
        y_train_shap = y_train

        if use_smote and SMOTE is not None:
             if smote_strategy == 'borderline':
                 try: sm = BorderlineSMOTE(random_state=Seed, kind='borderline-1')
                 except: sm = SMOTE(random_state=Seed)
             else:
                 sm = SMOTE(random_state=Seed)
             X_train_shap, y_train_shap = sm.fit_resample(X_train_shap, y_train)

        for i, (name, cfg) in enumerate(base_models_config.items()):
            if name == 'SVC': continue
            print(f"\n=== [{i+1}/{len(base_models_config)}] {name} ===")

            model_map = {
                'RF': RandomForestClassifier, 'ET': ExtraTreesClassifier, 'XGBoost': xgb.XGBClassifier,
                'LightGBM': lgb.LGBMClassifier, 'CatBoost': CatBoostClassifier, 'MLP': MLPClassifier,
                'LR': LogisticRegression, 'KNN': KNeighborsClassifier, 'HGB': HistGradientBoostingClassifier,
                'ENLR': LogisticRegression, 'GNB': GaussianNB, 'BNB': BernoulliNB,
                'DT': DecisionTreeClassifier, 'LDA': LinearDiscriminantAnalysis,
                'AdaBoost': AdaBoostClassifier, 'GBC': GradientBoostingClassifier
            }

            if name == 'Ridge':
                final_model = CalibratedClassifierCV(RidgeClassifier(**cfg), method='sigmoid', cv=3)
            elif name in model_map:
                final_model = model_map[name](**cfg)
            else:
                continue

            try:
                final_model.fit(X_train_shap, y_train_shap)
                is_tree = name in ['RF', 'ET', 'XGBoost', 'LightGBM', 'DT', 'GBC']
                X_shap = X_train_shap[np.random.choice(len(X_train_shap), min(2000, len(X_train_shap)), replace=False)]
                top10, full_df, _ = self.analyze_model_shap(final_model, name, X_shap, None, is_tree)
                all_shap_top10.append(top10)
                all_shap_full[name] = full_df
                self._save_shap_excel(all_shap_top10, all_shap_full, {})
            except: pass

            top_features = full_df['Feature'].head(feature_select_num).tolist()
            sel_idx = [self.col.index(f) for f in top_features if f in self.col]
            X_train_sel = X_train[:, sel_idx] if sel_idx else X_train
            X_test_sel = X_test[:, sel_idx] if sel_idx else X_test

            param_grids = self._get_param_grids(name)
            if param_grids is None: continue
            param_grid, base = param_grids
            model_cls = CalibratedRidgeClassifier if name == 'Ridge' else (xgb.XGBClassifier if name == 'XGBoost' else type(final_model))

            print(f"  Full feature optimization (Goal: {optimization_objective})...")
            df_origin = search_top_parameters(model_cls, base, param_grid, name, X_train, y_train, X_test, y_test,
                                             groups=self.groups, fold_ids=self.fold_ids, file_tag="Origin")

            if enable_SHAP_feature_optimization and sel_idx and X_train_sel.shape[1] < X_train.shape[1]:
                print(f"  Selected feature ({X_train_sel.shape[1]}) optimization...")
                df_selected = search_top_parameters(model_cls, base, param_grid, name, X_train_sel, y_train, X_test_sel, y_test,
                                                   groups=self.groups, fold_ids=self.fold_ids, file_tag="Selected")
                if not df_selected.empty:
                    best = df_selected.iloc[0].to_dict()
                    best['Model_Name'] = f"{name}_Selected"
                    results.append(best)
            elif not df_origin.empty:
                best = df_origin.iloc[0].to_dict()
                best['Model_Name'] = f"{name}_Origin"
                results.append(best)

        results_df = pd.DataFrame(results)
        results_df.to_excel(os.path.join(base_dir, f"base_models_report_{timestamp}.xlsx"), index=False)
        self._save_shap_excel(all_shap_top10, all_shap_full, {})
        print(f"\nResults saved to: {base_dir}")
        return results_df

    def _get_param_grids(self, name):
        grids = {
            'XGBoost': ({'learning_rate': [0.001, 0.5], 'n_estimators': [100, 5000], 'max_depth': [3, 20],
                        'min_child_weight': [0.1, 100], 'subsample': [0.4, 1.0], 'colsample_bytree': [0.4, 1.0],
                        'gamma': [0, 20], 'reg_lambda': [0.0001, 1000], 'reg_alpha': [0.0001, 1000]},
                       {'tree_method': 'hist', 'device': 'cuda' if use_gpu else 'cpu', 'eval_metric': 'logloss', 'n_jobs': -1}),
            'ET': ({
                'n_estimators': [100, 800],
                'max_depth': [5, 25],
                'min_samples_split': [2, 20],
                'min_samples_leaf': [2, 10],
                'max_features': ['sqrt', 'log2', 0.5, 0.9],
                'bootstrap': [True, False]
            }, {'random_state': 42, 'n_jobs': -1}),
            'LightGBM': ({'learning_rate': [0.001, 0.5], 'n_estimators': [100, 5000], 'num_leaves': [20, 1024],
                        'max_depth': [-1, 50], 'min_child_samples': [5, 300], 'feature_fraction': [0.4, 1.0],
                        'bagging_fraction': [0.4, 1.0], 'lambda_l1': [1e-8, 100], 'lambda_l2': [1e-8, 100]},
                       {'random_state': 42, 'objective': 'binary', 'device': 'gpu' if use_gpu else 'cpu', 'verbosity': -1}),
            'RF': ({
                'n_estimators': [100, 800], 'max_depth': [5, 15],
                'min_samples_split': [2, 20],'min_samples_leaf': [2, 10],  'max_features': ['sqrt', 'log2', 0.5, 0.8],
            }, {'random_state': 42, 'n_jobs': -1}),
            'KNN': ({'n_neighbors': [3, 200], 'weights': ['uniform', 'distance'], 'p': [1, 2]}, {'n_jobs': -1}),
            'HGB': ({'learning_rate': [0.001, 0.5], 'max_iter': [100, 5000], 'max_depth': [1, 50, None],
                    'min_samples_leaf': [5, 300], 'l2_regularization': [1e-8, 100]}, {'random_state': 42}),
            'LR': ({
                'C': [0.001, 100.0],
                'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga'],
            }, {'random_state': 42, 'n_jobs': -1, 'max_iter': 1000}),
            'ENLR': ({
                'C': [0.001, 100.0],'l1_ratio': [0.1, 0.9],
            }, {'random_state': 42, 'penalty': 'elasticnet', 'solver': 'saga',
                'max_iter': 2000}),
            'GNB': ({'var_smoothing': [1e-15, 1e-1]}, {}),
            'BNB': ({'alpha': [0.0001, 1000.0], 'binarize': [0.0, 1.0]}, {}),
            'DT': ({'max_depth': [1, 50, None], 'min_samples_split': [2, 100], 'min_samples_leaf': [1, 50]}, {'random_state': 42}),
            'LDA': ({'solver': ['svd', 'lsqr'], 'tol': [1e-8, 1e-1]}, {}),
            'Ridge': ({'alpha': [0.00001, 10000.0], 'max_iter': [100, 10000]},
                     {'random_state': 42}),
            'AdaBoost': ({'n_estimators': [50, 500], 'learning_rate': [0.01, 1.0]}, {'random_state': 42}),
            'GBC': ({'n_estimators': [100, 500], 'learning_rate': [0.01, 0.5], 'max_depth': [3, 10]}, {'random_state': 42})
        }
        return grids.get(name)

    def _save_shap_excel(self, all_top10_list, all_full_dict, model_weights):
        excel_path = os.path.join(base_dir, f"SHAP_Analysis_{timestamp}.xlsx")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            if all_top10_list:
                pd.concat(all_top10_list, ignore_index=True).to_excel(writer, sheet_name='All_Top10_Summary', index=False)
            for name, df in all_full_dict.items():
                df.to_excel(writer, sheet_name=f'{name[:31]}_Full', index=False)

    def analyze_model_shap(self, model, model_name, X_data, y_none, is_tree_model=True):
        import shap
        explaining_size = len(X_data) if is_tree_model else min(shap_sample_size_for_not_tree_models, len(X_data))
        try:
            if is_tree_model:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_data[:explaining_size])
                if isinstance(shap_values, list): shap_values = shap_values[1]
                if len(shap_values.shape) == 3: shap_values = shap_values[:, :, 1]
            else:
                background = shap.kmeans(X_data, 20) if len(X_data) > 50 else X_data
                runner = lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x)
                explainer = shap.KernelExplainer(runner, background)
                shap_values = explainer.shap_values(X_data[:explaining_size], l1_reg="num_features(10)")

            if isinstance(shap_values, list): shap_values = shap_values[0]
            feature_importance = np.abs(shap_values).mean(axis=0).ravel()
            expected_len = len(self.col)
            if len(feature_importance) > expected_len: feature_importance = feature_importance[:expected_len]
            elif len(feature_importance) < expected_len: feature_importance = np.pad(feature_importance, (0, expected_len - len(feature_importance)))

            total = feature_importance.sum() + 1e-9
            shap_df = pd.DataFrame({
                'Rank': range(1, expected_len + 1), 'Feature': self.col,
                'SHAP_Importance': feature_importance, 'Contribution_Rate(%)': feature_importance / total * 100
            }).sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)
            shap_df['Rank'] = range(1, len(shap_df) + 1)

            top50 = shap_df.head(50).copy()
            top50['Model'] = model_name
            return top50, shap_df, shap_values
        except Exception as e:
            print(f"{model_name} SHAP failed: {e}")
            empty = pd.DataFrame({'Rank': range(1, 51), 'Feature': ['Error']*50, 'SHAP_Importance': [0.0]*50,
                                 'Contribution_Rate(%)': [0.0]*50, 'Model': [model_name]*50})
            return empty, empty, None


if __name__ == "__main__":
    print(f"GPU mode: {use_gpu}")
    print(f"Optimization target: {optimization_objective} (Recall penalty threshold: {min_recall_threshold})")

    source_years_paths = [
        'path/to/your/Merged_2020_全特征_updated.csv',
        'path/to/your/Merged_2021_全特征_updated.csv',
        'path/to/your/Merged_2022_全特征_updated.csv',
        'path/to/your/Merged_2023_全特征_updated.csv',
        'path/to/your/Merged_2024_全特征_updated.csv',
    ]

    target_year_path = 'path/to/your/Merged_2025_全特征_updated.csv'

    tr = Transfer(source_years_paths, target_year_path)

    base_model_results = tr.run_blending_with_proper_separation()
    print(f"\nResults saved to: {base_dir}")
