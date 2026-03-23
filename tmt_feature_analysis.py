"""
SHAP Analysis Tool for Model Interpretability in Academic Early Warning

A comprehensive SHAP (SHapley Additive exPlanations) analysis toolkit for
explaining machine learning model predictions in recall-constrained ensemble learning.

Key Features:
- Support for multiple model types (tree-based: XGBoost, LightGBM, CatBoost; linear models)
- Feature importance calculation based on mean absolute SHAP values
- Top features selection for TMT-based constructs (LE, LS, LAcc, RPI)
- Batch processing of multiple models with weighted importance propagation

This module provides interpretability analysis to:
1. Identify most influential features for at-risk student prediction
2. Validate TMT-based feature engineering effectiveness
3. Explain ensemble model decisions at both base and fusion levels

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
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from tqdm import tqdm

use_smote = True
smote_strategy = 'regular'
use_cost_sensitive_learning = True
use_gpu = False
Seed = 42

has_finish_time = "有完成时间"

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
except ImportError:
    SMOTE = None
    use_smote = False

def clean_param(param_name, param_value):
    if pd.isna(param_value): return None
    str_val = str(param_value).lower()
    if str_val in ['none', 'nan']: return None

    int_params = [
        'n_estimators', 'max_depth', 'num_leaves', 'min_child_weight',
        'min_samples_split', 'min_samples_leaf', 'n_neighbors', 'max_iter',
        'min_child_samples', 'random_state', 'iterations', 'depth'
    ]
    bool_params = ['fit_intercept', 'early_stopping', 'bootstrap', 'fit_prior', 'probability', 'verbose', 'replace', 'shuffle']
    float_params = ['learning_rate', 'subsample', 'colsample_bytree', 'gamma', 'reg_alpha', 'reg_lambda', 'alpha', 'l1_ratio', 'tol', 'C']

    try:
        if param_name in int_params: return int(float(param_value))
        if param_name in bool_params:
            if isinstance(param_value, bool): return param_value
            return True if str_val in ['true', 'yes'] else (False if str_val in ['false', 'no'] else bool(float(param_value)))
        if param_name in float_params: return float(param_value)
        return param_value
    except:
        return param_value

class FinalSHAPAnalyzer:
    def __init__(self, source_paths, target_path, param_file_path):
        self.source_paths = source_paths
        self.target_path = target_path
        self.param_file_path = param_file_path
        self.timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        self.save_dir = os.path.dirname(param_file_path) if param_file_path else "./results"

        self.col = []
        self.X_train = None
        self.y_train = None

    def load_data(self):
        print(">>> 1. Loading and Preprocessing Data...")

        df_train = pd.concat([pd.read_csv(f) for f in self.source_paths], ignore_index=True)
        df_target = pd.read_csv(self.target_path)

        exclude_cols = ['SID', 'grade', 'grade_label', 'MOOCgrade', 'custom_fold_id']

        if has_finish_time == "有完成时间":
            self.col = [c for c in df_train.columns if c not in exclude_cols + ['LE_score_off', 'LS_score_off'] and 'Fixed' not in c]
        elif has_finish_time == "无完成时间":
            self.col = [c for c in df_train.columns if c not in exclude_cols and "完成时间" not in c]
        elif has_finish_time == "只有处理前时间":
            self.col = [c for c in df_train.columns if c not in exclude_cols and "完成处理后时间" not in c and "提交处理后时间" not in c]
        elif has_finish_time == "预测线下":
            self.col = [c for c in df_train.columns if c not in exclude_cols + ['offline_grade_label', '班级', "实验成绩_fg", "期末成绩_fg", "grade_to01", "总成绩_fg"] and "线下考试相关" not in c]

        self.col = [c for c in self.col if c in df_target.columns]

        X_raw = df_train[self.col].replace([np.inf, -np.inf], np.nan).values
        y_raw = df_train['grade_label'].values

        print(f"    Feature Count: {len(self.col)}")
        print(f"    Sample Count: {len(X_raw)}")

        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        X_imp = imputer.fit_transform(X_raw)
        self.X_train = scaler.fit_transform(X_imp)
        self.y_train = y_raw

        if use_smote and SMOTE is not None:
            print(f"    Applying SMOTE ({smote_strategy})...")
            if smote_strategy == 'borderline':
                sm = BorderlineSMOTE(random_state=Seed, kind='borderline-1')
            else:
                sm = SMOTE(random_state=Seed)
            self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)

    def get_best_params(self, model_keyword, row_index=0):
        try:
            df = pd.read_excel(self.param_file_path)
            col_name = 'Model_Name' if 'Model_Name' in df.columns else 'Model'

            target_rows = df[df[col_name].astype(str).str.contains(model_keyword, case=False)]

            if target_rows.empty:
                print(f"    [Warning] No parameters found for keyword '{model_keyword}'.")
                return None

            if row_index >= len(target_rows):
                print(f"    [Warning] Index {row_index} out of range, only {len(target_rows)} rows available.")
                return None

            selected_row = target_rows.iloc[row_index]
            print(f"    >>> Loaded params from Excel Row {row_index + 2} (Index {row_index}): {selected_row[col_name]}")

            exclude_cols = [
                'Model', 'Model_Name', 'ACC', 'AUC', 'F1', 'Precision', 'Recall',
                'Specificity', 'G-Mean', 'Weighted_Score', 'TP', 'FP', 'TN', 'FN',
                'Threshold', 'Fail_Miss_Rate', 'Balanced_ACC', 'FPR', 'FNR'
            ]

            params = {}
            for k in df.columns:
                if k not in exclude_cols and 'Contrib' not in k:
                    val = clean_param(k, selected_row[k])
                    if val is not None:
                        params[k] = val
            return params
        except Exception as e:
            print(f"    [Error] Failed to read Excel: {e}")
            return None

    def build_model(self, model_type, params):
        fit_params = {}
        model_params = params.copy()

        if use_cost_sensitive_learning:
            n_neg = np.sum(self.y_train == 0)
            n_pos = np.sum(self.y_train == 1)
            ratio = n_neg / (n_pos + 1e-9)
            sw = compute_sample_weight(class_weight='balanced', y=self.y_train)

            if model_type in ['RF', 'ET', 'LR', 'DT', 'Ridge', 'HGB', 'SVM', 'MLP', 'ENLR']:
                model_params['class_weight'] = 'balanced'
            elif model_type in ['XGBoost', 'LightGBM']:
                model_params['scale_pos_weight'] = ratio
            elif model_type == 'CatBoost':
                model_params['auto_class_weights'] = 'Balanced'

            if model_type in ['AdaBoost', 'GBC']:
                fit_params['sample_weight'] = sw
            elif model_type == 'CatBoost' and model_params.get('auto_class_weights') is None:
                fit_params['sample_weight'] = sw

        try:
            if model_type == 'XGBoost':
                model = xgb.XGBClassifier(**model_params)
            elif model_type == 'LightGBM':
                model_params['verbosity'] = -1
                model = lgb.LGBMClassifier(**model_params)
            elif model_type == 'CatBoost':
                model = CatBoostClassifier(**model_params)
            elif model_type == 'RF':
                model = RandomForestClassifier(**model_params)
            elif model_type == 'ET':
                model = ExtraTreesClassifier(**model_params)
            elif model_type == 'Ridge':
                base = RidgeClassifier(**model_params)
                model = CalibratedClassifierCV(base, method='sigmoid', cv=3)
            elif model_type == 'LR':
                model = LogisticRegression(**model_params)
            elif model_type == 'HGB':
                model = HistGradientBoostingClassifier(**model_params)
            elif model_type == 'BNB':
                if 'random_state' in model_params: del model_params['random_state']
                model = BernoulliNB(**model_params)
            elif model_type == 'DT':
                model = DecisionTreeClassifier(**model_params)
            elif model_type == 'AdaBoost':
                model = AdaBoostClassifier(**model_params)
            elif model_type == 'GBC':
                model = GradientBoostingClassifier(**model_params)
            else:
                return None, {}
            return model, fit_params
        except Exception as e:
            print(f"    [Error] Init model failed: {e}")
            return None, {}

    def run(self, target_models_dict, target_row_index=0):
        self.load_data()
        all_summaries = []
        full_details = {}

        for model_type, keyword in target_models_dict.items():
            print(f"\n>>> Processing Model: {model_type} (Keyword: {keyword})")

            params = self.get_best_params(keyword, row_index=target_row_index)
            if not params: continue

            model, fit_params = self.build_model(model_type, params)
            if not model: continue

            print("    Fitting on full source data...")
            try:
                if model_type == 'CatBoost':
                    model.fit(self.X_train, self.y_train, verbose=False, **fit_params)
                elif model_type == 'Ridge' and 'sample_weight' in fit_params:
                    model.fit(self.X_train, self.y_train, sample_weight=fit_params['sample_weight'])
                else:
                    try: model.fit(self.X_train, self.y_train, **fit_params)
                    except: model.fit(self.X_train, self.y_train)
            except Exception as e:
                print(f"    [Error] Fit failed: {e}")
                continue

            print("    Calculating SHAP values...")
            try:
                shap_values = None
                is_tree = model_type in ['RF', 'ET', 'XGBoost', 'LightGBM', 'DT', 'GBC', 'CatBoost', 'HGB']

                bg_size = 2000 if len(self.X_train) > 2000 else len(self.X_train)
                idx = np.random.choice(len(self.X_train), bg_size, replace=False)
                X_bg = self.X_train[idx]

                if is_tree:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_bg)

                    if isinstance(shap_values, list): shap_values = shap_values[1]
                    if len(np.array(shap_values).shape) == 3: shap_values = shap_values[:, :, 1]

                else:
                    kmeans_bg = shap.kmeans(self.X_train, 20)

                    if model_type == 'Ridge':
                        runner = lambda x: model.predict_proba(x)[:, 1]
                    else:
                        runner = lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x)

                    explainer = shap.KernelExplainer(runner, kmeans_bg)
                    shap_values = explainer.shap_values(X_bg, l1_reg="num_features(10)")

                    if isinstance(shap_values, list): shap_values = shap_values[0]

                feature_importance = np.abs(shap_values).mean(axis=0).ravel()

                if len(feature_importance) != len(self.col):
                    print(f"    [Warning] Dim mismatch: SHAP {len(feature_importance)} vs Col {len(self.col)}")
                    min_l = min(len(feature_importance), len(self.col))
                    feature_importance = feature_importance[:min_l]
                    self.col = self.col[:min_l]

                total_imp = feature_importance.sum() + 1e-9

                df_shap = pd.DataFrame({
                    'Feature': self.col,
                    'SHAP_Importance': feature_importance,
                    'Contribution_Rate(%)': (feature_importance / total_imp) * 100
                })

                df_shap = df_shap.sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)
                df_shap.insert(0, 'Rank', range(1, len(df_shap) + 1))

                full_details[model_type] = df_shap

                top50 = df_shap.head(50).copy()
                top50['Model'] = model_type
                all_summaries.append(top50)

            except Exception as e:
                print(f"    [Error] SHAP calculation error: {e}")
                import traceback
                traceback.print_exc()

        if all_summaries:
            out_file = os.path.join(self.save_dir, f"SHAP_Analysis_{self.timestamp}.xlsx")
            with pd.ExcelWriter(out_file, engine='xlsxwriter') as writer:
                pd.concat(all_summaries, ignore_index=True).to_excel(writer, sheet_name='Summary_Top50', index=False)
                for m_name, df in full_details.items():
                    df.to_excel(writer, sheet_name=f'{m_name}_Full', index=False)

            print(f"\nSHAP Analysis Finished! File saved at:\n{out_file}")
        else:
            print("\nNo SHAP results generated.")


if __name__ == "__main__":

    PARAM_FILE = "path/to/your/model_params.xlsx"

    SRC_FILES = [
        'path/to/your/Merged_2020_全特征_updated.csv',
        'path/to/your/Merged_2021_全特征_updated.csv',
        'path/to/your/Merged_2022_全特征_updated.csv',
        'path/to/your/Merged_2023_全特征_updated.csv',
        'path/to/your/Merged_2024_全特征_updated.csv'
    ]
    TGT_FILE = 'path/to/your/Merged_2025_全特征_updated.csv'

    MODELS_TO_ANALYZE = {
        'HGB': 'HistGradientBoostingClassifier'
    }
    MY_TARGET_ROW = 1

    analyzer = FinalSHAPAnalyzer(SRC_FILES, TGT_FILE, PARAM_FILE)
    analyzer.run(MODELS_TO_ANALYZE, target_row_index=MY_TARGET_ROW)
