# Mancy Chen 20/02/2025
# Classification with XGBoost
import os,joblib, csv
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
from sklearn.exceptions import ConvergenceWarning
# Ignore the specific FutureWarnings from XGBoost about glibc versions.
warnings.filterwarnings(
    "ignore",
    message="Your system has an old version of glibc",
    category=FutureWarning
)
# Ignore all ConvergenceWarnings from IterativeImputer.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# import sys
# sys.path.append('/.../miniconda3/lib/python3.10/site-packages')
import re
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
# from sklearn.pipeline import Pipeline # mute for SMOTE pipeline
from imblearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut, StratifiedKFold, cross_validate, GridSearchCV, \
    cross_val_score, cross_val_predict, RepeatedStratifiedKFold, RandomizedSearchCV, StratifiedShuffleSplit, learning_curve,\
    StratifiedGroupKFold
from skopt.space import Integer, Real, Categorical
from skopt import BayesSearchCV, Optimizer
from statsmodels import robust  # if needed for median_abs_deviation; otherwise use scipy.stats
from scipy.stats import median_abs_deviation
np.int = int  # Patch to allow legacy code to work
import matplotlib.pyplot as plt
import shap
from scipy.stats import ttest_ind, pearsonr, binomtest, spearmanr, kendalltau, pointbiserialr, median_abs_deviation
from neuroHarmonize import harmonizationLearn, harmonizationApply
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from imblearn.over_sampling import SMOTE         # or other sampler
import random
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

################################################################################################################
# Load the data
# --- Utility Functions ---
def remove_substrings(df, remove_in_cells=False):
    """
    Remove '+AF8' and '+AC0' from:
      - row labels (index)
      - column labels (columns)
      - optionally from the cell contents themselves
    """
    # Remove from row labels (index)
    df.index = df.index.astype(str)
    df.index = df.index.str.replace('+AF8', '', regex=False).str.replace('+AC0', '', regex=False)

    # Remove from index name if present
    if df.index.name:
        df.index.name = str(df.index.name).replace('+AF8', '').replace('+AC0', '')

    # Remove from column labels (columns)
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.replace('+AF8', '', regex=False).str.replace('+AC0', '', regex=False)

    # Remove from column name if present
    if df.columns.name:
        df.columns.name = str(df.columns.name).replace('+AF8', '').replace('+AC0', '')

    # Optionally remove from cell contents
    if remove_in_cells:
        # This will only affect string cells; numeric columns remain unchanged
        df.replace({r'\+AF8': '', r'\+AC0': ''}, regex=True, inplace=True)

    return df

def get_config(i: int,
               j: int,
               base_x_dir="/.../EMBARC/data/06_BART_regression/Input/x",
               base_y_dir="/.../EMBARC/data/06_BART_regression/Input/y/Imputation/classification/Remission",
               base_out_dir="/.../EMBARC/data/06_BART_regression/Output/Classification_plot/Remission/Tier2b",
               x_subpath="Site_normalization/Tier2b",
               y_prefix="deltaHAMD",
               out_suffix="save_feature_and_model",
               tier_label="Tier2b"):
    """
    i mapping:
      1 -> ses-1 SER
      2 -> ses-1 PLA
      3 -> ses-2 SER
      4 -> ses-2 PLA

    j controls the output folder prefix:
      e.g., j=1  -> "01_..."
            j=11 -> "11_..."
    """
    mapping = {
        1: ("ses-1", "SER"),
        2: ("ses-1", "PLA"),
        3: ("ses-2", "SER"),
        4: ("ses-2", "PLA"),
    }
    if i not in mapping:
        raise ValueError("i must be 1, 2, 3, or 4.")

    ses_number, medication = mapping[i]

    # X filename like: Tier1_selected_ses-1_SER.csv
    x_filename = f"{tier_label}_selected_{ses_number}_{medication}.csv"
    x_path = os.path.join(base_x_dir, x_subpath, x_filename)

    # Y filename like: deltaHAMD_ses_1_SER.csv  (ses_1 not ses-1)
    ses_for_y = ses_number.replace("ses-", "ses_")
    y_filename = f"{y_prefix}_{ses_for_y}_{medication}.csv"
    y_path = os.path.join(base_y_dir, y_filename)

    # Output folder like: 11_ses-1_SER_save_feature_and_model
    out_folder = f"{int(j):02d}_{ses_number}_{medication}_{out_suffix}"
    output_path = os.path.join(base_out_dir, out_folder)
    os.makedirs(output_path, exist_ok=True)

    return x_path, y_path, output_path, medication, ses_number

i = 4
j = 8
x_path, y_path, output_path, medication, ses_number = get_config(i, j)

print("x_path:", x_path)
print("y_path:", y_path)
print("output_path:", output_path)
print("medication:", medication)
print("ses_number:", ses_number)


feature_number = 413 # model A: 14; model B: 1302; model C: 413; ablation: 0
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
selected_features = 10
max_display = 20

# Process X
# X_df = pd.read_csv(x_path, index_col=0, header = None) # No labels for columns
X_df = pd.read_csv(x_path, index_col=0) # have labels for columns
X_df = remove_substrings(X_df, remove_in_cells=True)
X = X_df.T # Becareful!!!
feature_names = X_df.index.tolist() # Row is feature name
# X = X_df
# feature_names = X_df.columns.tolist() # Columns is feature name
feature_names = [fname.replace("original-", "") for fname in feature_names]

print("X shape (subjects, features):", X.shape)  # Expected (93, 1302)
# print(repr(X.columns))

# Process y

y_df = pd.read_csv(y_path, header=None, encoding='utf-8-sig')
y_df = remove_substrings(y_df, remove_in_cells=True)
y = y_df.to_numpy(dtype=np.float64)
print("y shape:", y.shape)  # Expected (93,)



n_samples = X.shape[0]
n_features = X.shape[1]
print('After filtering by medication: n_samples:', n_samples, '; n_features:', n_features, '\n')

#######################################################################################################################
class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that:
      1. Imputes BMI by median
      2. Imputes is_employed by most frequent
      3. IterativeImputer for MASQ w0/w1 columns (aa/ad/gd)
      4. IterativeImputer for w1_score_17 (using w0, w2, w3, w4, w6)
      5. Creates r1_score_17 = w1_score_17 / w0_score_17
      6. IterativeImputer for shaps_total_continuous-w0/w1
    """

    def __init__(self, session='ses-1', n_neighbors=5, random_state=0):
        self.session = session
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        # --------------------------------------
        # 1) BMI (median) / 2) is_employed (mode)
        # --------------------------------------
        self.bmi_median_ = None
        self.employment_mode_ = None

        # --------------------------------------
        # 3) IterativeImputer for MASQ w1
        # --------------------------------------
        self.masq_w1_cols_ = [
            "masq2-score-aa-w0", "masq2-score-aa-w1",
            "masq2-score-ad-w0", "masq2-score-ad-w1",
            "masq2-score-gd-w0", "masq2-score-gd-w1"
        ]
        self.masq_w1_iter_ = IterativeImputer(random_state=self.random_state)

        # --------------------------------------
        # 4) IterativeImputer for w1_score_17
        # --------------------------------------
        self.w1_cols_ = [
            "w0-score-17", "w1-score-17", "w2-score-17", "w3-score-17",
            "w4-score-17", "w6-score-17"
        ]
        self.w1_iter_ = IterativeImputer(random_state=self.random_state)

        # --------------------------------------
        # 6) IterativeImputer for shaps
        # --------------------------------------
        self.shaps_w1_cols_ = [
            "shaps-total-continuous-w0",
            "shaps-total-continuous-w1"
        ]
        self.shaps_w1_iter_ = IterativeImputer(random_state=self.random_state)


    def _convert_numeric(self, df):
        """
        Helper method to convert relevant columns to numeric.
        """
        numeric_cols = set()
        # BMI (if exists)
        if "BMI" in df.columns:
            numeric_cols.add("BMI")
        # MASQ columns
        numeric_cols.update([c for c in self.masq_w1_cols_ if c in df.columns])
        # w1 score columns (including w0_score_17 and w1_score_17 for division later)
        numeric_cols.update([c for c in self.w1_cols_ if c in df.columns])
        # shaps columns
        numeric_cols.update([c for c in self.shaps_w1_cols_ if c in df.columns])
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def fit(self, X, y=None):
        X_fit = X.copy()

        # Convert all relevant columns to numeric
        X_fit = self._convert_numeric(X_fit)

        # --------------------------------------
        # 1) Fit median for BMI
        # --------------------------------------
        if "BMI" in X_fit.columns:
            self.bmi_median_ = X_fit["BMI"].median()

        # --------------------------------------
        # 2) Fit mode for is_employed
        # --------------------------------------
        if "is-employed" in X_fit.columns:
            self.employment_mode_ = X_fit["is-employed"].mode()[0]

        # --------------------------------------
        # 3) Fit IterativeImputer for MASQ w1
        # --------------------------------------
        masq_exist = [c for c in self.masq_w1_cols_ if c in X_fit.columns]
        if len(masq_exist) == len(self.masq_w1_cols_):
            self.masq_w1_iter_.fit(X_fit[masq_exist])

        # --------------------------------------
        # 4) Fit IterativeImputer for w1_score_17
        # --------------------------------------
        w1_exist = [c for c in self.w1_cols_ if c in X_fit.columns]
        if len(w1_exist) == len(self.w1_cols_):
            self.w1_iter_.fit(X_fit[w1_exist])

        # --------------------------------------
        # 6) Fit IterativeImputer for shaps
        # --------------------------------------
        shaps_exist = [c for c in self.shaps_w1_cols_ if c in X_fit.columns]
        if len(shaps_exist) == len(self.shaps_w1_cols_):
            self.shaps_w1_iter_.fit(X_fit[shaps_exist])
        return self

    def transform(self, X, y=None):
        X_out = X.copy()

        # Convert all relevant columns to numeric
        X_out = self._convert_numeric(X_out)

        # --------------------------------------
        # 1) Impute BMI by median
        # --------------------------------------
        if self.bmi_median_ is not None and "BMI" in X_out.columns:
            X_out["BMI"] = X_out["BMI"].fillna(self.bmi_median_)

        # --------------------------------------
        # 2) Impute is_employed by most frequent
        # --------------------------------------
        if self.employment_mode_ is not None and "is-employed" in X_out.columns:
            X_out["is-employed"] = X_out["is-employed"].fillna(self.employment_mode_)

        # --------------------------------------
        # 3) IterativeImputer for MASQ w1
        # --------------------------------------
        masq_exist = [c for c in self.masq_w1_cols_ if c in X_out.columns]
        if len(masq_exist) == len(self.masq_w1_cols_):
            X_out[masq_exist] = self.masq_w1_iter_.transform(X_out[masq_exist])

        # --------------------------------------
        # 4) IterativeImputer for w1_score_17
        # --------------------------------------
        w1_exist = [c for c in self.w1_cols_ if c in X_out.columns]
        if len(w1_exist) == len(self.w1_cols_):
            X_out[w1_exist] = self.w1_iter_.transform(X_out[w1_exist])

        # --------------------------------------
        # 5) Create r1_score_17 = w1_score_17 / w0_score_17
        # --------------------------------------
        if "w1-score-17" in X_out.columns and "w0-score-17" in X_out.columns:
            X_out["r1-score-17"] = X_out.apply(
                lambda row: row["w1-score-17"] / row["w0-score-17"]
                if row["w0-score-17"] != 0 else np.nan,
                axis=1
            )

        # --------------------------------------
        # 6) IterativeImputer for shaps
        # --------------------------------------
        shaps_exist = [c for c in self.shaps_w1_cols_ if c in X_out.columns]
        if len(shaps_exist) == len(self.shaps_w1_cols_):
            X_out[shaps_exist] = self.shaps_w1_iter_.transform(X_out[shaps_exist])

        return X_out


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, session='ses-1', cols_to_drop_1=None, cols_to_drop_2=None):
        self.session = session
        self.cols_to_drop_1 = cols_to_drop_1 or []
        self.cols_to_drop_2 = cols_to_drop_2 or []
        # This will hold the final columns we decide to drop
        self.cols_to_drop_ = None

    def fit(self, X, y=None):
        if self.session == 'ses-1':
                self.cols_to_drop_ = self.cols_to_drop_1
        else:  # session == 'ses-2'
                self.cols_to_drop_ = self.cols_to_drop_2
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        return X_copy.drop(columns=self.cols_to_drop_, errors='ignore')

class NeuroHarmonizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_number: int,
        covariate_cols=("Site", "Age", "Age-squared", "Gender"),
        skip_combat: bool = False,
        enforce_train_column_order: bool = True,
    ):
        self.feature_number = int(feature_number)
        self.covariate_cols = tuple(covariate_cols)
        self.skip_combat = bool(skip_combat)
        self.enforce_train_column_order = bool(enforce_train_column_order)

        self.model_ = None
        self.feature_cols_ = None
        self.covariate_cols_resolved_ = None

    @staticmethod
    def _resolve_columns_case_insensitive(df_cols, wanted):
        """Return a list of actual column names in df that match `wanted` case-insensitively."""
        lower_map = {c.lower(): c for c in df_cols}
        resolved = profiler = []
        resolved = []
        for w in wanted:
            key = w.lower()
            if key in lower_map:
                resolved.append(lower_map[key])
            else:
                resolved.append(None)
        return resolved

    def _build_covariates(self, X: pd.DataFrame) -> pd.DataFrame:
        # resolve covariate column names robustly (case-insensitive)
        resolved = self._resolve_columns_case_insensitive(X.columns, self.covariate_cols)
        missing = [w for w, r in zip(self.covariate_cols, resolved) if r is None]
        if missing:
            raise ValueError(
                f"NeuroHarmonizeTransformer: missing covariate columns {missing}. "
                f"Available columns include: {list(X.columns)[:20]} ..."
            )
        cov = X.loc[:, resolved].copy()

        # standardize covariate names expected by neuroHarmonize
        rename_map = {}
        for orig, want in zip(resolved, self.covariate_cols):
            wl = want.lower()
            if wl == "site":
                rename_map[orig] = "SITE"
            elif wl == "age":
                rename_map[orig] = "age"
            elif wl in ("age-squared", "age_squared", "agesquared"):
                rename_map[orig] = "age_squared"
            elif wl == "gender":
                rename_map[orig] = "gender"
            else:
                rename_map[orig] = want  # fallback
        cov.rename(columns=rename_map, inplace=True)

        # enforce types
        cov["SITE"] = cov["SITE"].astype(str)
        for c in ["age", "age_squared", "gender"]:
            if c in cov.columns:
                cov[c] = pd.to_numeric(cov[c], errors="coerce")

        return cov

    def fit(self, X, y=None):
        if not hasattr(X, "columns"):
            raise TypeError("NeuroHarmonizeTransformer expects a pandas DataFrame as X (so it can find covariates by name).")

        # Optionally lock feature column order from training set
        cov_resolved = self._resolve_columns_case_insensitive(X.columns, self.covariate_cols)
        self.covariate_cols_resolved_ = [c for c in cov_resolved if c is not None]
        self.feature_cols_ = [c for c in X.columns if c not in self.covariate_cols_resolved_]

        # Skip ComBat entirely if requested or feature_number <= 0
        if self.skip_combat or self.feature_number <= 0:
            self.model_ = None
            return self

        covariate_df = self._build_covariates(X)

        # Convert features (excluding covariates) to numeric array
        feat_df = X.loc[:, self.feature_cols_].copy()
        feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
        data = feat_df.to_numpy(dtype=np.float64)

        k = min(self.feature_number, data.shape[1])
        if k <= 0:
            self.model_ = None
            return self

        radiomics = data[:, :k]
        model_out = harmonizationLearn(radiomics, covars=covariate_df)
        self.model_ = model_out[0] if isinstance(model_out, tuple) else model_out
        return self

    def transform(self, X):
        if not hasattr(X, "columns"):
            raise TypeError("NeuroHarmonizeTransformer expects a pandas DataFrame as X.")

        # Reorder columns to match training (prevents train/test column-order drift)
        if self.enforce_train_column_order and self.feature_cols_ is not None and self.covariate_cols_resolved_ is not None:
            needed = self.feature_cols_ + self.covariate_cols_resolved_
            missing = [c for c in needed if c not in X.columns]
            if missing:
                raise ValueError(
                    f"NeuroHarmonizeTransformer: input is missing columns seen at fit(): {missing[:10]} ..."
                )
            X_use = X.loc[:, needed].copy()
        else:
            # still exclude covariates by name
            cov_resolved = self._resolve_columns_case_insensitive(X.columns, self.covariate_cols)
            cov_resolved = [c for c in cov_resolved if c is not None]
            feat_cols = [c for c in X.columns if c not in cov_resolved]
            X_use = X.loc[:, feat_cols + cov_resolved].copy()
            self.feature_cols_ = feat_cols
            self.covariate_cols_resolved_ = cov_resolved

        # Build covariates by NAME
        covariate_df = self._build_covariates(X_use)

        # Features (excluding covariates)
        feat_df = X_use.loc[:, self.feature_cols_].copy()
        feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
        data = feat_df.to_numpy(dtype=np.float64)

        # If skipping, just return features (radiomics+clinical) unchanged (covars excluded)
        k = min(self.feature_number, data.shape[1])
        if self.skip_combat or self.model_ is None or k <= 0:
            return data

        radiomics = data[:, :k]
        clinical = data[:, k:]

        harmonized = harmonizationApply(radiomics, covars=covariate_df, model=self.model_)
        if isinstance(harmonized, tuple):
            harmonized = harmonized[0]

        return np.hstack([harmonized, clinical]).astype(np.float64)

class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_features_to_select= selected_features, corr_th=0.8):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.corr_th = corr_th

    def fit(self, X, y=None):
        # IMPORTANT: specify importance_getter for XGBoost
        selector = RFE(
            estimator=self.estimator,
            n_features_to_select=self.n_features_to_select,
            step=1,
            importance_getter='feature_importances_'
        )
        self.selected_features_indices_ = self.selectNonIntercorrelated(X, y, self.corr_th, selector)
        return self

    def transform(self, X):
        return X[:, self.selected_features_indices_]

    def selectNonIntercorrelated(self, X, y, corr_th, selector):
        non_nan_indices = np.all(~np.isnan(X), axis=0)
        X_non_nan = X[:, non_nan_indices]
        mad_values = median_abs_deviation(X_non_nan, axis=0, scale='normal')
        non_zero_var_indices = mad_values > 0.001
        X_non_zero_var = X_non_nan[:, non_zero_var_indices]
        if X_non_zero_var.shape[1] == 0:
            raise ValueError("All features have zero MAD")

        corr_matrix = np.corrcoef(X_non_zero_var, rowvar=False)
        np.fill_diagonal(corr_matrix, 0)
        mean_absolute_corr = np.abs(corr_matrix).mean(axis=0)

        intercorrelated_features_set = set()
        high_corrs = np.argwhere(np.abs(corr_matrix) > corr_th)
        for i, j in high_corrs:
            if mean_absolute_corr[i] > mean_absolute_corr[j]:
                intercorrelated_features_set.add(i)
            else:
                intercorrelated_features_set.add(j)

        non_intercorrelated_indices = list(
            set(range(X_non_zero_var.shape[1])) - intercorrelated_features_set
        )
        X_train_non_intercorrelated = X_non_zero_var[:, non_intercorrelated_indices]

        if X_train_non_intercorrelated.shape[1] <= self.n_features_to_select:
            selected_indices = np.array(non_intercorrelated_indices)
        else:
            selector = selector.fit(X_train_non_intercorrelated, y)
            support = selector.get_support()
            selected_indices = np.array(non_intercorrelated_indices)[support]

        # Map back to original feature indices
        final_mask = np.zeros(non_nan_indices.shape[0], dtype=bool)
        non_nan_zero_var = np.where(non_nan_indices)[0][non_zero_var_indices]
        final_mask[non_nan_zero_var[selected_indices]] = True
        return np.where(final_mask)[0]

# ------------------ Build the Pipeline with XGBoost ------------------

ComBat_transformer = NeuroHarmonizeTransformer(
    feature_number=feature_number,
    covariate_cols=("Site", "Age", "Age-squared", "Gender"),
    skip_combat=False,  # <-- set to True to skip ComBat and just pass through features (but still require covariates for consistent column handling)
    enforce_train_column_order=True,
)



# Base XGBoost estimator for RFE
selector_estimator = XGBRegressor(random_state=random_seed, eval_metric='rmse')

# ses-1
cols_to_drop_1 = ['subject-id','Medication','w1-score-17','r1-score-17',\
    'shaps-total-continuous-w1','masq2-score-aa-w1','masq2-score-ad-w1','masq2-score-gd-w1',\
    'w2-score-17','w3-score-17','w4-score-17','w6-score-17','w8-score-17','w9-score-17',\
    'w10-score-17','w12-score-17','w16-score-17']

# ses-2
cols_to_drop_2 = ['subject-id','Medication', 'w2-score-17','w3-score-17','w4-score-17','w6-score-17',\
                  'w8-score-17','w9-score-17', 'w10-score-17','w12-score-17','w16-score-17']

# ------------------ Hyperparameter Search Space ------------------
# Example hyperparams to tune in both the selector's XGBoost and the final XGBoost
pbounds = {
    # XGBoost in RFE
    'selector__estimator__n_estimators': Integer(50, 300),
    'selector__estimator__max_depth': Integer(2, 8),
    'selector__estimator__learning_rate': Real(1e-3, 1e-1, prior='log-uniform'),
    'selector__estimator__subsample': Real(0.5, 1.0),
    'selector__estimator__colsample_bytree': Real(0.5, 1.0),
    'selector__estimator__gamma': Real(0, 10),
    'selector__estimator__reg_alpha': Real(1e-3, 1e1, prior='log-uniform'),
    'selector__estimator__reg_lambda': Real(1e-3, 1e1, prior='log-uniform'),

    # Final XGBoost
    'xgb__n_estimators': Integer(50, 300),
    'xgb__max_depth': Integer(2, 8),
    'xgb__learning_rate': Real(1e-3, 1e-1, prior='log-uniform'),
    'xgb__subsample': Real(0.5, 1.0),
    'xgb__colsample_bytree': Real(0.5, 1.0),
    'xgb__gamma': Real(0, 10),
    'xgb__reg_alpha': Real(1e-3, 1e1, prior='log-uniform'),
    'xgb__reg_lambda': Real(1e-3, 1e1, prior='log-uniform'),
}

# ------------------ BayesSearchCV ------------------
cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)

# ------------------ Outer CV Loop ------------------
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)


# Choose which feature name list to use in the CSV:
NAME_LIST = feature_names

long_csv_path = os.path.join(output_path, "selected_features_shap_long.csv")

# Write header once
csv_f = open(long_csv_path, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_f)
writer.writerow([
    "fold",
    "split",
    "sample_in_fold",
    "global_row_id",          # optional: stable id within this CSV
    "feature_rank_in_fold",   # j in selected-feature space
    "feature_index",          # index in full feature space
    "feature_name",
    "shap_value",
    "x_value",
])

global_row_id = 0

# Collect per-fold metadata to save as npy at end
fold_ids = []
selected_indices_per_fold = []
selected_feature_names_per_fold = []
train_indices_per_fold = []   # optional but very useful
test_indices_per_fold = []    # optional but very useful

y_pred_list = []
y_true_list = []
y_proba_list = []
fold_r2 = []
fold_rmse = []
outer_fold_counter = 0
total_outer_folds = cv_outer.get_n_splits()
best_models = []
all_train_shap_values = []
all_test_shap_values = []
all_best_X_train = []
all_best_X_test = []
elapsed_times = []
selected_indices_list = []
best_models = []
fold_auc         = []
fold_ap          = []
fold_f1          = []
fold_bacc        = []
fold_precision   = []
fold_recall      = []
fold_specificity = []


print(
    f"[{datetime.now().strftime('%H:%M:%S')}] Progress - Outer Folds: 0.00% | Start to process the first iteration of outer folds")

for train_index, test_index in cv_outer.split(X, y):
    start_time = time.time()
    outer_fold_counter += 1

    # X_train, X_test = X[train_index], X[test_index] # X as numpy array
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] # X as data frame
    y_train, y_test = y[train_index], y[test_index]
    # 1) Compute your imbalance ratio on THIS training fold
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"[Fold {outer_fold_counter}] neg={neg}, pos={pos}, scale_pos_weight={scale_pos_weight:.2f}")

    # 2) Rebuild your pipeline so that the classifier gets the correct weight
    pipeline = Pipeline([
        # Step 1: Impute the entire clinical_df (294 subjects)
        ("impute_clinical", CustomImputer(session=ses_number)),
        # Step 2: Drop columns (depends on session,  and your predefined lists)
        ("drop_cols", DropColumns(
            session=ses_number,
            cols_to_drop_1=cols_to_drop_1,
            cols_to_drop_2=cols_to_drop_2
        )),
        # Step 3: NeuroCombat with covariates
        ("Combat", ComBat_transformer),  # Assuming you've already configured covariates inside
        # Step 4: Scale
        ("scaler", RobustScaler()),
        # Step 5: inject a sampler to synthetically balance the minority class ---
        ("smote", SMOTE(random_state=random_seed, sampling_strategy="auto")),
        # Step 6: Feature selection
        ("selector", CustomFeatureSelector(estimator=selector_estimator)),
        # Step 7: XGB
        # --- NEW: classification model with imbalance handling ---
        ("xgb", XGBClassifier(
            # tree_method='gpu_hist',  # <-- switch on GPU
            # predictor='gpu_predictor',  # <-- use the CUDA predictor
            random_state=random_seed,
            use_label_encoder=False,  # suppress deprecation warning
            objective='binary:logistic',
            eval_metric='auc',  # or 'logloss', 'aucpr'
            scale_pos_weight=scale_pos_weight,  # balance via built‑in weighting
        ))
    ])

    # 3) Plug THAT pipeline into your BayesSearchCV
    optimizer = BayesSearchCV(
        estimator=pipeline,  # must end in XGBClassifier
        search_spaces=pbounds,  # tuned hyperparams for classifier
        n_iter=50,
        scoring='roc_auc',  # or 'average_precision', 'f1'
        n_jobs=-1,
        cv=cv_inner,
        random_state=random_seed
    )

    # 4) Fit the Bayesian search on the training fold
    optimizer.fit(X_train, y_train)
    print(f" Best inner-fold ROC AUC: {optimizer.best_score_:.4f}")
    print(f" Best params: {optimizer.best_params_}")

    # Evaluate on the test fold
    best_pipeline = optimizer.best_estimator_
    best_models.append(best_pipeline)
    fold_id = outer_fold_counter  # or i, however you index folds
    joblib.dump(best_pipeline, os.path.join(output_path, f"best_pipeline_fold{fold_id}.joblib"))
    y_pred = best_pipeline.predict(X_test)
    y_pred_list.extend(y_pred)
    y_true_list.extend(y_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_proba_list.extend(y_proba)

    # Compute metrics
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    fold_auc.append(auc)
    fold_ap.append(ap)
    fold_f1.append(f1)
    fold_bacc.append(bacc)
    fold_precision.append(precision)
    fold_recall.append(recall)
    fold_specificity.append(specificity)

    print(
        f"Fold {outer_fold_counter} — "
        f"AUC: {auc:.3f}, AP: {ap:.3f}, F1: {f1:.3f}, "
        f"BalAcc: {bacc:.3f}, Precision: {precision:.3f}, "
        f"Recall: {recall:.3f}, Specificity: {specificity:.3f}"
    )

    # --------------- SHAP for XGBoost ---------------
    # We'll illustrate using TreeExplainer on the final XGBRegressor
    transform_pipeline = Pipeline(best_pipeline.steps[:-1])  # all steps except the final xgb
    selected_indices = best_pipeline.named_steps['selector'].selected_features_indices_
    selected_indices_list.append(selected_indices)
    final_xgb = best_pipeline.named_steps['xgb']
    explainer = shap.TreeExplainer(final_xgb)
    best_X_train = transform_pipeline.transform(X_train)
    best_X_test = transform_pipeline.transform(X_test)
    # Compute the SHAP values
    shap_values_train = explainer.shap_values(best_X_train)
    shap_values_test = explainer.shap_values(best_X_test)

    # --- unwrap SHAP if it comes as a list (binary classification sometimes does this) ---
    if isinstance(shap_values_train, list):
        shap_values_train = shap_values_train[0]
    if isinstance(shap_values_test, list):
        shap_values_test = shap_values_test[0]

    # Selected indices (full feature space)
    selected_indices = best_pipeline.named_steps['selector'].selected_features_indices_
    selected_indices = np.array(selected_indices, dtype=int)

    # Selected names (same length as selected_indices)
    selected_names = [NAME_LIST[i] for i in selected_indices]

    # Save per-fold info
    fold_ids.append(outer_fold_counter)
    selected_indices_per_fold.append(selected_indices)
    selected_feature_names_per_fold.append(np.array(selected_names, dtype=object))
    train_indices_per_fold.append(np.array(train_index, dtype=int))  # mapping back to original X rows
    test_indices_per_fold.append(np.array(test_index, dtype=int))

    # Sanity checks
    assert shap_values_train.shape == best_X_train.shape, "Train SHAP shape != Train X shape"
    assert shap_values_test.shape == best_X_test.shape, "Test SHAP shape != Test X shape"
    assert shap_values_train.shape[1] == len(selected_indices), "Train SHAP cols != #selected features"
    assert shap_values_test.shape[1] == len(selected_indices), "Test SHAP cols != #selected features"

    # --- stream TRAIN rows ---
    n_train = shap_values_train.shape[0]
    for s in range(n_train):
        for j, (full_idx, fname) in enumerate(zip(selected_indices, selected_names)):
            writer.writerow([
                outer_fold_counter,
                "train",
                s,
                global_row_id,
                j,
                int(full_idx),
                fname,
                float(shap_values_train[s, j]),
                float(best_X_train[s, j]),
            ])
            global_row_id += 1

    # --- stream TEST rows ---
    n_test = shap_values_test.shape[0]
    for s in range(n_test):
        for j, (full_idx, fname) in enumerate(zip(selected_indices, selected_names)):
            writer.writerow([
                outer_fold_counter,
                "test",
                s,
                global_row_id,
                j,
                int(full_idx),
                fname,
                float(shap_values_test[s, j]),
                float(best_X_test[s, j]),
            ])
            global_row_id += 1

    # (optional) flush to ensure progress is written even if job crashes later
    csv_f.flush()

    # Store the SHAP values and the best X
    all_train_shap_values.append(shap_values_train)
    all_test_shap_values.append(shap_values_test)
    all_best_X_train.append(best_X_train)
    all_best_X_test.append(best_X_test)

    elapsed_time = time.time() - start_time
    elapsed_times.append(elapsed_time)
    progress = outer_fold_counter / total_outer_folds * 100
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Outer fold {outer_fold_counter}/{total_outer_folds} complete. Progress: {progress:.2f}%. "
          f"Time for this fold: {elapsed_time:.2f} seconds.")


# # Checking for each step's shape
# pipeline1 = pipeline.fit(X,y)
# X1 = pipeline.steps[0][1].fit_transform(X_train)
# print("X shape in step 1: ", X1.shape)
# X2 = pipeline.steps[1][1].fit_transform(X1)
# print("X shape in step 2: ", X2.shape)
# print(repr(X2.columns))
# X3 = pipeline.steps[2][1].fit_transform(X2)
# print("X shape in step 3: ", X3.shape)
# X4 = pipeline.steps[3][1].fit_transform(X3)
# print("X shape in step 4: ", X4.shape)
# X5 = pipeline.steps[4][1].fit_transform(X4)
# print("X shape in step 5: ", X5.shape)
# # X6 = pipeline.steps[5][1].transform(X5)
# # print("X shape in step 6: ", X6.shape)

# Save the selected features
csv_f.close()
print("Saved long-format CSV:", long_csv_path)

np.save(os.path.join(output_path, "fold_ids.npy"),
        np.array(fold_ids, dtype=int))

np.save(os.path.join(output_path, "selected_indices_per_fold.npy"),
        np.array(selected_indices_per_fold, dtype=object),
        allow_pickle=True)

np.save(os.path.join(output_path, "selected_feature_names_per_fold.npy"),
        np.array(selected_feature_names_per_fold, dtype=object),
        allow_pickle=True)

# optional but recommended
np.save(os.path.join(output_path, "train_indices_per_fold.npy"),
        np.array(train_indices_per_fold, dtype=object),
        allow_pickle=True)

np.save(os.path.join(output_path, "test_indices_per_fold.npy"),
        np.array(test_indices_per_fold, dtype=object),
        allow_pickle=True)

print("Saved per-fold selected indices/names and index mappings in:", output_path)


# Flatten into 1D arrays
y_true_arr  = np.array(y_true_list).ravel().astype(int)
y_pred_arr  = np.array(y_pred_list).ravel().astype(int)
y_proba_arr = np.array(y_proba_list)

assert y_true_arr.shape == y_pred_arr.shape, "Mismatch in true vs pred lengths"
n = len(y_true_arr)

# -----------------  Binomial test on accuracy -----------------
k = int((y_true_arr == y_pred_arr).sum())
res     = binomtest(k, n, p= 0.5, alternative="greater")

binom_p = res.pvalue
prop_hat = res.proportion_estimate

print(f"Accuracy = {k}/{n} = {k/n:.3f}")
print(f"Estimated p̂ = {prop_hat:.3f}, Binomial p‑value = {binom_p:.4f}")

# -----------------  Compute pooled summary metrics  -----------------
overall_auc   = roc_auc_score(y_true_arr, y_proba_arr)
overall_ap    = average_precision_score(y_true_arr, y_proba_arr)
overall_f1    = f1_score(y_true_arr, y_pred_arr)
overall_bacc  = balanced_accuracy_score(y_true_arr, y_pred_arr)
overall_ppv   = precision_score(y_true_arr, y_pred_arr)
overall_rec   = recall_score(y_true_arr, y_pred_arr)
tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr).ravel()
overall_spec  = tn / (tn + fp)

# NNT = 1 / (recall + specificity – 1)
youden_j     = overall_rec + overall_spec - 1.0
overall_nnt  = (1.0 / youden_j) if youden_j > 0 else np.inf

# — per‑fold averages for reporting
avg_auc      = np.mean(fold_auc)
avg_ap       = np.mean(fold_ap)
avg_f1       = np.mean(fold_f1)
avg_bacc     = np.mean(fold_bacc)
avg_ppv      = np.mean(fold_precision)   # PPV is same as precision
avg_rec      = np.mean(fold_recall)
avg_spec     = np.mean(fold_specificity)

# -----------------  Permutation tests -----------------
n_permutations = 1000
rng = np.random.RandomState(random_seed)

perm_aucs  = []
perm_baccs = []
perm_ppvs  = []
perm_nnts  = []

for _ in range(n_permutations):
    y_perm = rng.permutation(y_true_arr)
    perm_aucs.append(roc_auc_score(y_perm, y_proba_arr))
    perm_baccs.append(balanced_accuracy_score(y_perm, y_pred_arr))
    perm_ppvs.append(precision_score(y_perm, y_pred_arr, zero_division=0))
    r_p = recall_score(y_perm, y_pred_arr, zero_division=0)
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_perm, y_pred_arr).ravel()
    spec_p = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else 0
    j_p = r_p + spec_p - 1.0
    perm_nnts.append((1.0 / j_p) if j_p > 0 else np.inf)

perm_p_auc  = np.mean(np.array(perm_aucs)  >= overall_auc)
perm_p_bacc = np.mean(np.array(perm_baccs) >= overall_bacc)
perm_p_ppv  = np.mean(np.array(perm_ppvs)  >= overall_ppv)
perm_p_nnt  = np.mean(np.array(perm_nnts)  <= overall_nnt)

# -----------------  Bootstrap 95% CI for ROC AUC -----------------
n_bootstraps = 1000
boot_aucs = []
for _ in range(n_bootstraps):
    idx = rng.randint(0, n, n)
    boot_aucs.append(roc_auc_score(y_true_arr[idx], y_proba_arr[idx]))
ci_lower, ci_upper = np.percentile(boot_aucs, [2.5, 97.5])

print(f"\nBootstrapped ROC AUC 95% CI: {ci_lower:.3f} – {ci_upper:.3f}")
print("\nPermutation p‑values:")
print(f"  ROC AUC p‑value:  {perm_p_auc:.4f}")
print(f"  bACC p‑value:     {perm_p_bacc:.4f}")
print(f"  PPV p‑value:      {perm_p_ppv:.4f}")
print(f"  NNT p‑value:      {perm_p_nnt:.4f}")

# -----------------  Save everything to CSV -----------------
metrics_df = pd.DataFrame({
    "Metric": [
        # per-fold averages
        "Avg ROC AUC", "Avg Avg Precision", "Avg F1-score", "Avg Bal Accuracy",
        "Avg PPV", "Avg Recall", "Avg Specificity",
        # pooled
        "Pooled ROC AUC", "Pooled Avg Precision", "Pooled F1-score",
        "Pooled Bal Accuracy", "Pooled PPV", "Pooled Recall",
        "Pooled Specificity", "Pooled NNT",
        # statistical tests
        "Accuracy", "Accuracy p-value (binomial)",
        "ROC AUC 95% CI", "ROC AUC p-value",
        "bACC p-value", "PPV p-value", "NNT p-value"
    ],
    "Value": [
        # per-fold
        f"{avg_auc:.4f}", f"{avg_ap:.4f}", f"{avg_f1:.4f}", f"{avg_bacc:.4f}",
        f"{avg_ppv:.4f}", f"{avg_rec:.4f}", f"{avg_spec:.4f}",
        # pooled
        f"{overall_auc:.4f}", f"{overall_ap:.4f}", f"{overall_f1:.4f}",
        f"{overall_bacc:.4f}", f"{overall_ppv:.4f}", f"{overall_rec:.4f}",
        f"{overall_spec:.4f}", f"{overall_nnt:.2f}",
        # tests
        f"{k}/{n}={k/n:.4f}", f"{binom_p:.4f}",
        f"{ci_lower:.4f}–{ci_upper:.4f}", f"{perm_p_auc:.4f}",
        f"{perm_p_bacc:.4f}", f"{perm_p_ppv:.4f}", f"{perm_p_nnt:.4f}"
    ]
})

csv_file = os.path.join(output_path, "classification_metrics.csv")
metrics_df.to_csv(csv_file, index=False)
print(f"\nMetrics saved to: {csv_file}")

# Save each list separately as a .npy file
np.save(os.path.join(output_path, "y_pred_list.npy"), y_pred_list, allow_pickle=True)
np.save(os.path.join(output_path, "y_true_list.npy"), y_true_list, allow_pickle=True)
np.save(os.path.join(output_path, "y_proba_list.npy"), y_proba_list, allow_pickle=True)
np.save(os.path.join(output_path, "all_train_shap_values.npy"), np.array(all_train_shap_values, dtype=object), allow_pickle=True)
np.save(os.path.join(output_path, "all_test_shap_values.npy"), np.array(all_test_shap_values, dtype=object), allow_pickle=True)
np.save(os.path.join(output_path, "all_best_X_train.npy"), np.array(all_best_X_train, dtype=object), allow_pickle=True)
np.save(os.path.join(output_path, "all_best_X_test.npy"), np.array(all_best_X_test, dtype=object), allow_pickle=True)
print("All lists saved as separate .npy files in:", output_path)

# SHAP value plot
unchanged_features = {
    "bmi",
    "masq2_score_gd",
    "shaps_total_continuous",
    "w0_score_17",
    "w1_score_17",
    "w2_score_17",
    "w3_score_17",
    "w4_score_17",
    "w6_score_17",
    "interview_age",
    "is_male",
    "is_employed",
    "is_chronic",
    "Site",
    "age",
    "age_squared",
    "gender"
}

# Restructure of the feature names
def rename_feature(feature_name):
    # If the feature is in the unchanged set, return as-is
    if feature_name in unchanged_features:
        return feature_name
    # Split on underscores
    tokens = feature_name.split("_")
    # Remove any "original" token
    tokens = [t for t in tokens if t != "original"]
    # Join back with underscores
    short_name = "_".join(tokens)
    return short_name
new_feature_names = [rename_feature(fname) for fname in feature_names]

##
# Map the indices of features to the correct subset for each model (fold)
selected_indices_in_each_model = [
    model.named_steps['selector'].selected_features_indices_
    for model in best_models
]
# We'll store the expanded arrays for each fold in lists.
expanded_train_shap_list = []
expanded_train_X_list    = []
expanded_test_shap_list  = []
expanded_test_X_list     = []

n_samples = len(best_models)          # number of folds/models
n_features = len(feature_names)       # total number of features (e.g., 31)

# Loop over each fold:
for i in range(n_samples):
    # Get number of training and test samples for the current fold
    n_train_i = all_train_shap_values[i].shape[0]  # may differ per fold
    n_test_i  = all_test_shap_values[i].shape[0]

    # Create fold-specific arrays (for the full feature space)
    fold_train_shap = np.zeros((n_train_i, n_features))
    fold_train_X    = np.zeros((n_train_i, n_features))
    fold_test_shap  = np.zeros((n_test_i,  n_features))
    fold_test_X     = np.zeros((n_test_i,  n_features))

    # Get selected feature indices for this fold
    selected_indices = selected_indices_in_each_model[i]
    num_selected_features = len(selected_indices)

    # Populate the fold-specific arrays:
    for j in range(num_selected_features):
        idx = selected_indices[j]  # index in the full feature space

        # Map training SHAP and X values from the selected subset into the full array
        fold_train_shap[:, idx] = all_train_shap_values[i][:, j]
        fold_train_X[:, idx]    = all_best_X_train[i][:, j]

        # Map test SHAP and X values
        fold_test_shap[:, idx]  = all_test_shap_values[i][:, j]
        fold_test_X[:, idx]     = all_best_X_test[i][:, j]

    # Append the fold-specific arrays to our lists
    expanded_train_shap_list.append(fold_train_shap)
    expanded_train_X_list.append(fold_train_X)
    expanded_test_shap_list.append(fold_test_shap)
    expanded_test_X_list.append(fold_test_X)

# Now, concatenate the fold-specific arrays along the sample axis.
reshaped_train_shap_array = np.concatenate(expanded_train_shap_list, axis=0)
reshaped_train_X_array    = np.concatenate(expanded_train_X_list, axis=0)
reshaped_test_shap_array  = np.concatenate(expanded_test_shap_list, axis=0)
reshaped_test_X_array     = np.concatenate(expanded_test_X_list, axis=0)

# Save the new arrays if needed:
np.save(os.path.join(output_path, 'reshaped_train_shap_array.npy'), reshaped_train_shap_array)
np.save(os.path.join(output_path, 'reshaped_train_X_array.npy'), reshaped_train_X_array)
np.save(os.path.join(output_path, 'reshaped_test_shap_array.npy'), reshaped_test_shap_array)
np.save(os.path.join(output_path, 'reshaped_test_X_array.npy'), reshaped_test_X_array)

# Reload the arrays
# reshaped_train_shap_array = np.load(os.path.join(output_path, 'reshaped_train_shap_array.npy'))
# reshaped_train_X_array    = np.load(os.path.join(output_path, 'reshaped_train_X_array.npy'))
# reshaped_test_shap_array  = np.load(os.path.join(output_path, 'reshaped_test_shap_array.npy'))
# reshaped_test_X_array     = np.load(os.path.join(output_path, 'reshaped_test_X_array.npy'))

# Plot the SHAP summary plot for the training set:
plt.figure(figsize=(15, 7))
shap.summary_plot(
    reshaped_train_shap_array,
    reshaped_train_X_array,
    feature_names=new_feature_names,
    max_display=max_display,
    plot_size=(13, 8),
    show=False
)
plt.title('SHAP Value of impact on model output (Training Set)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'shap_summary_training1.png'))
plt.show()

# Plot the SHAP summary plot for the test set:
plt.figure(figsize=(15, 7))
shap.summary_plot(
    reshaped_test_shap_array,
    reshaped_test_X_array,
    feature_names=new_feature_names,
    max_display=max_display,
    plot_size=(13, 8),
    show=False
)
plt.title('SHAP Value of impact on model output (Test Set)', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(output_path, 'shap_summary_test1.png'))
plt.show()

# Convert to NumPy arrays if they are Python lists
y_true_arr = np.array(y_true_list)
y_pred_arr = np.array(y_pred_list)
# Calculate Pearson correlation
corr, p_value = pearsonr(np.ravel(y_true_arr), np.ravel(y_pred_arr))
# Create a high-resolution figure
plt.figure(figsize=(6, 6), dpi=300)
# Scatter plot: Actual vs. Predicted
plt.scatter(y_true_arr, y_pred_arr, alpha=0.6, edgecolors='k')
# Plot best-fit regression line
p = np.polyfit(np.ravel(y_true_arr), np.ravel(y_pred_arr), 1)
y_line = np.polyval(p, np.ravel(y_true_arr))
plt.plot(np.ravel(y_true_arr), y_line, 'b-', linewidth=1.5, label='Best-fit Line')
# Set equal aspect and add padding to axes
min_val = min(y_true_arr.min(), y_pred_arr.min())
max_val = max(y_true_arr.max(), y_pred_arr.max())
buffer = 0.05 * (max_val - min_val)
plt.xlim(min_val - buffer, max_val + buffer)
plt.ylim(min_val - buffer, max_val + buffer)
plt.gca().set_aspect('equal', adjustable='box')
# Optionally, add a y = x reference line
# min_val = min(y_true_arr.min(), y_pred_arr.min())
# max_val = max(y_true_arr.max(), y_pred_arr.max())
# plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1, label='y = x')
# Annotate Pearson r and p-value
plt.gca().text(0.025, 0.975, f"r: {corr:.2f}\np: {p_value:.3f}",
               transform=plt.gca().transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
# Labels and layout
plt.xlabel("Actual (y_true)")
plt.ylabel("Predicted (y_pred)")
plt.title("Regression Outcome Plot: y_true vs. y_pred")
plt.legend()
plt.tight_layout()
# Save the figure
plot_file = os.path.join(output_path, "regression_scatter_plot.png")
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.show()
print(f"Scatter plot saved to: {plot_file}")

