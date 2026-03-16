# Mancy Chen 20/02/2025
# Regression with BART
import os, joblib, csv
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
import sys
sys.path.append('/scratch/mchen/miniconda3/lib/python3.10/site-packages')
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
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut, StratifiedKFold, cross_validate, GridSearchCV, \
    cross_val_score, cross_val_predict, RepeatedStratifiedKFold, RandomizedSearchCV, StratifiedShuffleSplit, learning_curve
from skopt.space import Integer, Real, Categorical
from skopt import BayesSearchCV, Optimizer
from bartpy.sklearnmodel import SklearnModel
from statsmodels import robust  # if needed for median_abs_deviation; otherwise use scipy.stats
from scipy.stats import median_abs_deviation
np.int = int  # Patch to allow legacy code to work
import matplotlib.pyplot as plt
import shap
from scipy.stats import ttest_ind, pearsonr, binomtest, spearmanr, kendalltau, pointbiserialr, median_abs_deviation
from neurocombat_sklearn import CombatModel
from neuroHarmonize import harmonizationLearn, harmonizationApply
import xgboost as xgb
from xgboost import XGBRegressor
import random
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
               base_x_dir="/data/projects/EMBARC/data/06_BART_regression/Input/x",
               base_y_dir="/data/projects/EMBARC/data/06_BART_regression/Input/y/Imputation",
               base_out_dir="/data/projects/EMBARC/data/06_BART_regression/Output/Regression_plot/Tier_2b",
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
    out_folder = f"{int(j):02d}_{tier_label}_{ses_number}_{medication}_{out_suffix}"
    output_path = os.path.join(base_out_dir, out_folder)
    os.makedirs(output_path, exist_ok=True)

    return x_path, y_path, output_path, medication, ses_number

i = 4
j = 34
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
feature_names = X_df.index.tolist()
feature_names = [fname.replace("original-", "") for fname in feature_names]
X = X_df.T
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
    def __init__(self, feature_number, skip_combat=False):
        """
        Parameters
        ----------
        feature_number : int
            Number of radiomics features (starting from column 0).
        skip_combat : bool
            If True, bypass ComBat and just return raw features (excluding covariates),
            preserving output shape.
        """
        self.feature_number = feature_number
        self.skip_combat = skip_combat
        self.model_ = None

    def _build_covariates(self, X):
        covariate_df = X.iloc[:, -4:].copy()
        covariate_df.columns = ['SITE', 'age', 'age_squared', 'gender']
        covariate_df['SITE'] = covariate_df['SITE'].astype(str)
        covariate_df['age'] = pd.to_numeric(covariate_df['age'], errors='coerce')
        covariate_df['age_squared'] = pd.to_numeric(covariate_df['age_squared'], errors='coerce')
        covariate_df['gender'] = pd.to_numeric(covariate_df['gender'], errors='coerce')
        return covariate_df

    def fit(self, X, y=None):
        # If skipping ComBat, we intentionally don't learn any harmonization model.
        if self.skip_combat or self.feature_number <= 0:
            self.model_ = None
            return self

        covariate_df = self._build_covariates(X)

        data_array = X.iloc[:, :-4].to_numpy(dtype=np.float64)  # excludes covariates
        radiomics = data_array[:, :self.feature_number]

        model_out = harmonizationLearn(radiomics, covars=covariate_df)
        self.model_ = model_out[0] if isinstance(model_out, tuple) else model_out
        return self

    def transform(self, X):
        # Always exclude covariates in the output
        data_array = X.iloc[:, :-4].to_numpy(dtype=np.float64)

        # If skipping ComBat, return raw (radiomics + clinical) unchanged.
        if self.skip_combat or self.feature_number <= 0 or self.model_ is None:
            return data_array

        # Otherwise, harmonize radiomics only, keep clinical as-is, and preserve shape.
        covariate_df = self._build_covariates(X)

        radiomics = data_array[:, :self.feature_number]
        clinical = data_array[:, self.feature_number:]

        harmonized_radiomics = harmonizationApply(radiomics, covars=covariate_df, model=self.model_)
        return np.hstack([harmonized_radiomics, clinical])


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
    skip_combat=False
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


pipeline = Pipeline([
    # Step 1: Impute the entire clinical_df (294 subjects)
    ("impute_clinical", CustomImputer(session=ses_number)),
    # Step 2: Drop columns (depends on session, and your predefined lists)
    ("drop_cols", DropColumns(
        session=ses_number,
        cols_to_drop_1=cols_to_drop_1,
        cols_to_drop_2=cols_to_drop_2
    )),
    # Step 3: NeuroCombat with covariates
    ("Combat", ComBat_transformer),  # Assuming you've already configured covariates inside
    # Step 4: Scale
    ("scaler", RobustScaler()),
    # Step 5: Feature selection
    ("selector", CustomFeatureSelector(estimator=selector_estimator)),
    # Step 6: XGB
    ("xgb", XGBRegressor(
        random_state=random_seed,
        eval_metric='rmse'
    ))
])

# # Checking for each step's shape
# pipeline1 = pipeline.fit(X,y)
# X1 = pipeline.steps[0][1].transform(X)
# print("X shape in step 1 imputation: ", X1.shape)
# X2 = pipeline.steps[1][1].transform(X1)
# print("X shape in step 2 drop columns: ", X2.shape)
# print(repr(X2.columns))
# X3 = pipeline.steps[2][1].transform(X2)
# print("X shape in step 3 ComBat: ", X3.shape)
# X4 = pipeline.steps[3][1].transform(X3)
# print("X shape in step 4 Scaler: ", X4.shape)
# X5 = pipeline.steps[4][1].transform(X4)
# print("X shape in step 5 feature selection: ", X5.shape)
# # X6 = pipeline.steps[5][1].transform(X5)
# # print("X shape in step 6: ", X6.shape)


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
cv_inner = KFold(n_splits=10, shuffle=True, random_state=random_seed)
optimizer = BayesSearchCV(
    estimator=pipeline,
    search_spaces=pbounds,
    n_iter=50,  # Increase for a more thorough search
    scoring='neg_mean_squared_error',
    n_jobs= -1, # -1: use all cores, or specify a number like 5 to leave some free for system responsiveness
    cv=cv_inner,
    random_state=random_seed
)

# ------------------ Outer CV Loop ------------------
cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_seed)



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


print(
    f"[{datetime.now().strftime('%H:%M:%S')}] Progress - Outer Folds: 0.00% | Start to process the first iteration of outer folds")

for train_index, test_index in cv_outer.split(X, y):
    start_time = time.time()
    outer_fold_counter += 1

    # X_train, X_test = X[train_index], X[test_index] # X as numpy array
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] # X as data frame
    y_train, y_test = y[train_index], y[test_index]

    # Fit the Bayesian search on the training fold
    optimizer.fit(X_train, y_train)
    print(f"Best inner-fold score: {optimizer.best_score_:.4f}")
    print(f"Best params: {optimizer.best_params_}")

    # Evaluate on the test fold
    best_pipeline = optimizer.best_estimator_
    best_models.append(best_pipeline)
    fold_id = outer_fold_counter  # or i, however you index folds
    joblib.dump(best_pipeline, os.path.join(output_path, f"best_pipeline_fold{fold_id}.joblib"))
    y_pred = best_pipeline.predict(X_test)
    y_pred_list.extend(y_pred)
    y_true_list.extend(y_test)

    # Compute metrics
    r2_fold = r2_score(y_test, y_pred)
    rmse_fold = np.sqrt(mean_squared_error(y_test, y_pred))
    fold_r2.append(r2_fold)
    fold_rmse.append(rmse_fold)
    print(f"Fold {outer_fold_counter} - R²: {r2_fold:.4f}, RMSE: {rmse_fold:.4f}")

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

# save the selected features as csv
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



# ------------------ Final Metrics ------------------
avg_r2 = np.mean(fold_r2)
avg_rmse = np.mean(fold_rmse)
overall_r2 = r2_score(y_true_list, y_pred_list)
overall_rmse = np.sqrt(mean_squared_error(y_true_list, y_pred_list))
print("\nOverall Performance across outer folds:")
print(f"  Average R²: {avg_r2:.4f}")
print(f"  Average RMSE: {avg_rmse:.4f}")
print(f"  Overall R² (pooled): {overall_r2:.4f}")
print(f"  Overall RMSE (pooled): {overall_rmse:.4f}")

# Convert prediction lists to NumPy arrays
y_true_arr = np.array(y_true_list)
y_pred_arr = np.array(y_pred_list)
n_samples = len(y_true_arr)

# ------------------ Bootstrapping for 95% CI on Overall R² ------------------
n_bootstraps = 1000
boot_r2_scores = []
seed = 42
rng = np.random.default_rng(seed)
# ------------------ Bootstrapping for 95% CI on Overall R² ------------------
n_bootstraps = 1000
boot_r2_scores = []

for _ in range(n_bootstraps):
    indices = rng.choice(n_samples, size=n_samples, replace=True)
    boot_r2_scores.append(r2_score(y_true_arr[indices], y_pred_arr[indices]))

ci_lower = np.percentile(boot_r2_scores, 2.5)
ci_upper = np.percentile(boot_r2_scores, 97.5)
print(f"Bootstrapped R² 95% CI: {ci_lower:.4f} – {ci_upper:.4f}")

# ------------------ Permutation Testing for Overall R² p-value ------------------
n_permutations = 1000
permuted_r2_scores = []

for _ in range(n_permutations):
    y_true_perm = rng.permutation(y_true_arr)
    permuted_r2_scores.append(r2_score(y_true_perm, y_pred_arr))

# add +1 correction to avoid p=0
permuted_r2_scores = np.asarray(permuted_r2_scores)
p_value = (np.sum(permuted_r2_scores >= overall_r2) + 1) / (n_permutations + 1)
print(f"Permutation test p-value: {p_value:.4f}")

# Pearson correlation coefficient
# Ensure both arrays are numpy arrays
y_true_array = np.array(y_true_list)
y_pred_array = np.array(y_pred_list)
# Flatten the arrays to 1D
y_true_flat = np.squeeze(y_true_array)  # or use y_true_array.ravel()
y_pred_flat = np.squeeze(y_pred_array)  # if needed
# Compute Pearson's correlation coefficient
pearson_corr, p_value_Pearson = pearsonr(y_true_flat, y_pred_flat)
print("Pearson correlation coefficient:", pearson_corr)
print("p-value:", p_value_Pearson)

# ------------------ Create Metrics DataFrame ------------------
# Assuming you already have:
# avg_r2, avg_rmse, overall_r2, overall_rmse computed from your CV loop

metrics_df = pd.DataFrame({
    "Metric": [
        "Average R-squared",
        "Average RMSE",
        "Overall R-squared (pooled)",
        "Overall RMSE (pooled)",
        "Overall R-squared 95% CI",
        "Overall R-squared p-value",
        "Pearson Correlation Coefficient",
        "Pearson p-value"
    ],
    "Value": [
        f"{avg_r2:.4f}",
        f"{avg_rmse:.4f}",
        f"{overall_r2:.4f}",
        f"{overall_rmse:.4f}",
        f"{ci_lower:.4f} - {ci_upper:.4f}",
        f"{p_value:.4f}",
        f"{pearson_corr:.4f}",
        f"{p_value_Pearson:.4f}"
    ]
})

# Save the dataframe as CSV
csv_file = os.path.join(output_path, "metrics.csv")
metrics_df.to_csv(csv_file, index=False)
print(f"Metrics saved to: {csv_file}")


# Save each list separately as a .npy file
np.save(os.path.join(output_path, "y_pred_list.npy"), y_pred_list, allow_pickle=True)
np.save(os.path.join(output_path, "y_true_list.npy"), y_true_list, allow_pickle=True)
np.save(os.path.join(output_path, "fold_r2.npy"), fold_r2, allow_pickle=True)
np.save(os.path.join(output_path, "fold_rmse.npy"), fold_rmse, allow_pickle=True)
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
plt.savefig(os.path.join(output_path, 'shap_summary_training.png'))
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
plt.savefig(os.path.join(output_path, 'shap_summary_test.png'))
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