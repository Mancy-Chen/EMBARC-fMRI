# Mancy Chen 20/02/2025
# Regression with BART
import os
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
sys.path.append('/.../miniconda3/lib/python3.10/site-packages')
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
# --- Read and process data (unchanged) ---
x_path = '/.../EMBARC/data/06_BART_regression/Input/x/Site_normalization/Tier0_ablation/Imaging_features/Tier1_selected_ses-2_PLA.csv'
y_path = "/.../EMBARC/data/06_BART_regression/Input/y/Imputation/deltaHAMD_ses_2_PLA.csv"
output_path = '/.../EMBARC/data/06_BART_regression/Output/Regression_plot/Tier_0_ablation/Imaging_features/04_ses-2_PLA/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created directory: {output_path}")
else:
    print(f"Directory already exists: {output_path}")

medication = 'PLA' # SER or PLA
ses_number = 'ses-2' # ses-1 or ses-2
tier = 'Tier1/2' #Tier1/2 or Tier3
feature_number = 14 # 1302 or 14 or 413
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

# def filter_medication(X, y, medication='PLA'):
#     X_filtered = X.copy()
#     # Ensure Medication column is numeric for comparison
#     X_filtered["Medication"] = pd.to_numeric(X_filtered["Medication"], errors="coerce")
#
#     if medication == "PLA":
#         mask = X_filtered["Medication"] == 0
#     elif medication == "SER":
#         mask = X_filtered["Medication"] == 1
#     else:
#         mask = np.ones(len(X_filtered), dtype=bool)
#
#     X_filtered = X_filtered[mask]
#
#     # Filter y using the same mask
#     if hasattr(y, "loc"):
#         y_filtered = y.loc[X_filtered.index]
#     else:
#         y_filtered = y[mask.values]
#
#     return X_filtered, y_filtered
# # Apply the filtering before CV:
# X, y = filter_medication(X, y, medication= medication)

n_samples = X.shape[0]
n_features = X.shape[1]
print('After filtering by medication: n_samples:', n_samples, '; n_features:', n_features, '\n')

#######################################################################################################################
# Tier 1 XGBoost session 1
###############################################################################
# Example utility functions and data loading omitted for brevity...
# Assume you have X, y loaded similarly to your existing code.
###############################################################################
class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that:
      1. Imputes BMI by median
      2. Imputes is_employed by most frequent
      3. IterativeImputer for MASQ w0/w1 columns (aa/ad/gd)
      4. IterativeImputer for w1_score_17 (using w0, w2, w3, w4, w6, w8, w9, w10, w12, w16)
      5. Creates r1_score_17 = w1_score_17 / w0_score_17
      6. IterativeImputer for shaps_total_continuous-w0/w1
      7. KNNImputer for Hippocampus, Default-mode, and ACC columns,
         each in separate KNN models, but only for the specified session
         ('ses-1' or 'ses-2').
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

        # --------------------------------------
        # 7) KNN for HPC/DM/ACC (ses-1)
        # --------------------------------------
        self.hpc1_cols_ = [
            "ses-1-Left-Hippocampus-original-shape-VoxelVolume",
            "ses-1-Right-Hippocampus-original-shape-VoxelVolume"
        ]
        self.dm1_cols_ = [
            "default-mode-ses-1-mean",
            "default-mode-ses-1-std"
        ]
        self.acc1_cols_ = [
            "roi-Anterior-Cingulate-ses-1-mean",
            "roi-Anterior-Cingulate-ses-1-std"
        ]
        self.hpc1_knn_ = KNNImputer(n_neighbors=self.n_neighbors)
        self.dm1_knn_ = KNNImputer(n_neighbors=self.n_neighbors)
        self.acc1_knn_ = KNNImputer(n_neighbors=self.n_neighbors)

        # --------------------------------------
        # 7) KNN for HPC/DM/ACC (ses-2)
        # --------------------------------------
        self.hpc2_cols_ = [
            "ses-2-Left-Hippocampus-original-shape-VoxelVolume",
            "ses-2-Right-Hippocampus-original-shape-VoxelVolume"
        ]
        self.dm2_cols_ = [
            "default-mode-ses-2-mean",
            "default-mode-ses-2-std"
        ]
        self.acc2_cols_ = [
            "roi-Anterior-Cingulate-ses-2-mean",
            "roi-Anterior-Cingulate-ses-2-std"
        ]
        self.hpc2_knn_ = KNNImputer(n_neighbors=self.n_neighbors)
        self.dm2_knn_ = KNNImputer(n_neighbors=self.n_neighbors)
        self.acc2_knn_ = KNNImputer(n_neighbors=self.n_neighbors)

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
        # HPC/DM/ACC based on session
        if self.session == 'ses-1':
            numeric_cols.update([c for c in self.hpc1_cols_ if c in df.columns])
            numeric_cols.update([c for c in self.dm1_cols_ if c in df.columns])
            numeric_cols.update([c for c in self.acc1_cols_ if c in df.columns])
        elif self.session == 'ses-2':
            numeric_cols.update([c for c in self.hpc2_cols_ if c in df.columns])
            numeric_cols.update([c for c in self.dm2_cols_ if c in df.columns])
            numeric_cols.update([c for c in self.acc2_cols_ if c in df.columns])

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

        # --------------------------------------
        # 7) Fit HPC/DM/ACC KNN based on session
        # --------------------------------------
        if self.session == 'ses-1':
            # HPC
            hpc1_exist = [c for c in self.hpc1_cols_ if c in X_fit.columns]
            if len(hpc1_exist) == len(self.hpc1_cols_):
                self.hpc1_knn_.fit(X_fit[hpc1_exist])

            # DM
            dm1_exist = [c for c in self.dm1_cols_ if c in X_fit.columns]
            if len(dm1_exist) == len(self.dm1_cols_):
                self.dm1_knn_.fit(X_fit[dm1_exist])

            # ACC
            acc1_exist = [c for c in self.acc1_cols_ if c in X_fit.columns]
            if len(acc1_exist) == len(self.acc1_cols_):
                self.acc1_knn_.fit(X_fit[acc1_exist])

        elif self.session == 'ses-2':
            # HPC
            hpc2_exist = [c for c in self.hpc2_cols_ if c in X_fit.columns]
            if len(hpc2_exist) == len(self.hpc2_cols_):
                self.hpc2_knn_.fit(X_fit[hpc2_exist])

            # DM
            dm2_exist = [c for c in self.dm2_cols_ if c in X_fit.columns]
            if len(dm2_exist) == len(self.dm2_cols_):
                self.dm2_knn_.fit(X_fit[dm2_exist])

            # ACC
            acc2_exist = [c for c in self.acc2_cols_ if c in X_fit.columns]
            if len(acc2_exist) == len(self.acc2_cols_):
                self.acc2_knn_.fit(X_fit[acc2_exist])

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

        # --------------------------------------
        # 7) KNN HPC/DM/ACC based on session
        # --------------------------------------
        if self.session == 'ses-1':
            # HPC
            hpc1_exist = [c for c in self.hpc1_cols_ if c in X_out.columns]
            if len(hpc1_exist) == len(self.hpc1_cols_):
                X_out[hpc1_exist] = self.hpc1_knn_.transform(X_out[hpc1_exist])

            # DM
            dm1_exist = [c for c in self.dm1_cols_ if c in X_out.columns]
            if len(dm1_exist) == len(self.dm1_cols_):
                X_out[dm1_exist] = self.dm1_knn_.transform(X_out[dm1_exist])

            # ACC
            acc1_exist = [c for c in self.acc1_cols_ if c in X_out.columns]
            if len(acc1_exist) == len(self.acc1_cols_):
                X_out[acc1_exist] = self.acc1_knn_.transform(X_out[acc1_exist])

        elif self.session == 'ses-2':
            # HPC
            hpc2_exist = [c for c in self.hpc2_cols_ if c in X_out.columns]
            if len(hpc2_exist) == len(self.hpc2_cols_):
                X_out[hpc2_exist] = self.hpc2_knn_.transform(X_out[hpc2_exist])

            # DM
            dm2_exist = [c for c in self.dm2_cols_ if c in X_out.columns]
            if len(dm2_exist) == len(self.dm2_cols_):
                X_out[dm2_exist] = self.dm2_knn_.transform(X_out[dm2_exist])

            # ACC
            acc2_exist = [c for c in self.acc2_cols_ if c in X_out.columns]
            if len(acc2_exist) == len(self.acc2_cols_):
                X_out[acc2_exist] = self.acc2_knn_.transform(X_out[acc2_exist])

        return X_out

# class MedicationFilter(BaseEstimator, TransformerMixin):
#     """
#     Filters rows in the DataFrame based on the 'Medication' column.
#     If medication='PLA', keep rows where Medication == 0.
#     If medication='SER', keep rows where Medication == 1.
#     Also filters y accordingly if provided.
#     """
#
#     def __init__(self, medication='PLA'):
#         """
#         Parameters
#         ----------
#         medication : str, default='PLA'
#             Indicates which subjects to keep.
#             - 'PLA' => Medication == 0
#             - 'SER' => Medication == 1
#         """
#         self.medication = medication
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         X = X.copy()
#         if "Medication" in X.columns:
#             # Convert to numeric so that '0' becomes 0 and '1' becomes 1
#             X["Medication"] = pd.to_numeric(X["Medication"], errors="coerce")
#             if self.medication == "PLA":
#                 mask = X["Medication"] == 0
#             elif self.medication == "SER":
#                 mask = X["Medication"] == 1
#             else:
#                 # If medication is not 'PLA' or 'SER', keep all rows.
#                 mask = np.ones(len(X), dtype=bool)
#             X = X[mask]
#             if y is not None:
#                 # If y is a pandas Series or DataFrame, use its index to filter.
#                 if hasattr(y, "loc"):
#                     y = y.loc[X.index]
#                 else:
#                     # Otherwise, assume y is a numpy array aligned with X.
#                     y = y[mask.values]
#                 return X, y
#         if y is not None:
#             return X, y
#         return X

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, session='ses-1', tier='Tier1/2', cols_to_drop_1=None, cols_to_drop_2=None):
        self.session = session
        self.tier = tier
        self.cols_to_drop_1 = cols_to_drop_1 or []
        self.cols_to_drop_2 = cols_to_drop_2 or []
        # HPC/DM/ACC columns for ses-1
        self.additional_ses1 = [
            'ses-1-Left-Hippocampus-original-shape-VoxelVolume',
            'ses-1-Right-Hippocampus-original-shape-VoxelVolume',
            'default-mode-ses-1-mean',
            'default-mode-ses-1-std',
            'roi-Anterior-Cingulate-ses-1-mean',
            'roi-Anterior-Cingulate-ses-1-std'
        ]
        # HPC/DM/ACC columns for ses-2
        self.additional_ses2 = [
            "ses-2-Left-Hippocampus-original-shape-VoxelVolume",
            "ses-2-Right-Hippocampus-original-shape-VoxelVolume",
            "default-mode-ses-2-mean",
            "default-mode-ses-2-std",
            "roi-Anterior-Cingulate-ses-2-mean",
            "roi-Anterior-Cingulate-ses-2-std"
        ]
        # This will hold the final columns we decide to drop
        self.cols_to_drop_ = None

    def fit(self, X, y=None):
        if self.session == 'ses-1':
            if self.tier == 'Tier1/2':
                self.cols_to_drop_ = self.cols_to_drop_1 + self.additional_ses1
            else:  # tier == 'Tier3'
                self.cols_to_drop_ = self.cols_to_drop_1
        else:  # session == 'ses-2'
            if self.tier == 'Tier1/2':
                self.cols_to_drop_ = self.cols_to_drop_2 + self.additional_ses2
            else:  # tier == 'Tier3'
                self.cols_to_drop_ = self.cols_to_drop_2
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        return X_copy.drop(columns=self.cols_to_drop_, errors='ignore')


------------------ CustomCombatTransformer ------------------
class NeuroHarmonizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_number):
        """
        Parameters:
            feature_number (int): The number of radiomics features (starting from column 0)
                                  that will be harmonized.
        """
        self.feature_number = feature_number
        self.model_ = None

    def fit(self, X, y=None):
        """
        Learns the harmonization model using the radiomics features and covariates.

        Parameters:
            X (pandas.DataFrame): Input dataframe with feature columns and the last 4 covariate columns.
            y: Ignored.

        Returns:
            self
        """
        # --- Step 1: Prepare the covariate DataFrame ---
        covariate_df = X.iloc[:, -4:].copy()
        covariate_df.columns = ['SITE', 'age', 'age_squared', 'gender']
        covariate_df['SITE'] = covariate_df['SITE'].astype(str)
        covariate_df['age'] = pd.to_numeric(covariate_df['age'], errors='coerce')
        covariate_df['age_squared'] = pd.to_numeric(covariate_df['age_squared'], errors='coerce')
        covariate_df['gender'] = pd.to_numeric(covariate_df['gender'], errors='coerce')

        # --- Step 2: Convert the feature columns (all but the last 4) to a numpy array ---
        data_array = X.iloc[:, :-4].to_numpy(dtype=np.float64)

        # --- Step 3: Split the features into radiomics and clinical parts ---
        radiomics = data_array[:, :self.feature_number]
        # clinical part is not used during learning but will be preserved during transform
        # clinical = data_array[:, self.feature_number:]

        # --- Step 4: Learn the harmonization model on the radiomics features ---
        model_out = harmonizationLearn(radiomics, covars=covariate_df)
        # In case harmonizationLearn returns a tuple, use the first element as the model
        self.model_ = model_out[0] if isinstance(model_out, tuple) else model_out

        return self

    def transform(self, X):
        """
        Applies the learned harmonization model to the radiomics features and
        concatenates them with the clinical features.

        Parameters:
            X (pandas.DataFrame): Input dataframe with feature columns and the last 4 covariate columns.

        Returns:
            merged_data (numpy.ndarray): The transformed array where harmonized radiomics features
                                         are merged with the intact clinical features.
        """
        # --- Step 1: Prepare the covariate DataFrame ---
        covariate_df = X.iloc[:, -4:].copy()
        covariate_df.columns = ['SITE', 'age', 'age_squared', 'gender']
        covariate_df['SITE'] = covariate_df['SITE'].astype(str)
        covariate_df['age'] = pd.to_numeric(covariate_df['age'], errors='coerce')
        covariate_df['age_squared'] = pd.to_numeric(covariate_df['age_squared'], errors='coerce')
        covariate_df['gender'] = pd.to_numeric(covariate_df['gender'], errors='coerce')

        # --- Step 2: Convert the feature columns (all but the last 4) to a numpy array ---
        data_array = X.iloc[:, :-4].to_numpy(dtype=np.float64)

        # --- Step 3: Split the features into radiomics and clinical parts ---
        radiomics = data_array[:, :self.feature_number]
        clinical = data_array[:, self.feature_number:]

        # --- Step 4: Apply the learned harmonization model to the radiomics features ---
        harmonized_radiomics = harmonizationApply(radiomics, covars=covariate_df, model=self.model_)

        # --- Step 5: Merge the harmonized radiomics features with the clinical features ---
        merged_data = np.hstack([harmonized_radiomics, clinical])

        return merged_data

# ------------------ CustomFeatureSelector (RFE) ------------------

# Skip Combat when feature_number = 0
class NeuroHarmonizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_number):
        """
        Parameters:
            feature_number (int): The number of radiomics features (starting from column 0)
                                  that will be harmonized.
        """
        self.feature_number = feature_number
        self.model_ = None

    def fit(self, X, y=None):
        """
        Learns the harmonization model using the radiomics features and covariates.

        Parameters:
            X (pandas.DataFrame): Input dataframe with feature columns and the last 4 covariate columns.
            y: Ignored.

        Returns:
            self
        """
        # Prepare covariates
        covariate_df = X.iloc[:, -4:].copy()
        covariate_df.columns = ['SITE', 'age', 'age_squared', 'gender']
        covariate_df['SITE'] = covariate_df['SITE'].astype(str)
        covariate_df['age'] = pd.to_numeric(covariate_df['age'], errors='coerce')
        covariate_df['age_squared'] = pd.to_numeric(covariate_df['age_squared'], errors='coerce')
        covariate_df['gender'] = pd.to_numeric(covariate_df['gender'], errors='coerce')

        # Feature matrix (excluding covariates)
        data_array = X.iloc[:, :-4].to_numpy(dtype=np.float64)
        radiomics = data_array[:, :self.feature_number]

        # Learn harmonization model only if radiomics features exist
        if self.feature_number > 0:
            model_out = harmonizationLearn(radiomics, covars=covariate_df)
            self.model_ = model_out[0] if isinstance(model_out, tuple) else model_out
        else:
            self.model_ = None

        return self

    def transform(self, X):
        """
        Applies the learned harmonization model to the radiomics features and
        concatenates them with the clinical features.

        Parameters:
            X (pandas.DataFrame): Input dataframe with feature columns and the last 4 covariate columns.

        Returns:
            merged_data (numpy.ndarray): The transformed array where harmonized radiomics features
                                         are merged with the intact clinical features.
        """
        # Prepare covariates
        covariate_df = X.iloc[:, -4:].copy()
        covariate_df.columns = ['SITE', 'age', 'age_squared', 'gender']
        covariate_df['SITE'] = covariate_df['SITE'].astype(str)
        covariate_df['age'] = pd.to_numeric(covariate_df['age'], errors='coerce')
        covariate_df['age_squared'] = pd.to_numeric(covariate_df['age_squared'], errors='coerce')
        covariate_df['gender'] = pd.to_numeric(covariate_df['gender'], errors='coerce')

        # Feature matrix (excluding covariates)
        data_array = X.iloc[:, :-4].to_numpy(dtype=np.float64)
        radiomics = data_array[:, :self.feature_number]
        clinical = data_array[:, self.feature_number:]

        # Apply harmonization if radiomics model is available
        if self.feature_number > 0 and self.model_ is not None:
            harmonized_radiomics = harmonizationApply(radiomics, covars=covariate_df, model=self.model_)
            merged_data = np.hstack([harmonized_radiomics, clinical])
        else:
            # No radiomics to harmonize: return clinical features only
            merged_data = clinical

        return merged_data

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

ComBat_transformer = NeuroHarmonizeTransformer(feature_number=feature_number)

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
    # Step 2: Filter medication=1 or 0, depending on medication='SER' or 'PLA'
    # ("med_filter", MedicationFilter(medication=medication)),
    # Step 3: Drop columns (depends on session, tier, and your predefined lists)
    ("drop_cols", DropColumns(
        session=ses_number,
        tier=tier,
        cols_to_drop_1=cols_to_drop_1,
        cols_to_drop_2=cols_to_drop_2
    )),
    # Step 4: NeuroCombat with covariates
    ("Combat", ComBat_transformer),  # Assuming you've already configured covariates inside
    # Step 5: Scale
    ("scaler", RobustScaler()),
    # Step 6: Feature selection
    ("selector", CustomFeatureSelector(estimator=selector_estimator)),
    # Step 7: XGB
    ("xgb", XGBRegressor(
        random_state=random_seed,
        eval_metric='rmse'
    ))
])

# # # Checking for each step's shape
# pipeline1 = pipeline.fit(X,y)
# X1 = pipeline.steps[0][1].transform(X)
# print("X shape in step 1: ", X1.shape)
# X2 = pipeline.steps[1][1].transform(X1)
# print("X shape in step 2: ", X2.shape)
# print(repr(X2.columns))
# X3 = pipeline.steps[2][1].transform(X2)
# print("X shape in step 3: ", X3.shape)
# X4 = pipeline.steps[3][1].transform(X3)
# print("X shape in step 4: ", X4.shape)
# # X5 = pipeline.steps[4][1].transform(X4)
# # print("X shape in step 5: ", X5.shape)
# # X6 = pipeline.steps[5][1].transform(X5)
# # print("X shape in step 6: ", X6.shape)
#
# # Convert the array to a DataFrame. Optionally, specify column names.
# df = pd.DataFrame(X4.T)
#
# # Save the DataFrame to a CSV file without the index.
# df.to_csv("X4.csv", index=False)


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
    n_jobs=-1,
    cv=cv_inner,
    random_state=random_seed
)

# ------------------ Outer CV Loop ------------------
cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_seed)

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

for i in range(n_bootstraps):
    # Sample indices with replacement
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    boot_r2 = r2_score(y_true_arr[indices], y_pred_arr[indices])
    boot_r2_scores.append(boot_r2)

# Compute the 95% CI using the 2.5th and 97.5th percentiles
ci_lower = np.percentile(boot_r2_scores, 2.5)
ci_upper = np.percentile(boot_r2_scores, 97.5)
print(f"Bootstrapped R² 95% CI: {ci_lower:.2f} – {ci_upper:.2f}")

# ------------------ Permutation Testing for Overall R² p-value ------------------
n_permutations = 1000
permuted_r2_scores = []

for i in range(n_permutations):
    # Permute the true labels to break any association
    y_true_permuted = np.random.permutation(y_true_arr)
    permuted_r2 = r2_score(y_true_permuted, y_pred_arr)
    permuted_r2_scores.append(permuted_r2)

# p-value: proportion of permuted R² values that are >= observed overall_r2
p_value = np.mean(np.array(permuted_r2_scores) >= overall_r2)
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

#######################################################################################################################
# Permutation test
cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_seed)

# Containers for overall results
y_pred_list = []
y_true_list = []
fold_r2 = []
fold_rmse = []
permutation_p_values_r2 = []
permutation_p_values_rmse = []
outer_fold_counter = 0
total_outer_folds = cv_outer.get_n_splits()
all_train_shap_values = []
all_test_shap_values = []
all_best_X_train = []
all_best_X_test = []
elapsed_times = []

# Set the number of permutations per outer fold
n_permutations = 50

for train_index, test_index in cv_outer.split(X, y):
    start_time = time.time()
    outer_fold_counter += 1

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # ----- Fit on the true data -----
    optimizer.fit(X_train, y_train)
    print(f"Fold {outer_fold_counter} - Best inner-fold score: {optimizer.best_score_:.4f}")
    print(f"Fold {outer_fold_counter} - Best params: {optimizer.best_params_}")

    best_pipeline = optimizer.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    y_pred_list.extend(y_pred)
    y_true_list.extend(y_test)

    r2_obs = r2_score(y_test, y_pred)
    rmse_obs = np.sqrt(mean_squared_error(y_test, y_pred))
    fold_r2.append(r2_obs)
    fold_rmse.append(rmse_obs)
    print(f"Fold {outer_fold_counter} - Observed R²: {r2_obs:.4f}, RMSE: {rmse_obs:.4f}")

    # ----- Permutation Test for the current outer fold -----
    perm_r2 = []
    perm_rmse = []
    for perm in range(n_permutations):
        # Shuffle y_train for permutation
        y_train_perm = shuffle(y_train, random_state=perm)
        # Re-fit the optimizer on permuted training labels
        optimizer.fit(X_train, y_train_perm)
        best_pipeline_perm = optimizer.best_estimator_
        y_pred_perm = best_pipeline_perm.predict(X_test)
        perm_r2.append(r2_score(y_test, y_pred_perm))
        perm_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred_perm)))
    # Calculate empirical p-values:
    # For R², count how many permuted scores are greater than or equal to the observed
    p_value_r2 = np.mean(np.array(perm_r2) >= r2_obs)
    # For RMSE, lower is better so count permuted RMSE less than or equal to observed
    p_value_rmse = np.mean(np.array(perm_rmse) <= rmse_obs)
    permutation_p_values_r2.append(p_value_r2)
    permutation_p_values_rmse.append(p_value_rmse)
    print(f"Fold {outer_fold_counter} - Permutation test p-value for R²: {p_value_r2:.4f}, RMSE: {p_value_rmse:.4f}")

    # ----- SHAP Analysis for the final XGBoost -----
    transform_pipeline = Pipeline(best_pipeline.steps[:-1])  # all steps except the final xgb
    final_xgb = best_pipeline.named_steps['xgb']
    explainer = shap.TreeExplainer(final_xgb)
    best_X_train = transform_pipeline.transform(X_train)
    best_X_test = transform_pipeline.transform(X_test)
    shap_values_train = explainer.shap_values(best_X_train)
    shap_values_test = explainer.shap_values(best_X_test)
    all_train_shap_values.append(shap_values_train)
    all_test_shap_values.append(shap_values_test)
    all_best_X_train.append(best_X_train)
    all_best_X_test.append(best_X_test)

    elapsed_time = time.time() - start_time
    elapsed_times.append(elapsed_time)
    progress = outer_fold_counter / total_outer_folds * 100
    print(f"Outer fold {outer_fold_counter}/{total_outer_folds} complete. Progress: {progress:.2f}%. "
          f"Time for this fold: {elapsed_time:.2f} seconds.")

# Print overall results
mean_r2 = np.mean(fold_r2)
mean_rmse = np.mean(fold_rmse)
mean_p_value_r2 = np.mean(permutation_p_values_r2)
mean_p_value_rmse = np.mean(permutation_p_values_rmse)

print(f"\nFinal Mean R²: {mean_r2:.4f}, Mean RMSE: {mean_rmse:.4f}")
print(f"Average permutation test p-value for R²: {mean_p_value_r2:.4f}")
print(f"Average permutation test p-value for RMSE: {mean_p_value_rmse:.4f}")
# ------------------ Save p-values as CSV ------------------
# Create a DataFrame with fold-level results
output_file = os.path.join(output_path, "p_values.csv")
results_df = pd.DataFrame({
    "Fold": np.arange(1, total_outer_folds + 1),
    "Observed_R2": fold_r2,
    "Observed_RMSE": fold_rmse,
    "Permutation_p_value_R2": permutation_p_values_r2,
    "Permutation_p_value_RMSE": permutation_p_values_rmse
})
results_df.to_csv(output_file, index=False)
print(f"P-values and fold results saved to {output_file}")
#########################################################################################################################
# Compare among groups
# Wilcoxon test
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

# 1) Load your arrays (replace filenames as needed)
#    y_true might be shape (n,1) so we squeeze to 1-D
y_true = np.load("/.../EMBARC/data/06_BART_regression/Output/Regression_plot/Tier_1/23_Tier_1_ses_2_PLA/y_true_list.npy").squeeze()     # shape (n,)
pred1  = np.load("/.../EMBARC/data/06_BART_regression/Output/Regression_plot/Tier_1/23_Tier_1_ses_2_PLA/y_pred_list.npy")           # Tier 1 predictions
pred2  = np.load("/.../EMBARC/data/06_BART_regression/Output/Regression_plot/Tier_2/13_Tier_2_ses_2_PLA/y_pred_list.npy")           # Tier 2 predictions
pred3  = np.load("/.../EMBARC/data/06_BART_regression/Output/Regression_plot/RVM/23_ses_2_PLA/y_pred_list.npy")           # Tier 3 predictions

# 2) Sanity check: all must have the same length
assert y_true.ndim == 1, f"y_true must be 1-D, got shape {y_true.shape}"
n = y_true.shape[0]
for name, arr in [("pred1", pred1), ("pred2", pred2), ("pred3", pred3)]:
    assert arr.shape == (n,), f"{name} has shape {arr.shape}, expected ({n},)"

# 3) Compute per-sample errors (choose squared error here)
err1 = (y_true - pred1) ** 2
err2 = (y_true - pred2) ** 2
err3 = (y_true - pred3) ** 2

# 4) Omnibus Friedman test across the three models
stat, p = friedmanchisquare(err1, err2, err3)
print(f"Friedman test: χ² = {stat:.2f}, p = {p:.4f}")

# 5) If significant, do pairwise Wilcoxon signed-rank tests
alpha = 0.05
m = 3  # number of comparisons
alpha_corr = alpha / m
print(f"Bonferroni‐corrected α = {alpha_corr:.4f}\n")

def pairwise_wilcoxon(a, b, name_a, name_b):
    w_stat, p_val = wilcoxon(a, b)
    sig = "✓" if p_val < alpha_corr else "✗"
    print(f"{name_a} vs {name_b}: W = {w_stat:.2f}, p = {p_val:.4f} {sig}")

pairwise_wilcoxon(err1, err2, "Tier 1", "Tier 2")
pairwise_wilcoxon(err2, err3, "Tier 2", "Tier 3")
pairwise_wilcoxon(err1, err3, "Tier 1", "Tier 3")

#######################################################################
import numpy as np
import random
from sklearn.metrics import r2_score
from scipy.stats import ttest_rel
# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ------------------ Load Sertraline Group Data (Session 1) ------------------
ser_y_pred = np.load(
    '/.../EMBARC/data/06_BART_regression/Output/Regression_plot/RVM/22_ses_2_SER/y_pred_list.npy',
    allow_pickle=True
)
ser_y_true = np.load(
    '/.../EMBARC/data/06_BART_regression/Output/Regression_plot/RVM/22_ses_2_SER/y_true_list.npy',
    allow_pickle=True
)

# # ------------------ Load Placebo Group Data (Session 1) ------------------

pla_y_pred = np.load(
    '/.../EMBARC/data/06_BART_regression/Output/Regression_plot/RVM/23_ses_2_PLA/y_pred_list.npy',
    allow_pickle=True
)
pla_y_true = np.load(
    '/.../EMBARC/data/06_BART_regression/Output/Regression_plot/RVM/23_ses_2_PLA/y_true_list.npy',
    allow_pickle=True
)


print("Data loaded successfully.")


# Compute Pearson's correlation coefficient
pearson_corr, p_value_Pearson = pearsonr(np.ravel(ser_y_true), np.ravel(ser_y_pred))
print("Pearson correlation coefficient:", pearson_corr)
print("p-value:", p_value_Pearson)

# ------------------ Bootstrapping to Compare R² between Groups ------------------
# Calculate bootstrapped differences in R² (sertraline minus placebo)
n_bootstraps = 1000
diff_r2_bootstrap = []

n_ser = len(ser_y_true)
n_pla = len(pla_y_true)

for i in range(n_bootstraps):
    # Sample indices with replacement for each group
    indices_ser = np.random.choice(n_ser, size=n_ser, replace=True)
    indices_pla = np.random.choice(n_pla, size=n_pla, replace=True)

    boot_r2_ser = r2_score(ser_y_true[indices_ser], ser_y_pred[indices_ser])
    boot_r2_pla = r2_score(pla_y_true[indices_pla], pla_y_pred[indices_pla])

    diff_r2_bootstrap.append(boot_r2_ser - boot_r2_pla)

# Compute the 95% confidence interval for the difference in R²
ci_lower = np.percentile(diff_r2_bootstrap, 2.5)
ci_upper = np.percentile(diff_r2_bootstrap, 97.5)
mean_diff = np.mean(diff_r2_bootstrap)

print(f"Bootstrapped Difference in R² (Sertraline - Placebo): {mean_diff:.2f}")
print(f"95% CI for Difference in R²: {ci_lower:.2f} to {ci_upper:.2f}")

# ------------------ Permutation Testing for R² Difference ------------------
# Compute observed R² for each group
observed_r2_ser = r2_score(ser_y_true, ser_y_pred)
observed_r2_pla = r2_score(pla_y_true, pla_y_pred)
observed_diff = observed_r2_ser - observed_r2_pla

# Combine data for permutation testing
combined_y_true = np.concatenate([ser_y_true, pla_y_true])
combined_y_pred = np.concatenate([ser_y_pred, pla_y_pred])
group_labels = np.array([1] * len(ser_y_true) + [0] * len(pla_y_true))

n_permutations = 1000
permuted_diffs = []

for i in range(n_permutations):
    # Shuffle the group labels
    shuffled_labels = np.random.permutation(group_labels)

    # Create permuted groups based on the shuffled labels
    ser_indices = (shuffled_labels == 1)
    pla_indices = (shuffled_labels == 0)

    # Compute R² only if both groups have at least one sample
    if np.sum(ser_indices) > 0 and np.sum(pla_indices) > 0:
        r2_ser_perm = r2_score(combined_y_true[ser_indices], combined_y_pred[ser_indices])
        r2_pla_perm = r2_score(combined_y_true[pla_indices], combined_y_pred[pla_indices])
        permuted_diffs.append(r2_ser_perm - r2_pla_perm)

permuted_diffs = np.array(permuted_diffs)
# p-value: proportion of permuted differences as extreme as the observed difference
p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

print(f"Observed Difference in R² (Sertraline - Placebo): {observed_diff:.2f}")
print(f"Permutation test p-value: {p_value:.4f}")

# ------------------ Paired t test ------------------
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# Compute per‐sample squared errors
ser_sq_errors = (ser_y_true - ser_y_pred) ** 2
pla_sq_errors = (pla_y_true - pla_y_pred) ** 2

# Make sure shapes match
if ser_sq_errors.shape != pla_sq_errors.shape:
    raise ValueError("ser and pla error arrays must have the same shape!")

# --- Option A: Flatten everything into one long vector ---
ser_flat = ser_sq_errors.ravel()
pla_flat = pla_sq_errors.ravel()

# Paired t-test (all observations)
t_stat_flat, p_flat = ttest_rel(ser_flat, pla_flat, axis=None)
print("Flattened paired t-test:        "
      f"t = {t_stat_flat:.2f}, p = {p_flat:.4f}")

# Wilcoxon signed‐rank (all observations)
w_stat_flat, p_w_flat = wilcoxon(ser_flat, pla_flat)
print("Flattened Wilcoxon signed-rank: "
      f"W = {w_stat_flat:.2f}, p = {p_w_flat:.4f}")

# --- Option B: First average per subject (across measures) ---
if ser_sq_errors.ndim > 1:
    # compute one mean error per subject
    ser_mean = np.nanmean(ser_sq_errors, axis=1)
    pla_mean = np.nanmean(pla_sq_errors, axis=1)

    # Paired t-test across subjects
    t_stat_subj, p_subj = ttest_rel(ser_mean, pla_mean)
    print("Subject-mean paired t-test:        "
          f"t = {t_stat_subj:.2f}, p = {p_subj:.4f}")

    # Wilcoxon signed‐rank across subjects
    w_stat_subj, p_w_subj = wilcoxon(ser_mean, pla_mean)
    print("Subject-mean Wilcoxon signed-rank: "
          f"W = {w_stat_subj:.2f}, p = {p_w_subj:.4f}")
else:
    print("Option B skipped: errors are already 1D.")


##################################################################################################################################
# Open .pkl files
import pandas as pd
import pickle

# Replace 'filename.pkl' with the path to your pickle file
with open('/.../EMBARC/data/07_Maarten/data/data_frames/tbss.pkl', 'rb') as file:
    data = pickle.load(file)

# Check if the loaded object is a DataFrame
if isinstance(data, pd.DataFrame):
    # Save DataFrame as CSV without the index
    data.to_csv('/.../EMBARC/data/06_BART_regression/Input/x/Previous_study_result/tbss.csv', sep='\t', index=True)
    print("Data saved to output.csv")
else:
    print("The loaded object is not a pandas DataFrame.")

# Now 'data' holds the Python object loaded from the pickle file
print(data)
#####################################################################################################################
# Filter the csv
import pandas as pd

# Path to your Excel file
file_path = '/.../EMBARC/data/06_BART_regression/Input/x/Previous_study_result/ses-2_Combined-model.xlsx'

# Read in the sheets from the Excel file
sheet1 = pd.read_excel(file_path, sheet_name='Sheet1')
sheet2 = pd.read_excel(file_path, sheet_name='Sheet2')
sheet3 = pd.read_excel(file_path, sheet_name='Sheet3')
sheet4 = pd.read_excel(file_path, sheet_name='Sheet4')

# Extract the unique subject IDs from sheet1 (using 'subject_id')
subjects = sheet1[['subject_id']].drop_duplicates()

# Merge sheets 2, 3, and 4 with sheet1's subject IDs using a left join.
sheet2_merged = subjects.merge(sheet2, on='subject_id', how='left')
sheet3_merged = subjects.merge(sheet3, on='subject_id', how='left')
sheet4_merged = subjects.merge(sheet4, on='subject_id', how='left')

# Display the first few rows of each merged dataframe
print("Sheet 2 Merged:")
print(sheet2_merged.head())
print("\nSheet 3 Merged:")
print(sheet3_merged.head())
print("\nSheet 4 Merged:")
print(sheet4_merged.head())

# Optionally, save the merged data back to a new Excel file with multiple sheets
output_path = '/.../EMBARC/data/06_BART_regression/Input/x/Previous_study_result/ses-2_Combined-model_merged.xlsx'
with pd.ExcelWriter(output_path) as writer:
    sheet1.to_excel(writer, sheet_name='Sheet1', index=False)
    sheet2_merged.to_excel(writer, sheet_name='Sheet2', index=False)
    sheet3_merged.to_excel(writer, sheet_name='Sheet3', index=False)
    sheet4_merged.to_excel(writer, sheet_name='Sheet4', index=False)



#######################################################################################################################
# Feature distribution
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

# Load the data
# --- Utility Functions ---
def parse_score(score_str):
    """
    Extract numeric value from a string.
    For example, '+AC0-20' will return -20.0.
    """
    matches = re.findall(r"[-+]?\d*\.?\d+", score_str)
    if not matches:
        return np.nan
    for m in matches:
        if m.startswith('-'):
            try:
                return float(m)
            except:
                pass
    try:
        return float(matches[0])
    except:
        return np.nan

def remove_AF8(feature):
    # This pattern matches an optional plus sign, then "AF8", then an optional hyphen.
    return re.sub(r'\+?AF8?', '', feature)

# --- Read and process data (unchanged) ---
x_path = "/.../EMBARC/data/06_BART_regression/Input/x/Radiomics/radiomics_ses_1_SER.csv"
y_path = "/.../EMBARC/data/06_BART_regression/Input/y/deltaHAMD_ses_1_SER.csv"

# Process X
X_df = pd.read_csv(x_path, index_col=0)
X_df = X_df.astype(str).map(parse_score)
feature_names = X_df.index.tolist()
feature_names = [remove_AF8(fname) for fname in feature_names]
feature_names = [fname.replace("original-", "") for fname in feature_names]
X = X_df.T.astype(np.float64).to_numpy()
print("X shape (subjects x features + covariate):", X.shape)  # Expected (93, 1302)

# Process y
try:
    y_df = pd.read_csv(y_path, header=None, encoding='utf-8-sig')
    if y_df.shape[1] == 0:
        raise ValueError("Empty DataFrame")
    y = y_df.iloc[:, 0].apply(parse_score).to_numpy(dtype=np.float64)
except Exception:
    with open(y_path, 'r', encoding='utf-8-sig') as f:
        lines = f.read().strip().splitlines()
    y = np.array([
        parse_score(line.split(',')[0] if ',' in line else line.split()[0])
        for line in lines if line.strip()
    ], dtype=np.float64)
print("y shape:", y.shape)  # Expected (93,)

# Convert X (a NumPy array) into a DataFrame with the processed feature names as columns.
n_subjects, n_features = X.shape
data = pd.DataFrame(X, columns=feature_names)
print("Final X shape (subjects x features):", data.shape)

# Set parameters for plotting: 100 histograms per page (10x10 grid)
graphs_per_page = 100
n_pages = int(np.ceil(n_features / graphs_per_page))
output_pdf = "/.../EMBARC/data/06_BART_regression/Output/Feature_distribution/Scaled_ses_1_SER_histograms_grid.pdf"

with PdfPages(output_pdf) as pdf:
    for page in range(n_pages):
        start = page * graphs_per_page
        end = min((page + 1) * graphs_per_page, n_features)
        features_on_page = data.columns[start:end]

        n_plots = len(features_on_page)
        n_cols = 5
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 2 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(features_on_page):
            axes[i].hist(data[col], bins=20, edgecolor='black')
            axes[i].set_title(col, fontsize=8)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        # Remove any unused subplots in the grid.
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved multipage PDF with histograms to {output_pdf}")
