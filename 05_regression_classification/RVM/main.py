# Mancy Chen 21/03/2025
# Relevance Vector Machine (RVM)
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
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
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
from sklearn.metrics import mean_squared_error, make_scorer, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut, StratifiedKFold, cross_validate, GridSearchCV, \
    cross_val_score, cross_val_predict, RepeatedStratifiedKFold, RandomizedSearchCV, StratifiedShuffleSplit, learning_curve, RepeatedKFold
from skopt.space import Integer, Real, Categorical
from skopt import BayesSearchCV, Optimizer
from statsmodels import robust  # if needed for median_abs_deviation; otherwise use scipy.stats
from scipy.stats import median_abs_deviation
np.int = int  # Patch to allow legacy code to work
import matplotlib.pyplot as plt
import shap
from scipy.stats import ttest_ind, pearsonr, binomtest, spearmanr, kendalltau, pointbiserialr, median_abs_deviation
from neurocombat_sklearn import CombatModel
from neuroHarmonize import harmonizationLearn, harmonizationApply
import random


###########################################################################################################
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
x_path = '/.../EMBARC/data/10_ASL/ASL_data/radiomics/aggregate/Seperated/radiomics_matrix_ses-1_SER.csv'
y_path = "/.../EMBARC/data/10_ASL/ASL_data/y/y_seperated/remission/deltaHAMD_ses_1_SER.csv"
output_path = '/.../EMBARC/data/10_ASL/Output/Remission/01_ses-1_SER'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created directory: {output_path}")
else:
    print(f"Directory already exists: {output_path}")

# medication = 'SER' # SER or PLA
# ses_number = 'ses-2' # ses-1 or ses-2
# tier = 'Tier1/2' #Tier1/2 or Tier3
feature_number = 324 # 1302 or 14 or 413 or 252
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
# selected_features = 10
# max_display = 20

# For demonstration, we generate synthetic data.
# Imagine each row corresponds to a participant with 1235 ROI features.
# np.random.seed(42)
# n_samples = 100  # Number of participants
# n_features = 1235  # Number of ROI features (adjust as needed)
# X = np.random.randn(n_samples, n_features)
#
# # Simulate symptom change as a linear combination of features plus noise
# true_coef = np.random.randn(n_features)
# y = X.dot(true_coef) + np.random.randn(n_samples) * 0.1

# Process X
# X_df = pd.read_csv(x_path, index_col=0, header = None) # Without labels of columns
X_df = pd.read_csv(x_path, index_col=0) # with labels of columns
X_df = remove_substrings(X_df, remove_in_cells=True)
feature_names = X_df.index.tolist()
feature_names = [fname.replace("original-", "") for fname in feature_names]
# X = X_df.T # Be careful!
X = X_df
print("X shape (subjects, features):", X.shape)  # Expected (93, 1302)
# print(repr(X.columns))
n_samples = X.shape[0]

# Process y

y_df = pd.read_csv(y_path, header=None, encoding='utf-8-sig')
y_df = remove_substrings(y_df, remove_in_cells=True)
y = y_df.to_numpy(dtype=np.float64).squeeze()  # Convert to (n_samples,) shape
print("y shape:", y.shape)  # Should now output (93,)

#
#
# #######################################################################################################################
class RVMRegressor(BaseEstimator, RegressorMixin):
    """
    A simple implementation of a Relevance Vector Machine (RVM) for regression
    based on sparse Bayesian learning.

    Parameters
    ----------
    stoper : float, default=1e-3
        Convergence tolerance on the change in the weight vector.
    maxIts : int, default=10000
        Maximum number of iterations.
    maxAlpha : float, default=1e12
        Threshold for pruning weights. When alpha exceeds this value,
        the corresponding weight is set to zero.
    """

    def __init__(self, stoper=1e-3, maxIts=10000, maxAlpha=1e12):
        self.stoper = stoper
        self.maxIts = maxIts
        self.maxAlpha = maxAlpha

    def fit(self, X, y):
        # Ensure y is a 1D array
        y = np.ravel(y)
        # Add bias term (column of ones)
        N, D = X.shape
        X_bias = np.hstack([np.ones((N, 1)), X])
        D_bias = D + 1  # total parameters including bias

        # Initialize parameters
        w = np.zeros(D_bias)  # weights (bias and features)
        alpha = np.ones(D_bias - 1)  # hyperparameters for non-bias weights
        beta = 1.0  # noise precision
        biasalpha = 1.0  # fixed inverse variance for bias

        Xt_y = X_bias.T.dot(y)
        # Active set for non-bias features: True if alpha is below threshold
        act = alpha < self.maxAlpha
        # Always include the bias (first element is active)
        idxact = np.concatenate(([True], act))

        for itr in range(self.maxIts):
            prew = w.copy()
            # Use only the active parameters
            X_act = X_bias[:, idxact]
            alpha_act = alpha[act]
            diag_vec = np.concatenate(([biasalpha], alpha_act))
            A = beta * (X_act.T.dot(X_act)) + np.diag(diag_vec)
            Sigma = np.linalg.pinv(A)
            w[idxact] = beta * Sigma.dot(Xt_y[idxact])

            # Compute residual error
            residual = y - X_act.dot(w[idxact])
            Er = np.sum(residual ** 2)

            # Compute gamma values (effective number of parameters)
            diagSigma = np.diag(Sigma)
            gamma = 1 - np.concatenate(([biasalpha], alpha_act)) * diagSigma

            # Update alpha for non-bias features
            w_non_bias = w[1:]
            gamma_non_bias = gamma[1:]
            epsilon = 1e-16  # small constant to avoid division by zero
            alpha[act] = gamma_non_bias / (w_non_bias[act]**2 + epsilon)

            # Update noise precision beta
            beta = (N - np.sum(gamma)) / (Er + epsilon)

            # Update active set
            act = alpha < self.maxAlpha
            idxact = np.concatenate(([True], act))
            w[~idxact] = 0  # prune inactive weights

            # Check convergence on weight change
            errw = np.linalg.norm(w - prew)
            if errw < self.stoper:
                print(f"Converged at iteration {itr + 1}")
                break

        self.intercept_ = w[0]
        self.coef_ = w[1:]
        self.alpha_ = alpha
        self.beta_ = beta
        return self

    def predict(self, X):
        """Return continuous predictions for input X."""
        return self.intercept_ + X.dot(self.coef_)

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
        covariate_df.columns = ['SITE', 'Age', 'Age-squared', 'Gender']
        covariate_df['SITE'] = covariate_df['SITE'].astype(str)
        covariate_df['Age'] = pd.to_numeric(covariate_df['Age'], errors='coerce')
        covariate_df['Age-squared'] = pd.to_numeric(covariate_df['Age-squared'], errors='coerce')
        covariate_df['Gender'] = pd.to_numeric(covariate_df['Gender'], errors='coerce')

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
        covariate_df.columns = ['SITE', 'Age', 'Age-squared', 'Gender']
        covariate_df['SITE'] = covariate_df['SITE'].astype(str)
        covariate_df['Age'] = pd.to_numeric(covariate_df['Age'], errors='coerce')
        covariate_df['Age-squared'] = pd.to_numeric(covariate_df['Age-squared'], errors='coerce')
        covariate_df['Gender'] = pd.to_numeric(covariate_df['Gender'], errors='coerce')

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


# ----------------- Cross-Validation with Aggregation -----------------

ComBat_transformer = NeuroHarmonizeTransformer(feature_number=feature_number)
#
# # # Create a pipeline with feature scaling and the RVM regressor
# pipeline = Pipeline([
#     ("Combat", ComBat_transformer),  # Assuming you've already configured covariates inside
#     ('scaler', RobustScaler()),
#     ('rvm', RVMRegressor())
# ])
#
# # Setup 10×10 cross-validation using RepeatedKFold: 10 folds, 10 repetitions.
# rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
#
# # Dictionary to collect predictions for each participant (indexed by sample number)
# predictions = {i: [] for i in range(n_samples)}
#
# fold = 0
# for train_index, test_index in rkf.split(X):
#     fold += 1
#     # Fit the model on the training set using .iloc for DataFrame row indexing,
#     # and direct indexing for y (a NumPy array)
#     pipeline.fit(X.iloc[train_index], y[train_index])
#     # Predict on the test set (left-out participants) using .iloc for X
#     y_pred = pipeline.predict(X.iloc[test_index])
#     # Record predictions for each left-out participant
#     for i, idx in enumerate(test_index):
#         predictions[idx].append(y_pred[i])
#     print(f"Fold {fold} complete")
#
# # For each participant, compute the median prediction across the 10 times they were left out.
# y_pred_median = np.array([np.median(predictions[i]) for i in range(n_samples)])
#
# # Calculate basic metrics for regression
# rmse = np.sqrt(mean_squared_error(y, y_pred_median))
# r2 = r2_score(y, y_pred_median)
# y_flat = np.ravel(y)
# y_pred_median_flat = np.ravel(y_pred_median)
#
# # Pearson correlation test: returns correlation coefficient and p-value
# corr_coef, p_value = pearsonr(y_flat, y_pred_median_flat)
# mae = mean_absolute_error(y, y_pred_median)
#
# print("\nAggregated Cross-Validated Performance:")
# print(f"RMSE: {rmse:.4f}")
# print(f"R^2: {r2:.4f}")
# print(f"Correlation Coefficient: {corr_coef:.4f} (p-value: {p_value:.4f})")
# print(f"MAE: {mae:.4f}")
#
# # Save metrics to a CSV file
# metrics_dict = {
#     "RMSE": [rmse],
#     "R2": [r2],
#     "CorrelationCoefficient": [corr_coef],
#     "Correlation_p_value": [p_value],
#     "MAE": [mae]
# }
# df_metrics = pd.DataFrame(metrics_dict)
# csv_filename = os.path.join(output_path, "metrics.csv")
# df_metrics.to_csv(csv_filename, index=False)
# print(f"Metrics saved to {csv_filename}")
#
# # Calculate Pearson correlation and p-value
# r, p_value = pearsonr(y_flat, y_pred_median_flat)
#
# # Create the scatter plot
# plt.figure()
# plt.scatter(y, y_pred_median, alpha=0.7)
# plt.xlabel("True Symptom Change")
# plt.ylabel("Predicted Symptom Change")
# plt.title("10×10 Cross-Validated Predictions (Median Aggregation)")
#
# # Plot a best-fit line using flattened arrays
# p = np.polyfit(y_flat, y_pred_median_flat, 1)
# y_line = np.polyval(p, y_flat)
# plt.plot(y_flat, y_line, 'r-', linewidth=1.5)
#
# # Add correlation coefficient and p-value in the upper left corner of the plot
# plt.gca().text(0.05, 0.95, f"r: {r:.2f}\np: {p_value:.2f}",
#                transform=plt.gca().transAxes, fontsize=12,
#                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
#
# # Set axis limits and ticks
# plt.xlim(-15, 30)
# plt.ylim(-15, 30)
# plt.xticks([-10, 0, 10, 20, 30])
# plt.yticks([-10, 0, 10, 20, 30])
# plt.gca().set_aspect('equal', adjustable='box')
#
# # Save the plot to the output path using 300 dpi
# plot_filename = os.path.join(output_path, "predicted_vs_true_plot.png")
# plt.savefig(plot_filename, dpi=300)
# print(f"Plot saved to {plot_filename}")
# plt.show()
#
# #######################################################################################################################
# # sklearn RVM
# from skrvm import RVR  # Import the RVR implementation from sklearn-rvm
# #
# # # Create a pipeline with feature scaling and the RVR regressor
# pipeline = Pipeline([
#     ("Combat", ComBat_transformer),  # NeuroHarmonizeTransformer remains unchanged
#     ('scaler', RobustScaler()),
#     ('rvm', RVR(kernel='linear'))  # Use a linear kernel; modify parameters as needed
# ])
#
# # Setup 10×10 cross-validation using RepeatedKFold: 10 folds, 10 repetitions.
# rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
#
# # Dictionary to collect predictions for each participant (indexed by sample number)
# predictions = {i: [] for i in range(n_samples)}
#
# fold = 0
# for train_index, test_index in rkf.split(X):
#     fold += 1
#     # Fit the model on the training set using .iloc for DataFrame row indexing,
#     # and direct indexing for y (a NumPy array)
#     pipeline.fit(X.iloc[train_index], y[train_index])
#     # Predict on the test set (left-out participants) using .iloc for X
#     y_pred = pipeline.predict(X.iloc[test_index])
#     # Record predictions for each left-out participant
#     for i, idx in enumerate(test_index):
#         predictions[idx].append(y_pred[i])
#     print(f"Fold {fold} complete")
#
# # For each participant, compute the median prediction across the 10 times they were left out.
# y_pred_median = np.array([np.median(predictions[i]) for i in range(n_samples)])
#
# # Evaluate the aggregated predictions
# rmse = np.sqrt(mean_squared_error(y, y_pred_median))
# r2 = r2_score(y, y_pred_median)
# y_flat = np.ravel(y)
# y_pred_median_flat = np.ravel(y_pred_median)
# corrcoef = np.corrcoef(y_flat, y_pred_median_flat)[0, 1]
#
# print("\nAggregated Cross-Validated Performance:")
# print(f"RMSE: {rmse:.4f}")
# print(f"R^2: {r2:.4f}")
# print(f"Correlation Coefficient: {corrcoef:.4f}")
#
# # Save metrics to a CSV file
# metrics_dict = {
#     "RMSE": [rmse],
#     "R2": [r2],
#     "CorrelationCoefficient": [corrcoef]
# }
# df_metrics = pd.DataFrame(metrics_dict)
# csv_filename = os.path.join(output_path, "metrics.csv")
# df_metrics.to_csv(csv_filename, index=False)
# print(f"Metrics saved to {csv_filename}")
#
# # Create the plot of predicted vs. true symptom change values
# plt.figure()
# plt.scatter(y, y_pred_median, alpha=0.7)
# plt.xlabel("True Symptom Change")
# plt.ylabel("Predicted Symptom Change")
# plt.title("10×10 Cross-Validated Predictions (Median Aggregation)")
#
# # Plot a best-fit line for visualization using flattened arrays
# p = np.polyfit(y_flat, y_pred_median_flat, 1)
# y_line = np.polyval(p, y_flat)
# plt.plot(y_flat, y_line, 'r-', linewidth=1.5)
#
# # Save the plot to the output path
# plot_filename = os.path.join(output_path, "predicted_vs_true_plot.png")
# plt.savefig(plot_filename, dpi=300)
# print(f"Plot saved to {plot_filename}")
# plt.show()

###############################################################################################################
# Other algorithms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score

# Alternative regression models
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet

# -----------------------------------------------------------------------------
# Assume X, y, n_samples, and ComBat_transformer (an instance of your
# NeuroHarmonizeTransformer) are already defined.
#
# For example:
# X = pd.read_csv("your_features.csv")
# y = X.pop("target").values  # or however you define y
# n_samples = X.shape[0]
#
# And your transformer might be set up like:
# feature_number = ...  # number of radiomics features to harmonize
# ComBat_transformer = NeuroHarmonizeTransformer(feature_number=feature_number)
# -----------------------------------------------------------------------------

# Define a dictionary of models to evaluate.
models = {
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GaussianProcess": GaussianProcessRegressor(),
    "KernelRidge": KernelRidge(kernel='rbf', alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
}

# Prepare cross-validation settings: 10 folds repeated 10 times.
rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

# Dictionary to store performance metrics for each model.
results = {}

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    # Create the pipeline: apply harmonization, scale features, then regression.
    pipeline_model = Pipeline([
        ("Combat", ComBat_transformer),
        ("scaler", RobustScaler()),
        (model_name, model)
    ])

    # Dictionary to collect predictions for each sample.
    predictions = {i: [] for i in range(n_samples)}

    fold = 0
    for train_index, test_index in rkf.split(X):
        fold += 1
        # Fit the pipeline on the training set.
        pipeline_model.fit(X.iloc[train_index], y[train_index])
        # Predict on the test set.
        y_pred = pipeline_model.predict(X.iloc[test_index])
        # Record predictions for each test sample.
        for i, idx in enumerate(test_index):
            predictions[idx].append(y_pred[i])
        print(f"{model_name} - Fold {fold} complete")

    # Aggregate predictions (here using the median across folds).
    y_pred_median = np.array([np.median(predictions[i]) for i in range(n_samples)])

    # Compute evaluation metrics.
    rmse = np.sqrt(mean_squared_error(y, y_pred_median))
    r2 = r2_score(y, y_pred_median)
    corrcoef = np.corrcoef(np.ravel(y), np.ravel(y_pred_median))[0, 1]

    # Store the results.
    results[model_name] = {"RMSE": rmse, "R2": r2, "Correlation": corrcoef}

    print(f"\nAggregated Performance for {model_name}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"Correlation Coefficient: {corrcoef:.4f}")

    # Optionally, plot predicted vs. true values for this model.
    plt.figure()
    plt.scatter(y, y_pred_median, alpha=0.7)
    plt.xlabel("True Symptom Change")
    plt.ylabel("Predicted Symptom Change")
    plt.title(f"{model_name}: 10×10 Cross-Validated Predictions (Median Aggregation)")
    # Best-fit line
    p = np.polyfit(np.ravel(y), np.ravel(y_pred_median), 1)
    y_line = np.polyval(p, np.ravel(y))
    plt.plot(np.ravel(y), y_line, 'r-', linewidth=1.5)
    plt.show()

# Convert the results to a DataFrame for easy comparison.
results_df = pd.DataFrame(results).T
print("\nSummary of Model Performance:")
print(results_df)
#########################################################################################################################
# With feature selection
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shap

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RepeatedKFold, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from scipy.stats import pearsonr
from skrvm import RVR

# --- Assume that X (a DataFrame), y (a Series/array), output_path,
# and ComBat_transformer (an instance of NeuroHarmonizeTransformer) are defined elsewhere ---

# Define models and a simplified hyperparameter grid.

# -----------------------
# Define models and hyperparameter grids.
# -----------------------
models = {
    "SVR": SVR(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GaussianProcess": GaussianProcessRegressor(),
    "KernelRidge": KernelRidge(),
    "ElasticNet": ElasticNet(random_state=42),
    "RVM": RVR()
}

# Update the param_grids dictionary with hyperparameters for RVM.
# Note: The available hyperparameters for RVR may depend on the kernel choice.
param_grids = {
    "SVR": {
        "feature_selection__k": [5, 10, "all"],
        "SVR__C": [0.1, 1.0, 10.0],
        "SVR__epsilon": [0.01, 0.1, 0.2],
    },
    "RandomForest": {
        "feature_selection__k": [5, 10, "all"],
        "RandomForest__n_estimators": [50, 100],
        "RandomForest__max_depth": [None, 5, 10]
    },
    "GaussianProcess": {
        "feature_selection__k": [5, 10, "all"],
        "GaussianProcess__alpha": [1e-10, 1e-2, 1e-1]
    },
    "KernelRidge": {
        "feature_selection__k": [5, 10, "all"],
        "KernelRidge__alpha": [0.1, 1.0, 10.0],
    },
    "ElasticNet": {
        "feature_selection__k": [5, 10, "all"],
        "ElasticNet__alpha": [0.1, 1.0, 10.0],
        "ElasticNet__l1_ratio": [0.1, 0.5, 0.9]
    },
    "RVM": {
        "feature_selection__k": [5, 10, "all"],
        "RVM__kernel": ['linear', 'rbf'],  # Example kernel choices
        "RVM__gamma": [0.1, 1.0, 10.0]      # Applicable if using an RBF kernel
    }
}
# -----------------------
# Setup cross-validation and directories.
# -----------------------
outer_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
shap_plots_path = os.path.join(output_path, "shap_plots")
os.makedirs(shap_plots_path, exist_ok=True)

results = {}  # To store performance metrics for each model

# -----------------------
# Loop over models.
# -----------------------
for model_name, model in models.items():
    print(f"\nProcessing model: {model_name}")
    fold_idx = 0
    shap_values_list = []  # To store SHAP values for each outer fold

    # Initialize dictionary to collect predictions from each outer fold.
    fold_predictions = {i: [] for i in range(X.shape[0])}

    # Outer CV loop.
    for train_idx, test_idx in outer_cv.split(X):
        fold_idx += 1
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        # Build pipeline: harmonization (Combat), scaling, feature selection, then regression.
        pipeline = Pipeline([
            ("Combat", ComBat_transformer),
            ("scaler", RobustScaler()),
            ("feature_selection", SelectKBest(score_func=f_regression)),
            (model_name, model)
        ])

        # Inner CV for hyperparameter tuning.
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        grid = GridSearchCV(pipeline, param_grid=param_grids[model_name],
                            cv=inner_cv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"Fold {fold_idx} complete")
        # --- Collect predictions for performance evaluation ---
        y_pred_fold = best_model.predict(X_test)
        for j, idx in enumerate(test_idx):
            fold_predictions[idx].append(y_pred_fold[j])

    #     # --- Pre-transform X_test using all steps EXCEPT the final estimator ---
    #     X_test_transformed = best_model[:-1].transform(X_test)
    #     try:
    #         feature_names = list(best_model.named_steps['feature_selection'].get_feature_names_out())
    #     except Exception:
    #         feature_names = [f"f{i}" for i in range(X_test_transformed.shape[1])]
    #
    #     X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    #
    #     # Prepare background data from training set.
    #     background = X_train.sample(n=min(100, len(X_train)), random_state=42)
    #     background_transformed = best_model[:-1].transform(background)
    #
    #     # Define prediction function that takes pre-transformed data.
    #     def predict_transformed(data):
    #         return best_model.named_steps[model_name].predict(data)
    #
    #     explainer = shap.KernelExplainer(predict_transformed, background_transformed)
    #     shap_vals = explainer.shap_values(X_test_transformed)
    #
    #     # Store SHAP values with a fold indicator.
    #     shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    #     shap_df["fold"] = fold_idx
    #     shap_values_list.append(shap_df)
    #
    # # --- Aggregate SHAP values across folds ---
    # all_shap = pd.concat(shap_values_list, ignore_index=True)
    # all_shap_features = all_shap.drop(columns=["fold"])
    #
    # # Reconstruct the original feature names used in the model.
    # # Here, feature_names was loaded from your CSV and has one entry per original column.
    # # Since your Combat transformer drops the last 4 columns, we take the first len(feature_names)-4 entries.
    # original_features = np.array(feature_names)[:-4]
    #
    # # Get the support mask from the feature_selection step.
    # mask = best_model.named_steps["feature_selection"].get_support()
    #
    # # If the mask length is larger than the number of features after Combat,
    # # slice the mask accordingly.
    # if len(mask) > len(original_features):
    #     mask = mask[:len(original_features)]
    #
    # # Use the boolean mask to select the names of the features that made it through feature selection.
    # selected_feature_names = original_features[mask]
    #
    # # Update the aggregated SHAP DataFrame columns to use these selected feature names.
    # all_shap_features.columns = selected_feature_names
    #
    # # Use SHAP's summary_plot (bar mode) to display only the top 20 features.
    # plt.figure()
    # shap.summary_plot(
    #     all_shap_features.to_numpy(),  # SHAP values as a NumPy array.
    #     feature_names=selected_feature_names,  # Mapped feature names.
    #     plot_type="bar",  # Bar plot mode.
    #     max_display=20,  # Display top 20 features.
    #     show=False
    # )
    # aggregate_shap_filename = os.path.join(shap_plots_path, f"{model_name}_aggregate_shap_top20.png")
    # plt.title(f"Top 20 Average Absolute SHAP Values for {model_name}")
    # plt.savefig(aggregate_shap_filename, bbox_inches='tight')
    # plt.close()
    # print(f"Aggregate top 20 SHAP plot saved for {model_name} at {aggregate_shap_filename}")

    # --- Performance Evaluation ---
    # Aggregate predictions across all outer folds (median for each sample).
    y_pred_median = np.array([np.median(fold_predictions[i]) for i in range(X.shape[0])])
    rmse = np.sqrt(mean_squared_error(y, y_pred_median))
    r2 = r2_score(y, y_pred_median)
    corr, p_value = pearsonr(np.ravel(y), np.ravel(y_pred_median))
    results[model_name] = {"RMSE": rmse, "R2": r2, "Correlation": corr, "p_value": p_value}
    print(f"\n{model_name} performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2: {r2:.4f}")
    print(f"  Correlation: {corr:.4f} (p-value: {p_value:.4e})")

    # Optionally, plot predicted vs. true values for this model.
    plt.figure()
    plt.scatter(y, y_pred_median, alpha=0.7)
    plt.xlabel("True Symptom Change")
    plt.ylabel("Predicted Symptom Change")
    plt.title(f"{model_name}: 10×10 Cross-Validated Predictions (Median Aggregation)")
    # Best-fit line
    p = np.polyfit(np.ravel(y), np.ravel(y_pred_median), 1)
    y_line = np.polyval(p, np.ravel(y))
    plt.plot(np.ravel(y), y_line, 'r-', linewidth=1.5)
    # Add correlation coefficient and p-value in the upper left corner of the plot
    plt.gca().text(0.05, 0.95, f"r: {corr:.2f}\np: {p_value:.2f}",
                   transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    scatter_plot_filename = os.path.join(shap_plots_path, f"{model_name}_scatter_plot.png")
    plt.title(f"{model_name}: 10×10 Cross-Validated Predictions (Median Aggregation)")
    plt.savefig(scatter_plot_filename, bbox_inches='tight')
    plt.show()

# -----------------------
# Save and print summary performance results.
# -----------------------
results_df = pd.DataFrame(results).T
print("\nSummary of Model Performance:")
print(results_df)
results_df.to_csv(os.path.join(output_path, "model_performance.csv"))

####################################################################################################################
# Debugging
# Checking for each step's shape
# pipeline1 = pipeline.fit(X,y)
# X1 = pipeline.steps[0][1].fit_transform(X)
# print("X shape in step 1: ", X1.shape)
# X2 = pipeline1.steps[1][1].transform(X1)
# print("X shape in step 2: ", X2.shape)
# X3 = pipeline1.steps[2][1].transform(X2)
# print("X shape in step 3: ", X3.shape)
######################################################################################################################
# Without ComBat
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def RVM(X, y, stoper=1e-3, maxIts=10000, maxAlpha=1e12):
    """
    Relevance Vector Machine (RVM) implementation via Sparse Bayesian Learning.

    Parameters:
        X       : (N x D) design matrix (first column should be ones for bias)
        y       : (N,) target vector
        stoper  : convergence criterion on weight change
        maxIts  : maximum number of iterations
        maxAlpha: threshold for considering a weight as pruned

    Returns:
        w     : weight vector (D,)
        alpha : ARD parameters for non-bias weights (length D-1)
        beta  : noise precision
    """
    N, D = X.shape
    w = np.zeros(D)
    alpha = np.ones(D - 1)  # for non-bias features
    beta = 1.0
    biasalpha = 1.0  # prior inverse variance for bias term

    Xt = X.T @ y
    act = alpha < maxAlpha  # Boolean mask for non-bias weights (length D-1)
    idxact = np.empty(D, dtype=bool)
    idxact[0] = True
    idxact[1:] = act
    w[~idxact] = 0

    for itr in range(maxIts):
        prew = w.copy()
        X_act = X[:, idxact]
        alpha_act = alpha[act]
        diag_params = np.diag(np.concatenate(([biasalpha], alpha_act)))
        Sigma = np.linalg.inv(beta * (X_act.T @ X_act) + diag_params)
        w_active = beta * (Sigma @ Xt[idxact])
        w[idxact] = w_active

        residuals = y - X_act @ w_active
        Er = np.sum(residuals ** 2)
        diagSigma = np.diag(Sigma)
        gamma = 1 - np.concatenate(([biasalpha], alpha_act)) * diagSigma

        eps = 1e-10  # avoid division by zero
        active_nonbias_w = w[1:][act]
        gamma_nonbias = gamma[1:]
        alpha[act] = gamma_nonbias / (active_nonbias_w ** 2 + eps)
        beta = (N - np.sum(gamma)) / (Er + eps)

        act = alpha < maxAlpha
        idxact = np.empty(D, dtype=bool)
        idxact[0] = True
        idxact[1:] = act
        w[~idxact] = 0

        if np.linalg.norm(w - prew) < stoper:
            break

    print(f'Successful Convergence at iteration #{itr + 1}')
    return w, alpha, beta


def residualize_features_df(df):
    """
    Residualize imaging features by regressing out imaging site effects.

    Parameters:
        df : Pandas DataFrame with shape (N x 414), where the first 413 columns
             are imaging features and the 414th column is 'SITE'.

    Returns:
        X_resid : N x 413 numpy array of residualized imaging features.
    """
    # Convert imaging features to float explicitly.
    X_imaging = df.iloc[:, :-1].astype(float).values  # (N x 413)
    sites = df.iloc[:, -1].values  # (N,)

    N, D = X_imaging.shape
    # One-hot encode site labels (drop first to avoid collinearity)
    df_sites = pd.get_dummies(sites, drop_first=True)
    X_resid = np.zeros_like(X_imaging, dtype=float)

    for d in range(D):
        y_feature = X_imaging[:, d]
        # Design matrix: intercept + site dummies (ensure numeric type)
        X_design = np.column_stack((np.ones(N), df_sites.values.astype(float)))
        beta_ls, _, _, _ = np.linalg.lstsq(X_design, y_feature, rcond=None)
        y_pred = X_design @ beta_ls
        X_resid[:, d] = y_feature - y_pred
    return X_resid


def main_pipeline(X, y, output_path):
    """
    Main analysis pipeline using the provided X and y.

    Parameters:
        X : pandas DataFrame with 414 columns (first 413 are imaging features,
            the 414th column is 'SITE')
        y : target vector (e.g., symptom change scores) as a NumPy array
        output_path : string path to save output files (plot and metrics table)
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # -----------------------------
    # 1. Use Provided Data
    # -----------------------------
    X_df = X.copy()

    # -----------------------------
    # 2. Residualize the Imaging Features
    # -----------------------------
    X_resid = residualize_features_df(X_df)
    N = X_df.shape[0]
    # For the RVM, add a bias term (a column of ones)
    X_resid_bias = np.hstack((np.ones((N, 1)), X_resid))

    # -----------------------------
    # 3. Repeated 10-Fold Cross-Validation with RVM
    # -----------------------------
    n_reps = 10
    n_folds = 10
    preds = np.empty((N, n_reps))
    preds.fill(np.nan)

    for rep in range(n_reps):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rep)
        for train_index, test_index in kf.split(X_resid_bias):
            X_train = X_resid_bias[train_index]
            y_train = y[train_index]
            X_test = X_resid_bias[test_index]
            w, _, _ = RVM(X_train, y_train)
            y_pred = X_test @ w
            preds[test_index, rep] = y_pred

    # Final prediction for each subject is the median across repetitions.
    final_pred = np.median(preds, axis=1)

    # -----------------------------
    # 4. Evaluate Prediction Performance
    # -----------------------------
    r, p_val = pearsonr(final_pred, y)
    R2 = 1 - np.sum((y - final_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    RMSE = np.sqrt(np.mean((y - final_pred) ** 2))
    print(f'Primary Treatment Arm: Pearson r = {r:.3f}, p = {p_val:.3f}')
    print(f'R² = {R2:.3f}, RMSE = {RMSE:.3f}')

    # -----------------------------
    # 5. Plot y_true vs y_pred Scatter Plot and Save the Plot
    # -----------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(y, final_pred, alpha=0.7, label='Data points')
    min_val = min(np.min(y), np.min(final_pred))
    max_val = max(np.max(y), np.max(final_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title(f"Scatter Plot: R² = {R2:.3f}, RMSE = {RMSE:.3f}")
    plt.legend()
    plt.grid(True)
    # Save the plot to output_path
    plot_filepath = os.path.join(output_path, "scatter_plot.png")
    plt.savefig(plot_filepath)
    plt.close()
    print(f"Scatter plot saved to {plot_filepath}")

    # -----------------------------
    # 6. Permutation Testing (1,000 permutations)
    # -----------------------------
    # n_perm = 1000
    # perm_corrs = np.zeros(n_perm)
    # for i in range(n_perm):
    #     y_perm = np.random.permutation(y)
    #     perm_preds = np.empty((N, n_reps))
    #     perm_preds.fill(np.nan)
    #     for rep in range(n_reps):
    #         kf = KFold(n_splits=n_folds, shuffle=True, random_state=rep)
    #         for train_index, test_index in kf.split(X_resid_bias):
    #             X_train = X_resid_bias[train_index]
    #             y_train_perm = y_perm[train_index]
    #             X_test = X_resid_bias[test_index]
    #             w, _, _ = RVM(X_train, y_train_perm)
    #             y_pred = X_test @ w
    #             perm_preds[test_index, rep] = y_pred
    #     final_perm_pred = np.median(perm_preds, axis=1)
    #     perm_corrs[i], _ = pearsonr(final_perm_pred, y_perm)
    # p_perm = np.mean(perm_corrs >= r)
    # print(f'Permutation test p-value: {p_perm:.3f}')

    # -----------------------------
    # 7. Save Performance Metrics as a Table
    # -----------------------------
    metrics = {
        # "Metric": ["Pearson r", "p-value", "R²", "RMSE", "Permutation p-value"],
        # "Value": [r, p_val, R2, RMSE, p_perm]
        "Metric": ["Pearson r", "p-value", "R²", "RMSE"],
        "Value": [r, p_val, R2, RMSE]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_filepath = os.path.join(output_path, "performance_metrics.csv")
    metrics_df.to_csv(metrics_filepath, index=False)
    print(f"Performance metrics saved to {metrics_filepath}")


if __name__ == '__main__':
    # Example: assume you already have X and y defined as input.
    # X must be a DataFrame with 414 columns (first 413 are imaging features, last is 'SITE')
    # y must be a NumPy array or convertible to one.
    #
    # For instance:
    # X = pd.read_csv('your_imaging_features.csv')  # the CSV should have a 'SITE' column as the last column
    # y = np.load('your_target_scores.npy')
    #
    # Specify the output path where results will be saved:
    output_path = "output_results"
    # Then call main_pipeline with your data:
    main_pipeline(X, y, output_path)
############################################################################################################################
# Without ComBat: tested with many algorithms
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RepeatedKFold, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from scipy.stats import pearsonr
from skrvm import RVR

# -----------------------
# Define a transformer for residualizing imaging features
# -----------------------
class ResidualizeFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that residualizes imaging features by regressing out
    imaging site effects. Expects X as a pandas DataFrame with 414 columns:
    the first 413 columns are imaging features and the 414th column is 'SITE'.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No fitting needed for residualization
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")
        return residualize_features_df(X)

def residualize_features_df(df):
    """
    Residualize imaging features by regressing out imaging site effects.

    Parameters:
        df : Pandas DataFrame with shape (N x 414), where the first 413 columns
             are imaging features and the 414th column is 'SITE'.

    Returns:
        X_resid : N x 413 numpy array of residualized imaging features.
    """
    # Convert imaging features to float explicitly.
    X_imaging = df.iloc[:, :-1].astype(float).values  # (N x 413)
    sites = df.iloc[:, -1].values  # (N,)

    N, D = X_imaging.shape
    # One-hot encode site labels (drop first to avoid collinearity)
    df_sites = pd.get_dummies(sites, drop_first=True)
    X_resid = np.zeros_like(X_imaging, dtype=float)

    for d in range(D):
        y_feature = X_imaging[:, d]
        # Design matrix: intercept + site dummies (ensure numeric type)
        X_design = np.column_stack((np.ones(N), df_sites.values.astype(float)))
        beta_ls, _, _, _ = np.linalg.lstsq(X_design, y_feature, rcond=None)
        y_pred = X_design @ beta_ls
        X_resid[:, d] = y_feature - y_pred
    return X_resid

# -----------------------
# Define models and hyperparameter grids.
# -----------------------
models = {
    "SVR": SVR(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GaussianProcess": GaussianProcessRegressor(),
    "KernelRidge": KernelRidge(),
    "ElasticNet": ElasticNet(random_state=42),
    "RVM": RVR()
}

param_grids = {
    "SVR": {
        "feature_selection__k": [5, 10, "all"],
        "SVR__C": [0.1, 1.0, 10.0],
        "SVR__epsilon": [0.01, 0.1, 0.2],
    },
    "RandomForest": {
        "feature_selection__k": [5, 10, "all"],
        "RandomForest__n_estimators": [50, 100],
        "RandomForest__max_depth": [None, 5, 10]
    },
    "GaussianProcess": {
        "feature_selection__k": [5, 10, "all"],
        "GaussianProcess__alpha": [1e-10, 1e-2, 1e-1]
    },
    "KernelRidge": {
        "feature_selection__k": [5, 10, "all"],
        "KernelRidge__alpha": [0.1, 1.0, 10.0],
    },
    "ElasticNet": {
        "feature_selection__k": [5, 10, "all"],
        "ElasticNet__alpha": [0.1, 1.0, 10.0],
        "ElasticNet__l1_ratio": [0.1, 0.5, 0.9]
    },
    "RVM": {
        "feature_selection__k": [5, 10, "all"],
        "RVM__kernel": ['linear', 'rbf'],  # Example kernel choices
        "RVM__gamma": [0.1, 1.0, 10.0]      # Applicable if using an RBF kernel
    }
}

# -----------------------
# Setup cross-validation and directories.
# -----------------------
outer_cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
shap_plots_path = os.path.join(output_path, "shap_plots")
os.makedirs(shap_plots_path, exist_ok=True)

results = {}  # To store performance metrics for each model

# -----------------------
# Loop over models.
# -----------------------
for model_name, model in models.items():
    print(f"\nProcessing model: {model_name}")
    fold_idx = 0
    shap_values_list = []  # To store SHAP values for each outer fold

    # Initialize dictionary to collect predictions from each outer fold.
    fold_predictions = {i: [] for i in range(X.shape[0])}

    # Outer CV loop.
    for train_idx, test_idx in outer_cv.split(X):
        fold_idx += 1
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        # Build pipeline: residualization, scaling, feature selection, then regression.
        pipeline = Pipeline([
            ("Residualize", ResidualizeFeaturesTransformer()),
            ("scaler", RobustScaler()),
            ("feature_selection", SelectKBest(score_func=f_regression)),
            (model_name, model)
        ])

        # Inner CV for hyperparameter tuning.
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        grid = GridSearchCV(pipeline, param_grid=param_grids[model_name],
                            cv=inner_cv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"Fold {fold_idx} complete")
        # --- Collect predictions for performance evaluation ---
        y_pred_fold = best_model.predict(X_test)
        for j, idx in enumerate(test_idx):
            fold_predictions[idx].append(y_pred_fold[j])

        # (Optional: SHAP explanation code is commented out)

    # --- Performance Evaluation ---
    # Aggregate predictions across all outer folds (median for each sample).
    y_pred_median = np.array([np.median(fold_predictions[i]) for i in range(X.shape[0])])
    rmse = np.sqrt(mean_squared_error(y, y_pred_median))
    r2 = r2_score(y, y_pred_median)
    corr, p_value = pearsonr(np.ravel(y), np.ravel(y_pred_median))
    results[model_name] = {"RMSE": rmse, "R2": r2, "Correlation": corr, "p_value": p_value}
    print(f"\n{model_name} performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2: {r2:.4f}")
    print(f"  Correlation: {corr:.4f} (p-value: {p_value:.4e})")

    # Optionally, plot predicted vs. true values for this model.
    plt.figure()
    plt.scatter(y, y_pred_median, alpha=0.7)
    plt.xlabel("True Symptom Change")
    plt.ylabel("Predicted Symptom Change")
    plt.title(f"{model_name}: 10×10 Cross-Validated Predictions (Median Aggregation)")
    # Best-fit line
    p = np.polyfit(np.ravel(y), np.ravel(y_pred_median), 1)
    y_line = np.polyval(p, np.ravel(y))
    plt.plot(np.ravel(y), y_line, 'r-', linewidth=1.5)
    # Add correlation coefficient and p-value in the upper left corner of the plot
    plt.gca().text(0.05, 0.95, f"r: {corr:.2f}\np: {p_value:.2f}",
                   transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    scatter_plot_filename = os.path.join(shap_plots_path, f"{model_name}_scatter_plot.png")
    plt.title(f"{model_name}: 10×10 Cross-Validated Predictions (Median Aggregation)")
    plt.savefig(scatter_plot_filename, bbox_inches='tight')
    plt.show()

# -----------------------
# Save and print summary performance results.
# -----------------------
results_df = pd.DataFrame(results).T
print("\nSummary of Model Performance:")
print(results_df)
results_df.to_csv(os.path.join(output_path, "model_performance.csv"))
