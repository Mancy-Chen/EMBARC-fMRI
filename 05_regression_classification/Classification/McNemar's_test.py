# Mancy Chen 16/03/2026
# Compare among groups
# McNemar's test with exact folder mapping
import os
import itertools
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# =========================================================
# Settings
# =========================================================
TASK = "Response"   # choose: "Remission" or "Response"

BASE_ROOT = "/data/projects/EMBARC/data/06_BART_regression/Output/Classification_plot"
BASE = os.path.join(BASE_ROOT, TASK)

MODEL_DIRS = {
    "Tier1":  os.path.join(BASE, "Tier1"),
    "Tier2a": os.path.join(BASE, "Tier2a"),
    "Tier2b": os.path.join(BASE, "Tier2b"),
}

# Exact folder mapping only
# Edit the Response section if its numbering differs from Remission.
FOLDER_MAPS = {
    "Remission": {
        "ses-1_SER": {
            "Tier1":  "06_ses-1_SER_save_feature_and_model",
            "Tier2a": "05_ses-1_SER_save_feature_and_model",
            "Tier2b": "05_ses-1_SER_save_feature_and_model",
        },
        "ses-1_PLA": {
            "Tier1":  "07_ses-1_PLA_save_feature_and_model",
            "Tier2a": "06_ses-1_PLA_save_feature_and_model",
            "Tier2b": "06_ses-1_PLA_save_feature_and_model",
        },
        "ses-2_SER": {
            "Tier1":  "08_ses-2_SER_save_feature_and_model",
            "Tier2a": "07_ses-2_SER_save_feature_and_model",
            "Tier2b": "07_ses-2_SER_save_feature_and_model",
        },
        "ses-2_PLA": {
            "Tier1":  "09_ses-2_PLA_save_feature_and_model",
            "Tier2a": "08_ses-2_PLA_save_feature_and_model",
            "Tier2b": "08_ses-2_PLA_save_feature_and_model",
        },
    },

    # Change these names if your Response folder numbering is different.
    "Response": {
        "ses-1_SER": {
            "Tier1":  "09_ses-1_SER_save_feature_and_model",
            "Tier2a": "05_ses-1_SER_save_feature_and_model",
            "Tier2b": "05_ses-1_SER_save_feature_and_model",
        },
        "ses-1_PLA": {
            "Tier1":  "10_ses-1_PLA_save_feature_and_model",
            "Tier2a": "06_ses-1_PLA_save_feature_and_model",
            "Tier2b": "06_ses-1_PLA_save_feature_and_model",
        },
        "ses-2_SER": {
            "Tier1":  "11_ses-2_SER_save_feature_and_model",
            "Tier2a": "07_ses-2_SER_save_feature_and_model",
            "Tier2b": "07_ses-2_SER_save_feature_and_model",
        },
        "ses-2_PLA": {
            "Tier1":  "12_ses-2_PLA_save_feature_and_model",
            "Tier2a": "08_ses-2_PLA_save_feature_and_model",
            "Tier2b": "08_ses-2_PLA_save_feature_and_model",
        },
    }
}

FOLDER_MAP = FOLDER_MAPS[TASK]

ALPHA = 0.05
N_COMPARISONS = 3
ALPHA_BONF = ALPHA / N_COMPARISONS


# =========================================================
# Helper functions
# =========================================================
def get_exact_folder(model_name, condition_name):
    folder_name = FOLDER_MAP[condition_name][model_name]
    folder_path = os.path.join(MODEL_DIRS[model_name], folder_name)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Expected folder not found:\n{folder_path}")

    return folder_path


def load_npy_concat_1d(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.concatenate([np.asarray(x).ravel() for x in arr.ravel()])
    return np.asarray(arr).ravel()


def load_true_and_pred(folder):
    y_true_path = os.path.join(folder, "y_true_list.npy")
    y_pred_path = os.path.join(folder, "y_pred_list.npy")

    if not os.path.exists(y_true_path):
        raise FileNotFoundError(f"Missing file: {y_true_path}")
    if not os.path.exists(y_pred_path):
        raise FileNotFoundError(f"Missing file: {y_pred_path}")

    y_true = load_npy_concat_1d(y_true_path)
    y_pred = load_npy_concat_1d(y_pred_path)

    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {y_true.shape} in {folder}")
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred must be 1-D, got shape {y_pred.shape} in {folder}")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in {folder}: y_true {y_true.shape}, y_pred {y_pred.shape}"
        )

    return y_true, y_pred, y_true_path, y_pred_path


def compute_accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def run_mcnemar(y_true, y_a, y_b, exact=False, correction=True):
    """
    Compare model A vs model B on the same true labels using McNemar's test.
    """
    correct_a = (y_a == y_true)
    correct_b = (y_b == y_true)

    b = int(np.sum(correct_a & (~correct_b)))   # A correct, B wrong
    c = int(np.sum((~correct_a) & correct_b))   # A wrong, B correct

    both_correct = int(np.sum(correct_a & correct_b))
    both_wrong = int(np.sum((~correct_a) & (~correct_b)))

    table = [
        [both_correct, b],
        [c, both_wrong]
    ]

    result = mcnemar(table, exact=exact, correction=correction)

    return {
        "both_correct": both_correct,
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "both_wrong": both_wrong,
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "table": table,
        "discordant_total": b + c,
    }


# =========================================================
# Main batch analysis
# =========================================================
summary_rows = []
pairwise_rows = []

print(f"TASK: {TASK}")
print(f"BASE: {BASE}")

for condition_name in ["ses-1_SER", "ses-1_PLA", "ses-2_SER", "ses-2_PLA"]:
    ses, drug = condition_name.split("_")
    print("\n" + "=" * 70)
    print(f"Condition: {condition_name}")

    condition_data = {}
    y_true_ref = None

    for model_name in ["Tier1", "Tier2a", "Tier2b"]:
        folder = get_exact_folder(model_name, condition_name)
        print(f"{model_name}: {folder}")

        y_true, y_pred, y_true_path, y_pred_path = load_true_and_pred(folder)

        if y_true_ref is None:
            y_true_ref = y_true
        else:
            if y_true.shape != y_true_ref.shape:
                raise ValueError(
                    f"Across-model y_true shape mismatch for {condition_name}: "
                    f"{model_name} has {y_true.shape}, expected {y_true_ref.shape}"
                )
            if not np.array_equal(y_true, y_true_ref):
                print(
                    f"[Warning] y_true in {model_name} is not identical to "
                    f"reference y_true for {condition_name}"
                )

        condition_data[model_name] = {
            "folder": folder,
            "folder_name": os.path.basename(folder),
            "y_true_path": y_true_path,
            "y_pred_path": y_pred_path,
            "y_true": y_true,
            "y_pred": y_pred,
            "accuracy": compute_accuracy(y_true, y_pred),
        }

    summary_rows.append({
        "task": TASK,
        "condition": condition_name,
        "session": ses,
        "drug": drug,
        "n": len(y_true_ref),

        "Tier1_accuracy": condition_data["Tier1"]["accuracy"],
        "Tier2a_accuracy": condition_data["Tier2a"]["accuracy"],
        "Tier2b_accuracy": condition_data["Tier2b"]["accuracy"],

        "Tier1_folder": condition_data["Tier1"]["folder"],
        "Tier2a_folder": condition_data["Tier2a"]["folder"],
        "Tier2b_folder": condition_data["Tier2b"]["folder"],

        "Tier1_y_true_path": condition_data["Tier1"]["y_true_path"],
        "Tier1_y_pred_path": condition_data["Tier1"]["y_pred_path"],
        "Tier2a_y_true_path": condition_data["Tier2a"]["y_true_path"],
        "Tier2a_y_pred_path": condition_data["Tier2a"]["y_pred_path"],
        "Tier2b_y_true_path": condition_data["Tier2b"]["y_true_path"],
        "Tier2b_y_pred_path": condition_data["Tier2b"]["y_pred_path"],
    })

    for a, b in itertools.combinations(["Tier1", "Tier2a", "Tier2b"], 2):
        # asymptotic McNemar
        result_asym = run_mcnemar(
            y_true_ref,
            condition_data[a]["y_pred"],
            condition_data[b]["y_pred"],
            exact=False,
            correction=True
        )

        # exact McNemar
        result_exact = run_mcnemar(
            y_true_ref,
            condition_data[a]["y_pred"],
            condition_data[b]["y_pred"],
            exact=True,
            correction=False
        )

        print(f"\n=== {a} vs {b} ===")
        print(f"Folder A           : {condition_data[a]['folder']}")
        print(f"Folder B           : {condition_data[b]['folder']}")
        print(f"both correct       : {result_asym['both_correct']}")
        print(f"{a} correct only   : {result_asym['a_correct_b_wrong']}")
        print(f"{b} correct only   : {result_asym['a_wrong_b_correct']}")
        print(f"both wrong         : {result_asym['both_wrong']}")
        print(f"McNemar chi2       = {result_asym['statistic']:.4f}, p = {result_asym['pvalue']:.6f}")
        print(f"McNemar exact p    = {result_exact['pvalue']:.6f}")

        pairwise_rows.append({
            "task": TASK,
            "condition": condition_name,
            "session": ses,
            "drug": drug,
            "comparison": f"{a} vs {b}",

            "model_a": a,
            "model_b": b,

            "folder_a": condition_data[a]["folder"],
            "folder_b": condition_data[b]["folder"],
            "y_true_path_a": condition_data[a]["y_true_path"],
            "y_pred_path_a": condition_data[a]["y_pred_path"],
            "y_true_path_b": condition_data[b]["y_true_path"],
            "y_pred_path_b": condition_data[b]["y_pred_path"],

            "acc_a": condition_data[a]["accuracy"],
            "acc_b": condition_data[b]["accuracy"],

            "both_correct": result_asym["both_correct"],
            "a_correct_b_wrong": result_asym["a_correct_b_wrong"],
            "a_wrong_b_correct": result_asym["a_wrong_b_correct"],
            "both_wrong": result_asym["both_wrong"],
            "discordant_total": result_asym["discordant_total"],

            "mcnemar_statistic_asymptotic": result_asym["statistic"],
            "mcnemar_pvalue_asymptotic": result_asym["pvalue"],
            "mcnemar_pvalue_exact": result_exact["pvalue"],

            "alpha_bonferroni": ALPHA_BONF,
            "significant_bonferroni_asymptotic": result_asym["pvalue"] < ALPHA_BONF,
            "significant_bonferroni_exact": result_exact["pvalue"] < ALPHA_BONF,
        })

# =========================================================
# Save results
# =========================================================
summary_df = pd.DataFrame(summary_rows)
pairwise_df = pd.DataFrame(pairwise_rows)

summary_csv = os.path.join(
    BASE,
    f"{TASK.lower()}_classification_accuracy_summary_exact_mapping.csv"
)
pairwise_csv = os.path.join(
    BASE,
    f"{TASK.lower()}_mcnemar_pairwise_results_exact_mapping.csv"
)

summary_df.to_csv(summary_csv, index=False)
pairwise_df.to_csv(pairwise_csv, index=False)

print("\n" + "=" * 70)
print("Done.")
print(f"Saved summary:  {summary_csv}")
print(f"Saved pairwise: {pairwise_csv}")

print("\nSummary table:")
print(summary_df)

print("\nPairwise McNemar results:")
print(pairwise_df)