import os
import glob
import itertools
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

# =========================================================
# Settings
# =========================================================
BASE = "/.../EMBARC/data/06_BART_regression/Output/Regression_plot"

MODEL_DIRS = {
    "Tier_1": os.path.join(BASE, "Tier_1"),
    "Tier_2a": os.path.join(BASE, "Tier_2a"),
    "Tier_2b": os.path.join(BASE, "Tier_2b"),
}

CONDITIONS = [
    ("ses-1", "SER"),
    ("ses-1", "PLA"),
    ("ses-2", "SER"),
    ("ses-2", "PLA"),
]

ALPHA = 0.05
PAIRWISE_COMPARISONS = 3
ALPHA_CORR = ALPHA / PAIRWISE_COMPARISONS

# error metric: squared error
def compute_error(y_true, y_pred):
    return (y_true - y_pred) ** 2

# =========================================================
# Helper functions
# =========================================================
def find_condition_folder(base_dir, keywords):
    """
    Find one folder under base_dir containing all keywords in its basename.
    Returns the matched folder path.
    """
    candidates = []
    for path in glob.glob(os.path.join(base_dir, "*")):
        if not os.path.isdir(path):
            continue
        name = os.path.basename(path)
        if all(k.lower() in name.lower() for k in keywords):
            candidates.append(path)

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No folder found in {base_dir} containing keywords {keywords}"
        )
    if len(candidates) > 1:
        print(f"[Warning] Multiple matches in {base_dir} for {keywords}:")
        for c in candidates:
            print("   ", c)
        print(f"Using the first match: {candidates[0]}")
    return candidates[0]


def load_true_and_pred(folder):
    """
    Load y_true_list.npy and y_pred_list.npy from a folder.
    """
    y_true_path = os.path.join(folder, "y_true_list.npy")
    y_pred_path = os.path.join(folder, "y_pred_list.npy")

    if not os.path.exists(y_true_path):
        raise FileNotFoundError(f"Missing file: {y_true_path}")
    if not os.path.exists(y_pred_path):
        raise FileNotFoundError(f"Missing file: {y_pred_path}")

    y_true = np.load(y_true_path).squeeze()
    y_pred = np.load(y_pred_path).squeeze()

    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {y_true.shape} in {folder}")
    if y_pred.ndim != 1:
        raise ValueError(f"y_pred must be 1-D, got shape {y_pred.shape} in {folder}")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch in {folder}: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}"
        )

    return y_true, y_pred


def run_pairwise_wilcoxon(errors_dict, alpha_corr):
    """
    Pairwise Wilcoxon signed-rank tests for all model pairs.
    Returns list of dict results.
    """
    results = []
    model_names = list(errors_dict.keys())

    for a, b in itertools.combinations(model_names, 2):
        err_a = errors_dict[a]
        err_b = errors_dict[b]

        # Handle the case where all paired differences are zero
        diff = err_a - err_b
        if np.allclose(diff, 0):
            w_stat = 0.0
            p_val = 1.0
        else:
            # zero_method='wilcox' is default; can fail if all differences are zero
            w_stat, p_val = wilcoxon(err_a, err_b)

        results.append({
            "comparison": f"{a} vs {b}",
            "W": float(w_stat),
            "p": float(p_val),
            "significant_bonferroni": p_val < alpha_corr
        })

    return results


# =========================================================
# Main batch analysis
# =========================================================
summary_rows = []
pairwise_rows = []

for ses, drug in CONDITIONS:
    print("=" * 70)
    print(f"Condition: {ses}, {drug}")

    # find one matching folder per model
    condition_data = {}
    y_true_ref = None

    for model_name, model_dir in MODEL_DIRS.items():
        # keyword strategy:
        # folder name should contain session and drug
        # and model-specific base dir already narrows the search
        keywords = [ses, drug]

        folder = find_condition_folder(model_dir, keywords)
        print(f"{model_name}: {folder}")

        y_true, y_pred = load_true_and_pred(folder)

        if y_true_ref is None:
            y_true_ref = y_true
        else:
            if len(y_true) != len(y_true_ref):
                raise ValueError(
                    f"Across-model y_true length mismatch for {ses}_{drug}: "
                    f"{model_name} has {len(y_true)}, expected {len(y_true_ref)}"
                )
            # optional strict equality check
            if not np.allclose(y_true, y_true_ref):
                print(f"[Warning] y_true in {model_name} is not identical to reference y_true for {ses}_{drug}")

        condition_data[model_name] = {
            "folder": folder,
            "y_true": y_true,
            "y_pred": y_pred,
            "error": compute_error(y_true, y_pred)
        }

    # collect errors
    errors_dict = {k: v["error"] for k, v in condition_data.items()}

    # Friedman test across 3 matched models
    friedman_stat, friedman_p = friedmanchisquare(
        errors_dict["Tier_1"],
        errors_dict["Tier_2a"],
        errors_dict["Tier_2b"]
    )

    print(f"Friedman test: chi2 = {friedman_stat:.4f}, p = {friedman_p:.6f}")
    print(f"Bonferroni-corrected alpha for pairwise Wilcoxon: {ALPHA_CORR:.6f}")

    # summary row
    summary_rows.append({
        "condition": f"{ses}_{drug}",
        "session": ses,
        "drug": drug,
        "n": len(y_true_ref),
        "friedman_chi2": float(friedman_stat),
        "friedman_p": float(friedman_p),
        "friedman_significant": friedman_p < ALPHA,
        "Tier_1_folder": condition_data["Tier_1"]["folder"],
        "Tier_2a_folder": condition_data["Tier_2a"]["folder"],
        "Tier_2b_folder": condition_data["Tier_2b"]["folder"],
    })

    # pairwise only if you want always; this is common and fine
    pairwise_results = run_pairwise_wilcoxon(errors_dict, ALPHA_CORR)

    for res in pairwise_results:
        print(
            f"{res['comparison']}: W = {res['W']:.4f}, "
            f"p = {res['p']:.6f}, "
            f"{'SIGNIFICANT' if res['significant_bonferroni'] else 'ns'}"
        )

        pairwise_rows.append({
            "condition": f"{ses}_{drug}",
            "session": ses,
            "drug": drug,
            "comparison": res["comparison"],
            "W": res["W"],
            "p": res["p"],
            "alpha_bonferroni": ALPHA_CORR,
            "significant_bonferroni": res["significant_bonferroni"]
        })

# =========================================================
# Save results
# =========================================================
summary_df = pd.DataFrame(summary_rows)
pairwise_df = pd.DataFrame(pairwise_rows)

summary_csv = os.path.join(BASE, "friedman_summary_across_conditions.csv")
pairwise_csv = os.path.join(BASE, "wilcoxon_pairwise_across_conditions.csv")

summary_df.to_csv(summary_csv, index=False)
pairwise_df.to_csv(pairwise_csv, index=False)

print("\n" + "=" * 70)
print("Done.")
print(f"Saved summary:  {summary_csv}")
print(f"Saved pairwise: {pairwise_csv}")

print("\nSummary table:")
print(summary_df)

print("\nPairwise table:")
print(pairwise_df)