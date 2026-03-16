# Mancy Chen 17/02/2026 rebuttal of EMBARC
# Binomial test + DeLong test with exact folder mapping
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

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
# Edit the Response section if Response numbering differs from Remission.
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
ALPHA_CI = 0.95   # 95% CI


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
    """
    Load .npy and return 1D vector.
    Supports plain arrays and object arrays of folds.
    """
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.concatenate([np.asarray(x).ravel() for x in arr.ravel()])
    return np.asarray(arr).ravel()


def load_npy_concat(path):
    """
    Load .npy and return array.
    Supports plain arrays and object arrays of folds.
    """
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.concatenate([np.asarray(x) for x in arr.ravel()], axis=0)
    return np.asarray(arr)


def load_all_arrays(folder):
    y_true_path = os.path.join(folder, "y_true_list.npy")
    y_pred_path = os.path.join(folder, "y_pred_list.npy")
    y_proba_path = os.path.join(folder, "y_proba_list.npy")

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

    if os.path.exists(y_proba_path):
        y_proba = load_npy_concat(y_proba_path)
    else:
        y_proba = None

    return y_true, y_pred, y_proba, y_true_path, y_pred_path, y_proba_path


def proba_to_p1(y_proba):
    """
    Convert predict_proba output to P(class=1) vector.
    Accepts:
      (n,2) -> [:,1]
      (n,1) -> [:,0]
      (n,)  -> 그대로 사용
    """
    y_proba = np.asarray(y_proba)

    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        return y_proba[:, 1]
    if y_proba.ndim == 2 and y_proba.shape[1] == 1:
        return y_proba[:, 0]
    if y_proba.ndim == 1:
        return y_proba

    raise ValueError(f"Unexpected y_proba shape: {y_proba.shape}")


# =========================================================
# DeLong implementation
# =========================================================
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)

    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1.0
        T[i:j] = mid
        i = j

    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        preds = predictions_sorted_transposed[r]
        tz[r] = _compute_midrank(preds)
        tx[r] = _compute_midrank(preds[:m])
        ty[r] = _compute_midrank(preds[m:])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.atleast_2d(np.cov(v01, bias=True))
    sy = np.atleast_2d(np.cov(v10, bias=True))
    delong_cov = sx / m + sy / n

    return aucs, delong_cov


def delong_auc_test(y_true, y_score, alpha=0.95):
    """
    H0: AUC = 0.5
    one-sided: H1 AUC > 0.5
    two-sided: H1 AUC != 0.5
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()

    uniq = np.unique(y_true)
    if not set(uniq).issubset({0, 1}):
        raise ValueError(f"y_true must be binary {{0,1}}, got {uniq}")

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"Mismatch: y_true {y_true.shape} vs y_score {y_score.shape}")

    order = np.argsort(-y_true)  # positives first
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    m = int(y_true_sorted.sum())
    n = len(y_true_sorted) - m
    if m == 0 or n == 0:
        raise ValueError("Need at least one positive and one negative sample for AUC.")

    preds = y_score_sorted[np.newaxis, :]
    aucs, cov = _fast_delong(preds, m)

    auc = float(aucs[0])
    var_auc = float(cov[0, 0])
    se_auc = float(np.sqrt(var_auc)) if var_auc >= 0 else np.nan

    z_ci = norm.ppf(0.5 + alpha / 2.0)
    ci_low = max(0.0, auc - z_ci * se_auc)
    ci_high = min(1.0, auc + z_ci * se_auc)

    if se_auc > 0:
        z_stat = (auc - 0.5) / se_auc
        p_one = float(1.0 - norm.cdf(z_stat))   # H1: AUC > 0.5
        p_two = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
    else:
        z_stat = np.inf if auc > 0.5 else 0.0
        p_one = 0.0 if auc > 0.5 else 1.0
        p_two = 0.0 if auc != 0.5 else 1.0

    return {
        "auc": auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se_auc": se_auc,
        "z_auc": float(z_stat),
        "p_delong_one": p_one,
        "p_delong_two": p_two,
    }


# =========================================================
# Binomial test
# =========================================================
def binomial_tests_accuracy(y_true, y_pred):
    """
    Test accuracy against majority-class baseline.
    one-sided: accuracy > p0
    two-sided: accuracy != p0
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    n = int(y_true.size)
    k = int(np.sum(y_true == y_pred))
    acc = k / n if n > 0 else np.nan

    prev = float(np.mean(y_true))
    p0 = max(prev, 1.0 - prev)

    p_one = float(binomtest(k=k, n=n, p=p0, alternative="greater").pvalue)
    p_two = float(binomtest(k=k, n=n, p=p0, alternative="two-sided").pvalue)

    return {
        "n": n,
        "k_correct": k,
        "accuracy": acc,
        "prevalence": prev,
        "majority_baseline": p0,
        "p_binom_one": p_one,
        "p_binom_two": p_two,
    }


# =========================================================
# Main batch analysis
# =========================================================
rows = []

print(f"TASK: {TASK}")
print(f"BASE: {BASE}")

for condition_name in ["ses-1_SER", "ses-1_PLA", "ses-2_SER", "ses-2_PLA"]:
    ses, drug = condition_name.split("_")
    print("\n" + "=" * 70)
    print(f"Condition: {condition_name}")

    for model_name in ["Tier1", "Tier2a", "Tier2b"]:
        folder = get_exact_folder(model_name, condition_name)
        print(f"{model_name}: {folder}")

        y_true, y_pred, y_proba, y_true_path, y_pred_path, y_proba_path = load_all_arrays(folder)

        # Binomial
        b = binomial_tests_accuracy(y_true, y_pred)

        # DeLong
        if y_proba is not None:
            y_score = proba_to_p1(y_proba)
            d = delong_auc_test(y_true, y_score, alpha=ALPHA_CI)
        else:
            d = {
                "auc": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "se_auc": np.nan,
                "z_auc": np.nan,
                "p_delong_one": np.nan,
                "p_delong_two": np.nan,
            }

        print(f"\n--- {model_name} | {condition_name} ---")
        print(f"folder                             = {folder}")
        print(f"accuracy                           = {b['k_correct']}/{b['n']} = {b['accuracy']:.3f}")
        print(f"majority baseline                  = {b['majority_baseline']:.3f}")
        print(f"binomial one-sided                 = {b['p_binom_one']:.6f}")
        print(f"binomial two-sided                 = {b['p_binom_two']:.6f}")
        if not np.isnan(d["auc"]):
            print(f"AUC                                = {d['auc']:.3f}")
            print(f"95% CI                             = [{d['ci_low']:.3f}, {d['ci_high']:.3f}]")
            print(f"DeLong one-sided                   = {d['p_delong_one']:.6f}")
            print(f"DeLong two-sided                   = {d['p_delong_two']:.6f}")
        else:
            print("AUC / DeLong                       = skipped (missing y_proba_list.npy)")

        rows.append({
            "task": TASK,
            "condition": condition_name,
            "session": ses,
            "drug": drug,
            "model": model_name,

            "folder": folder,
            "folder_name": os.path.basename(folder),
            "y_true_path": y_true_path,
            "y_pred_path": y_pred_path,
            "y_proba_path": y_proba_path if os.path.exists(y_proba_path) else np.nan,

            "n": b["n"],
            "k_correct": b["k_correct"],
            "accuracy": b["accuracy"],
            "prevalence": b["prevalence"],
            "majority_baseline": b["majority_baseline"],
            "p_binom_one": b["p_binom_one"],
            "p_binom_two": b["p_binom_two"],

            "auc": d["auc"],
            "ci_low": d["ci_low"],
            "ci_high": d["ci_high"],
            "se_auc": d["se_auc"],
            "z_auc": d["z_auc"],
            "p_delong_one": d["p_delong_one"],
            "p_delong_two": d["p_delong_two"],
        })


# =========================================================
# Save results
# =========================================================
results_df = pd.DataFrame(rows)

out_csv = os.path.join(
    BASE,
    f"{TASK.lower()}_binomial_delong_results_exact_mapping.csv"
)

results_df.to_csv(out_csv, index=False)

print("\n" + "=" * 70)
print("Done.")
print(f"Saved results to: {out_csv}")

print("\nResults table:")
print(results_df)