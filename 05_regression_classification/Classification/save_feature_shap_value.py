import os
import numpy as np
import pandas as pd

# =========================
# EDIT THESE PATHS
# =========================
# Where your result folders live (each has reshaped_* arrays)
BASE_OUT_DIR = "/.../EMBARC/data/06_BART_regression/Output/Classification_plot/Response/Tier0_ablation/Clinical_variables/"

# Where the raw X files live (only used to read feature names)
BASE_X_DIR = "/.../EMBARC/data/06_BART_regression/Input/x"
X_SUBPATH  = "Site_normalization/Tier0_ablation/Clinical_variables"
TIER_LABEL = "Tier0"  # Tier1_selected_ses-1_SER.csv etc.

# Which folder IDs to process
START_ID = 5
END_ID   = 8

# Folder naming pattern you described
ORDER = [
    ("ses-1", "SER"),
    ("ses-1", "PLA"),
    ("ses-2", "SER"),
    ("ses-2", "PLA"),
]

# =========================
# Helpers
# =========================
def remove_substrings(df, remove_in_cells=False):
    df.index = df.index.astype(str)
    df.index = df.index.str.replace('+AF8', '', regex=False).str.replace('+AC0', '', regex=False)

    if df.index.name:
        df.index.name = str(df.index.name).replace('+AF8', '').replace('+AC0', '')

    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.replace('+AF8', '', regex=False).str.replace('+AC0', '', regex=False)

    if df.columns.name:
        df.columns.name = str(df.columns.name).replace('+AF8', '').replace('+AC0', '')

    if remove_in_cells:
        df.replace({r'\+AF8': '', r'\+AC0': ''}, regex=True, inplace=True)
    return df

# If you want exactly the same naming as your SHAP plots:
unchanged_features = {
    "bmi", "masq2_score_gd", "shaps_total_continuous",
    "w0_score_17", "w1_score_17", "w2_score_17", "w3_score_17", "w4_score_17", "w6_score_17",
    "interview_age", "is_male", "is_employed", "is_chronic",
    "Site", "age", "age_squared", "gender"
}

def rename_feature(feature_name: str) -> str:
    if feature_name in unchanged_features:
        return feature_name
    tokens = feature_name.split("_")
    tokens = [t for t in tokens if t != "original"]
    return "_".join(tokens)

def load_feature_names_from_raw_x(ses_number, medication):
    x_filename = f"{TIER_LABEL}_selected_{ses_number}_{medication}.csv"
    x_path = os.path.join(BASE_X_DIR, X_SUBPATH, x_filename)
    if not os.path.isfile(x_path):
        raise FileNotFoundError(f"Missing raw X file (for feature names): {x_path}")

    X_df = pd.read_csv(x_path, index_col=0)
    X_df = remove_substrings(X_df, remove_in_cells=True)

    # Your convention: rows are feature names in X_df, then you do X = X_df.T
    feature_names = X_df.index.tolist()
    feature_names = [fn.replace("original-", "") for fn in feature_names]
    feature_names = [rename_feature(fn) for fn in feature_names]
    return feature_names

def write_long_csv(shap_arr, feature_names, out_csv, split_name):
    n_samples, n_features = shap_arr.shape
    if len(feature_names) != n_features:
        raise ValueError(f"{split_name}: feature_names {len(feature_names)} != shap cols {n_features}")

    df = pd.DataFrame({
        "split": split_name,
        "sample_id": np.repeat(np.arange(n_samples), n_features),
        "feature": np.tile(feature_names, n_samples),
        "shap_value": shap_arr.reshape(-1),
    })
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

def summarize_shap(shap_arr, feature_names, split_name):
    # summary per feature
    mean_shap = shap_arr.mean(axis=0)
    mean_abs  = np.abs(shap_arr).mean(axis=0)
    std_abs   = np.abs(shap_arr).std(axis=0)

    return pd.DataFrame({
        "split": split_name,
        "feature": feature_names,
        "mean_shap": mean_shap,
        "mean_abs_shap": mean_abs,
        "std_abs_shap": std_abs,
    })

# =========================
# Main loop
# =========================
for folder_id in range(START_ID, END_ID + 1):
    ses_number, medication = ORDER[(folder_id - START_ID) % 4]

    folder_name = f"{folder_id:02d}_{ses_number}_{medication}_save_feature_and_model"
    output_path = os.path.join(BASE_OUT_DIR, folder_name)

    if not os.path.isdir(output_path):
        raise FileNotFoundError(f"Missing results folder: {output_path}")

    # required SHAP arrays
    train_shap_path = os.path.join(output_path, "reshaped_train_shap_array.npy")
    test_shap_path  = os.path.join(output_path, "reshaped_test_shap_array.npy")

    if not os.path.isfile(train_shap_path):
        raise FileNotFoundError(f"{folder_name}: missing {train_shap_path}")
    if not os.path.isfile(test_shap_path):
        raise FileNotFoundError(f"{folder_name}: missing {test_shap_path}")

    train_shap = np.load(train_shap_path)
    test_shap  = np.load(test_shap_path)

    # feature names from raw X (only mapping)
    feature_names = load_feature_names_from_raw_x(ses_number, medication)

    # --- LONG CSVs ---
    out_train_long = os.path.join(output_path, "shap_long_train.csv")
    out_test_long  = os.path.join(output_path, "shap_long_test.csv")
    out_all_long   = os.path.join(output_path, "shap_long_train_test.csv")

    write_long_csv(train_shap, feature_names, out_train_long, "train")
    write_long_csv(test_shap,  feature_names, out_test_long,  "test")

    df_long_all = pd.concat(
        [pd.read_csv(out_train_long, encoding="utf-8-sig"),
         pd.read_csv(out_test_long,  encoding="utf-8-sig")],
        ignore_index=True
    )
    df_long_all.to_csv(out_all_long, index=False, encoding="utf-8-sig")

    # --- SUMMARY CSV (per feature) ---
    df_sum_train = summarize_shap(train_shap, feature_names, "train")
    df_sum_test  = summarize_shap(test_shap,  feature_names, "test")
    df_sum_all   = pd.concat([df_sum_train, df_sum_test], ignore_index=True)

    out_summary = os.path.join(output_path, "shap_feature_summary.csv")
    df_sum_all.to_csv(out_summary, index=False, encoding="utf-8-sig")

    print(f"[OK] {folder_name}")
    print(f"  - {out_all_long}")
    print(f"  - {out_summary}")
