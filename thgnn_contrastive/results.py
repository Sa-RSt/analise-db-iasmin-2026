import os
from tqdm import tqdm
import re
from uuid import uuid4
import pandas as pd
import numpy as np
from pathlib import Path


def determine_model_kind(file_path, model_path_val):
    path_str = (str(file_path) + "_" + str(model_path_val)).lower()
    if "gru" in path_str:
        return "GRU"
    elif "lstm" in path_str:
        return "LSTM"
    elif "cnn" in path_str:
        return "CNN1D"
    else:
        return "THGNN"


def extract_from_model_path(model_path_str):
    filename = Path(model_path_str).name.lower()
    extracted = {"model_path": model_path_str}

    # 1. train_mode
    for mode in ["supcon", "jepa", "multitask", "simclr", "default"]:
        if mode in filename:
            extracted["train_mode"] = mode
            break

    # 2. dim
    m = re.search(r"-dim(\d+)", filename)
    if m:
        extracted["dim"] = float(m.group(1))

    # 3. hdim
    m = re.search(r"-hdim(\d+)", filename)
    if m:
        extracted["hdim"] = float(m.group(1))

    # 4. n_type_emb
    m = re.search(r"-nte(\d+)", filename)
    if m:
        extracted["n_type_emb"] = float(m.group(1))

    # 5. ema_decay
    m = re.search(r"-ed(\d+(?:\.\d+)?)", filename)
    if m:
        extracted["ema_decay"] = float(m.group(1))

    # 6. alpha
    m = re.search(r"-a(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", filename)
    if m:
        try:
            extracted["alpha"] = float(m.group(1))
        except ValueError:
            pass

    # 7. margin
    m = re.search(r"-m(\d+(?:\.\d+)?)", filename)
    if m:
        extracted["margin"] = float(m.group(1))

    # 8. cov_scale
    m = re.search(r"-c(\d+(?:\.\d+)?)", filename)
    if m:
        extracted["cov_scale"] = float(m.group(1))

    # 9. simclr_scale
    m = re.search(r"-s(\d+(?:\.\d+)?)", filename)
    if m:
        extracted["simclr_scale"] = float(m.group(1))

    return extracted


def assign_trial_ids(group_df, time_threshold=120):
    # group_df is sorted by mtime
    trial_ids = []
    current_trial_id = 0
    current_folds = set()
    last_time = None

    for idx, row in group_df.iterrows():
        fold = row["fold"]
        mtime = row["mtime"]

        # Start a new trial if:
        # - fold is already in current_folds (unless it is 'none')
        # - or time since last file is greater than threshold
        if (fold != "none" and fold in current_folds) or (
            last_time is not None and (mtime - last_time) > time_threshold
        ):
            current_trial_id += 1
            current_folds = {fold}
        else:
            if fold != "none":
                current_folds.add(fold)

        trial_ids.append(current_trial_id)
        last_time = mtime

    group_df = group_df.copy()
    group_df["trial_id"] = trial_ids
    return group_df


def log_files(workspace_dir):
    for path in workspace_dir.iterdir():
        if path.is_dir() and path.name.startswith("log"):
            for csv_file in path.rglob("*.csv"):
                yield (csv_file)


def main():
    # 2. Read, parse and process CSV files in memory-efficient batches
    workspace_dir = Path(__file__).parent
    dfs = []

    for i, f in enumerate(tqdm(log_files(workspace_dir))):
        try:
            df = pd.read_csv(f, low_memory=False)
            if df.empty:
                continue

            # Record metadata on the DataFrame, making a copy to prevent fragmentation warnings
            df = df.copy()
            df["file_path"] = str(f)
            df["mtime"] = os.path.getmtime(f)
            dfs.append(df)

        except Exception as e:
            print("\n")
            print(type(e), e, repr(e))

    if not dfs:
        print("No metrics rows loaded from CSVs. Exiting.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.copy()  # Defragment consolidated DataFrame
    dfs = None  # free memory

    # 3. Vectorized Preprocessing of the Consolidated DataFrame
    if "model_path" not in df_all.columns:
        print("Required column 'model_path' not found. Exiting.")
        return

    # Extract metadata and hyperparameters from model paths to backfill missing columns
    unique_paths = df_all["model_path"].dropna().unique()
    decoded_list = [extract_from_model_path(p) for p in unique_paths]
    decoded_df = pd.DataFrame(decoded_list)

    if not decoded_df.empty:
        df_all = pd.merge(
            df_all, decoded_df, on="model_path", how="left", suffixes=("", "_decoded")
        )
        for col in [
            "train_mode",
            "dim",
            "hdim",
            "n_type_emb",
            "ema_decay",
            "alpha",
            "margin",
            "cov_scale",
            "simclr_scale",
        ]:
            decoded_col = col + "_decoded"
            if decoded_col in df_all.columns:
                if col in df_all.columns:
                    df_all[col] = df_all[col].fillna(df_all[decoded_col])
                else:
                    df_all[col] = df_all[decoded_col]
                df_all = df_all.drop(columns=[decoded_col])

    # Determine model_kind (GRU vs LSTM vs CNN1D vs THGNN)
    path_str = (
        df_all["file_path"] + "_" + df_all["model_path"].astype(str)
    ).str.lower()

    conds = [
        path_str.str.contains("gru"),
        path_str.str.contains("lstm"),
        path_str.str.contains("cnn"),
    ]
    choices = ["GRU", "LSTM", "CNN1D"]
    df_all["model_kind"] = np.select(conds, choices, default="THGNN")

    # Determine subset (transfer_ds if exists and not empty/nan, else dataset_id)
    transfer_ds_col = (
        df_all["transfer_ds"].fillna("")
        if "transfer_ds" in df_all.columns
        else pd.Series("", index=df_all.index)
    )
    dataset_id_col = (
        df_all["dataset_id"].fillna("Unknown")
        if "dataset_id" in df_all.columns
        else pd.Series("Unknown", index=df_all.index)
    )

    df_all["subset"] = np.where(
        (transfer_ds_col == "") | (transfer_ds_col.astype(str).str.strip() == "nan"),
        dataset_id_col,
        transfer_ds_col,
    )

    # Parse validation boolean
    val_col = df_all["validation"].fillna(True)
    if pd.api.types.is_string_dtype(val_col):
        df_all["validation_bool"] = (
            val_col.astype(str).str.lower().isin(["true", "1", "1.0"])
        )
    else:
        df_all["validation_bool"] = val_col.astype(bool)

    # Get base model path
    df_all["base_model"] = (
        df_all["model_path"].astype(str).str.replace(r"-(f|loeo)\d+", "", regex=True)
    )

    # Extract fold
    fold_match = df_all["model_path"].astype(str).str.extract(r"-(f|loeo)(\d+)")
    df_all["fold"] = fold_match[1].fillna("none")

    # Ensure other expected columns exist, fillna with np.nan if missing
    for col in [
        "score",
        "rmse",
        "r2",
        "mae",
        "map",
        "cindex",
        "brier",
        "cdauc",
        "decap_kind",
        "train_mode",
    ]:
        if col not in df_all.columns:
            df_all[col] = (
                np.nan
                if col not in ["decap_kind", "train_mode"]
                else ("none" if col == "decap_kind" else "default")
            )
        elif col == "train_mode":
            df_all[col] = df_all[col].fillna("default")

    # Filter out rows with missing score
    df_all = df_all.dropna(subset=["score"])
    df_all.to_csv("df_all.csv")

    # 4. Separate validation=True and validation=False
    val_true_df = df_all[df_all["validation_bool"] == True]
    val_false_df = df_all[df_all["validation_bool"] == False]

    if val_true_df.empty:
        print(
            "No validation=True rows found. Using validation=False for selection fallback."
        )
        val_true_df = val_false_df.copy()

    if val_false_df.empty:
        print("Warning: No validation=False rows found. Cannot evaluate test metrics.")
        return

    # 5. Group validation=True into trials and find the best trial for each combination
    grouped_val_trials = []
    group_cols = ["subset", "model_kind", "train_mode", "base_model", "decap_kind"]
    for keys, group in val_true_df.groupby(group_cols):
        sorted_group = group.sort_values(by="mtime")
        trial_assigned_group = assign_trial_ids(sorted_group)
        grouped_val_trials.append(trial_assigned_group)

    if len(grouped_val_trials) > 0:
        val_true_with_trials = pd.concat(grouped_val_trials)
    else:
        val_true_with_trials = val_true_df.copy()
        val_true_with_trials["trial_id"] = 0

    # Calculate average score for each trial
    trial_scores = (
        val_true_with_trials.groupby(group_cols + ["trial_id"])
        .agg(mean_score=("score", "mean"), count_folds=("score", "count"))
        .reset_index()
    )

    # Find the minimum trial mean_score for each combination
    best_val_runs = (
        trial_scores.groupby(group_cols)
        .agg(best_val_score=("mean_score", "min"))
        .reset_index()
    )

    # 6. Group validation=False to get mean and std across folds
    test_metrics = (
        val_false_df.groupby(group_cols)
        .agg(
            test_score_mean=("score", "mean"),
            test_score_std=("score", "std"),
            test_rmse_mean=("rmse", "mean"),
            test_rmse_std=("rmse", "std"),
            test_r2_mean=("r2", "mean"),
            test_r2_std=("r2", "std"),
            test_mae_mean=("mae", "mean"),
            test_mae_std=("mae", "std"),
            test_map_mean=("map", "mean"),
            test_map_std=("map", "std"),
            test_cindex_mean=("cindex", "mean"),
            test_cindex_std=("cindex", "std"),
            test_brier_mean=("brier", "mean"),
            test_brier_std=("brier", "std"),
            test_cdauc_mean=("cdauc", "mean"),
            test_cdauc_std=("cdauc", "std"),
            test_folds_count=("score", "count"),
        )
        .reset_index()
    )

    # 7. Merge validation bests with test averages
    merged_runs = pd.merge(
        best_val_runs,
        test_metrics,
        on=group_cols,
        how="inner",
    )

    if merged_runs.empty:
        print(
            "No matched runs found with both validation=True and validation=False logs."
        )
        return

    # 8. For each combination of subset, model kind, and train mode, pick the best validation run
    best_run_per_group = []
    for keys, group in merged_runs.groupby(["subset", "model_kind", "train_mode"]):
        best_idx = group["best_val_score"].idxmin()
        best_row = group.loc[best_idx]
        best_run_per_group.append(best_row)

    df_best_runs = pd.DataFrame(best_run_per_group)

    # 8.5 Extract and merge hyperparameters for the selected best models
    HYPERPARAMS = [
        "pca",
        "epochs",
        "alpha",
        "margin",
        "ema_decay",
        "cov_scale",
        "simclr_scale",
        "dim",
        "hdim",
        "n_type_emb",
        "lstm_hidden_dim",
        "lstm_layers",
        "cnn_layers",
        "cnn_kernel_size",
        "sidekick",
        "blob_hash",
    ]
    existing_hyperparams = [col for col in HYPERPARAMS if col in df_all.columns]
    if existing_hyperparams:
        hp_df = df_all.groupby("base_model")[existing_hyperparams].first().reset_index()
        df_best_runs = pd.merge(df_best_runs, hp_df, on="base_model", how="left")

    # 9. Print results in a beautiful format
    print("=" * 80)
    print("RESULTS SUMMARY (Best Models per Subset, Train Mode, and Model Kind)")
    print("=" * 80)

    def fmt(val, std):
        if pd.isna(std):
            return f"{val:.4f}"
        return f"{val:.4f} ± {std:.4f}"

    for idx, row in df_best_runs.sort_values(
        by=["subset", "train_mode", "model_kind"]
    ).iterrows():
        print(
            f"\nSubset: {row['subset']} | Train Mode: {row['train_mode']} | Model Kind: {row['model_kind']}"
        )
        print(f"Selected Base Model: {Path(row['base_model']).name}")
        print(f"Selected Head Decap: {row['decap_kind']}")

        # Display hyperparameters beautifully
        hp_lines = []
        if "epochs" in row and pd.notna(row["epochs"]):
            hp_lines.append(f"Epochs: {int(row['epochs'])}")
        if "pca" in row and pd.notna(row["pca"]):
            hp_lines.append(f"PCA: {row['pca']}")
        if "sidekick" in row and pd.notna(row["sidekick"]):
            hp_lines.append(f"Sidekick: {row['sidekick']}")
        if "blob_hash" in row and pd.notna(row["blob_hash"]):
            hp_lines.append(f"Blob Hash: {row['blob_hash']}")

        # Display loss / contrastive params
        loss_hps = []
        for col in ["alpha", "margin", "ema_decay", "cov_scale", "simclr_scale"]:
            if col in row and pd.notna(row[col]):
                loss_hps.append(f"{col}={row[col]}")
        if loss_hps:
            hp_lines.append(f"Loss Config: {', '.join(loss_hps)}")

        # Model specific architecture parameters
        arch_hps = []
        if row["model_kind"] == "THGNN":
            cols = ["dim", "hdim", "n_type_emb"]
        elif row["model_kind"] == "GRU":
            cols = ["dim", "hdim"]
        elif row["model_kind"] == "LSTM":
            cols = ["lstm_hidden_dim", "lstm_layers", "dim", "hdim"]
        elif row["model_kind"] == "CNN1D":
            cols = ["cnn_layers", "cnn_kernel_size", "dim", "hdim"]
        else:
            cols = []

        for col in cols:
            if col in row and pd.notna(row[col]):
                val = row[col]
                if isinstance(val, (int, float)) and val == int(val):
                    val = int(val)
                arch_hps.append(f"{col}={val}")

        if arch_hps:
            hp_lines.append(f"Architecture: {', '.join(arch_hps)}")

        if hp_lines:
            print("Hyperparameters:")
            for line in hp_lines:
                print(f"  {line}")

        print(f"Best Validation NASA Score: {row['best_val_score']:.4f}")
        print(f"Test Metrics ({int(row['test_folds_count'])} folds):")
        print(f"  NASA Score:  {fmt(row['test_score_mean'], row['test_score_std'])}")
        print(f"  RMSE:        {fmt(row['test_rmse_mean'], row['test_rmse_std'])}")
        print(f"  R-squared:   {fmt(row['test_r2_mean'], row['test_r2_std'])}")
        print(f"  MAE:         {fmt(row['test_mae_mean'], row['test_mae_std'])}")
        print(f"  MAPE:        {fmt(row['test_map_mean'], row['test_map_std'])}")
        print(f"  C-index:     {fmt(row['test_cindex_mean'], row['test_cindex_std'])}")
        print(f"  Brier Score: {fmt(row['test_brier_mean'], row['test_brier_std'])}")
        print(f"  CD-AUC:      {fmt(row['test_cdauc_mean'], row['test_cdauc_std'])}")
        print("-" * 50)


if __name__ == "__main__":
    main()
