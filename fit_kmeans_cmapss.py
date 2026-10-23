import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from analise_db_iasmin_2026.thgnn_contrastive.preprocess import OPERATING_SETTINGS, load_data
from analise_db_iasmin_2026.thgnn_contrastive.split import get_engine_split


def evaluate_silhouette(X, labels, n_bootstrap=5, sample_size=1000):
    scores = []
    n_samples = len(X)
    actual_sample_size = min(sample_size, n_samples)
    for _ in range(n_bootstrap):
        # Random sample indices
        idx = np.random.choice(n_samples, size=actual_sample_size, replace=False)
        score = silhouette_score(X[idx], labels[idx])
        scores.append(score)
    return float(np.mean(scores))


def fit_kmeans_for_dataset(data_path, dataset_id, n_fits=1000, n_splits=5):
    print(f"=== Fitting K-Means for {dataset_id} ===")
    train_df, _, _ = load_data(data_path, dataset_id)
    units = train_df["unit"].unique()

    for fold in range(n_splits):
        print(f"Starting Fold {fold}/{n_splits - 1}...")
        train_units, _ = get_engine_split(
            units,
            val_size=0.2,
            fold=fold,
            n_splits=n_splits,
            random_state=42,
            mode="kfold",
        )
        fold_train_df = train_df[train_df["unit"].isin(train_units)]
        X = fold_train_df[OPERATING_SETTINGS].to_numpy()

        best_score = -float("inf")
        best_kmeans = None

        for _ in tqdm(range(n_fits), desc=f"{dataset_id} Fold {fold}"):
            km = KMeans(n_clusters=6, n_init=1, random_state=None)
            labels = km.fit_predict(X)
            score = evaluate_silhouette(X, labels, n_bootstrap=5, sample_size=1000)

            if score > best_score:
                best_score = score
                best_kmeans = km

        out_path = os.path.join(data_path, f"kmeans_{dataset_id}_fold_{fold}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(best_kmeans, f)
        print(f"Saved {out_path} with best silhouette score: {best_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fit K-Means per fold for CMAPSS datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["FD002", "FD004", "all"],
        help="Dataset to process (FD002, FD004, or all)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="CMAPSSData",
        help="Path to CMAPSS directory",
    )
    parser.add_argument(
        "--n_fits",
        type=int,
        default=1000,
        help="Number of K-Means fits per fold",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of folds",
    )
    args = parser.parse_args()

    datasets = ["FD002", "FD004"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        fit_kmeans_for_dataset(
            args.data_path, ds, n_fits=args.n_fits, n_splits=args.n_splits
        )


if __name__ == "__main__":
    main()
