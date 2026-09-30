import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree, DecisionTreeRegressor
from torch.utils.data import DataLoader, TensorDataset

from preprocess import (
    ACTIVE_SENSOR_TYPES,
    get_preprocessed_data,
)
from model import THGNN, build_model

WINDOW_SIZE = 50
BATCH_SIZE = 50
DATA_PATH = "../CMAPSSData"
TYPE_IDX_LIST = ACTIVE_SENSOR_TYPES
N_TYPES = len(set(ACTIVE_SENSOR_TYPES))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="model.pth", help="Path to the saved model"
    )
    parser.add_argument(
        "--dataset", type=str, default="FD001", help="Dataset ID (e.g., FD001)"
    )
    parser.add_argument(
        "--pca",
        type=int,
        default=0,
        help="PCA components, default is no PCA/full dimensionality",
    )
    parser.add_argument("--dt-max-depth", type=int, default=3, help="DT max_depth")
    parser.add_argument("--dt-min-samples-split", type=int, default=2, help="DT min_samples_split")
    parser.add_argument("--fold", type=int, default=None, help="Fold index")
    parser.add_argument("--n-folds", type=int, default=5, help="Total folds")
    parser.add_argument("--loeo", action="store_true", help="LOEO mode")
    parser.add_argument("--vis-max-depth", type=int, default=3, help="Max depth to plot in branching view")
    parser.add_argument("--output", type=str, default="tree_branching_view.png", help="Output plot filename")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    type_idx = torch.tensor(TYPE_IDX_LIST).to(device)

    print(f"Loading and preprocessing {args.dataset}...")
    X_train, y_train, X_val, y_val, X_test, y_test = get_preprocessed_data(
        DATA_PATH,
        args.dataset,
        window_size=WINDOW_SIZE,
        fold=args.fold,
        n_splits=args.n_folds,
        mode="loeo" if args.loeo else "kfold",
    )

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loading model from {args.model}...")
    model_path_simple = args.model.removesuffix("-best").removesuffix("-last")
    with open(f"{model_path_simple}.meta", "rb") as file:
        kw = pickle.load(file)

    model = build_model(
        n_sensors=X_train.shape[2], n_types=N_TYPES, window_size=WINDOW_SIZE, **kw
    ).to(device)
    
    with open(f"{args.model}.pth", "rb") as file:
        model.load_state_dict(torch.load(file, map_location=device), strict=False)

    model.eval()

    print("Extracting training embeddings...")
    X = []
    y = []
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            X.extend(
                model.forward_headless(batch_x, type_idx).numpy(force=True)
            )
            y.extend(batch_y.numpy(force=True))
    X = np.array(X)
    y = np.array(y)

    if args.pca > 0:
        print(f"Fitting PCA with {args.pca} components...")
        pca_model = PCA(n_components=args.pca)
        X = pca_model.fit_transform(X)
        feature_names = [f"PCA Component {i+1}" for i in range(args.pca)]
    else:
        D = X.shape[1]
        feature_names = [f"Embedding Dim {i+1}" for i in range(D)]

    print("Fitting Decision Tree Regressor...")
    dt = DecisionTreeRegressor(
        max_depth=args.dt_max_depth if args.dt_max_depth != -1 else None,
        min_samples_split=args.dt_min_samples_split,
        random_state=42,
    )
    dt.fit(X, y)

    print(f"Plotting logical branching view (max depth = {args.vis_max_depth})...")
    plt.figure(figsize=(24, 12), dpi=300)
    plot_tree(
        dt,
        max_depth=args.vis_max_depth,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        precision=2,
        fontsize=10,
    )
    plt.title(f"Decision Tree Branching View - {args.dataset} (Model: {Path(args.model).name})", fontsize=16)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()
    print(f"Success! Branching view saved to: {args.output}")

if __name__ == "__main__":
    main()
