import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os
from tqdm import tqdm

from model import THGNN, build_model
from preprocess import get_preprocessed_data, ACTIVE_SENSOR_TYPES


def main():
    parser = argparse.ArgumentParser(
        description="Generate a t-SNE plot of the THGNN latent space."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--dataset", type=str, default="FD001", help="Dataset ID (e.g., FD001)"
    )
    parser.add_argument(
        "--save", type=str, default="tsne.png", help="Output plot filename"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which data split to plot",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="headless",
        choices=["headless", "contrastive"],
        help="Which representation layer to visualize",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of samples to plot (randomly subsampled)",
    )

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WINDOW_SIZE = 50
    BATCH_SIZE = 256
    DATA_PATH = "../CMAPSSData"

    TYPE_IDX = torch.tensor(ACTIVE_SENSOR_TYPES).to(DEVICE)
    N_TYPES = len(set(ACTIVE_SENSOR_TYPES))

    print(f"Loading data for {args.dataset}...")
    X_train, y_train, X_val, y_val, X_test, y_test = get_preprocessed_data(
        DATA_PATH, args.dataset, window_size=WINDOW_SIZE
    )

    if args.split == "train":
        X, y = X_train, y_train
    elif args.split == "val":
        X, y = X_val, y_val
    else:
        X, y = X_test, y_test

    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    try:
        print(f"Loading model from {args.model}...")
        args.model_simple = args.model.removesuffix("-best").removesuffix("-last")
        print(args.model_simple)
        with open(f"{args.model_simple}.meta", "rb") as file:
            kw = pickle.load(file)

        model = build_model(
            n_sensors=X.shape[2], n_types=N_TYPES, window_size=WINDOW_SIZE, **kw
        ).to(DEVICE)
        with open(f"{args.model}.pth", "rb") as file:
            model.load_state_dict(torch.load(file, map_location=DEVICE), strict=False)
    except Exception as e:
        print(f"Error: ({e}) Model for {args.dataset} not found at {args.model}")
        return

    print("Extracting embeddings...")
    embeddings = []
    ruls = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc="Forward Pass"):
            batch_x = batch_x.to(DEVICE)
            if args.layer == "contrastive":
                emb = model.forward_contrastive(batch_x, TYPE_IDX)
            else:
                emb = model.forward_headless(batch_x, TYPE_IDX)

            embeddings.append(emb.cpu().numpy())
            ruls.append(batch_y.numpy())

    embeddings = np.concatenate(embeddings)
    ruls = np.concatenate(ruls)

    # Subsample if there are too many points, as t-SNE scales quadratically
    if len(embeddings) > args.max_samples:
        print(
            f"Subsampling {args.max_samples} points from {len(embeddings)} for t-SNE..."
        )
        idx = np.random.choice(len(embeddings), args.max_samples, replace=False)
        embeddings = embeddings[idx]
        ruls = ruls[idx]

    print("Running t-SNE (this might take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    embeddings_2d = tsne.fit_transform(embeddings)

    print("Plotting...")
    plt.figure(figsize=(10, 8))

    # Use a colormap that highlights the transition well
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=ruls,
        cmap="coolwarm_r",
        alpha=0.7,
        s=15,
        edgecolors="none",
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Remaining Useful Life (RUL)", fontsize=12)

    plt.title(
        f"Latent space t-SNE ({args.dataset} {args.split.capitalize()})",
        fontsize=14,
    )
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Save the plot
    plt.savefig(args.save, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {args.save}")


if __name__ == "__main__":
    main()
