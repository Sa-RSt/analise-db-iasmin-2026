import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import pickle
import math
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from preprocess import get_preprocessed_data, ACTIVE_SENSOR_TYPES
from model import THGNN, build_model
from ast import literal_eval

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate latent representation distribution plots."
    )
    p.add_argument("--model_path", required=True, help="Path to THGNN model (.pth)")
    p.add_argument("--unit_idx", type=int, default=0, help="Unit index for test data")
    p.add_argument("--all_units", action="store_true", help="Analyze all test units")
    p.add_argument(
        "--plot", help="Base path prefix to save output plots (e.g. results/dist)"
    )
    p.add_argument(
        "--kde_dims",
        type=int,
        nargs=2,
        default=None,
        help="Two latent dimensions for joint KDE plot",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for embedding extraction",
    )
    p.add_argument(
        "--n_components", type=str, default="0.8", help="Argumento for PCA n_components"
    )
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_idx = torch.tensor(ACTIVE_SENSOR_TYPES).to(dev)

    # Identify dataset ID from model path
    ds = ""
    for x_ds in ["FD001", "FD002", "FD003", "FD004"]:
        if x_ds in args.model_path:
            ds = x_ds
            break
    if not ds:
        raise ValueError(f"Dataset spec not found in model path: {args.model_path}")

    # Load data
    print(f"Loading data for {ds}...")
    X_tr, y_tr, _, _, X_te, y_te = get_preprocessed_data("../CMAPSSData", ds)
    if args.all_units:
        x = torch.tensor(X_te, dtype=torch.float32)
    else:
        x = torch.tensor(X_te[args.unit_idx : args.unit_idx + 1], dtype=torch.float32)

    # Load model metadata and construct THGNN model
    meta_path = Path(args.model_path).with_name(
        Path(args.model_path).name.replace(".pth", ".meta").replace("-best", "")
    )
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    model = build_model(
        n_sensors=X_te.shape[2],
        n_types=len(set(ACTIVE_SENSOR_TYPES)),
        window_size=50,
        **meta,
    ).to(dev)

    model.load_state_dict(torch.load(args.model_path, map_location=dev), strict=False)
    model.eval()

    # Find latent dimensions
    with torch.no_grad():
        dummy_h = model.forward_headless(x[:1].to(dev), t_idx)
        D_dim = dummy_h.shape[1]

    # Extract latent embeddings
    print(f"Extracting latent embeddings for {x.shape[0]} samples (D_dim={D_dim})...")
    embeddings = []
    with torch.no_grad():
        for i in range(0, x.shape[0], args.batch_size):
            batch_x = x[i : i + args.batch_size].to(dev)
            h_pool = model.forward_headless(batch_x, t_idx)
            embeddings.append(h_pool.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Extracted embeddings shape: {embeddings.shape}")

    # Set up plotting directories
    if args.plot:
        plot_path = Path(args.plot)
        base_stem = plot_path.stem
        parent = plot_path.parent
        parent.mkdir(parents=True, exist_ok=True)
    else:
        parent = Path(".")
        base_stem = "latent"

    # Figure 1: Histograms of all latent dimensions
    n_cols = min(4, D_dim)
    n_rows = math.ceil(D_dim / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if D_dim == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(D_dim):
        ax = axes[i]
        vals = embeddings[:, i]
        ax.hist(
            vals, bins=30, alpha=0.6, color="steelblue", edgecolor="black", density=True
        )
        try:
            sns.kdeplot(vals, ax=ax, color="darkorange", linewidth=1.5)
        except Exception:
            pass
        ax.set_title(f"Latent Dim {i}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.5)

    # Clean up empty subplots
    for j in range(D_dim, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Histograms of Latent Space Dimensions", fontsize=14, y=0.98)
    fig.tight_layout()

    # Save or show histograms
    if args.plot:
        hist_save_path = parent / f"{base_stem}_histograms.png"
        fig.savefig(hist_save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved histograms figure to: {hist_save_path}")
    else:
        plt.show()

    # Figure 2: PCA dimension information content (with n_components=0.8)
    if D_dim > 1 and embeddings.shape[0] >= 2:
        print("Fitting PCA...")
        pca = PCA(n_components=literal_eval(args.n_components))
        pca.fit(embeddings)

        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        n_comps = len(explained_var)

        fig_pca = plt.figure(figsize=(8, 5))
        components = np.arange(1, n_comps + 1)

        bars = plt.bar(
            components,
            explained_var,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            label="Individual Variance",
        )
        plt.plot(
            components,
            cumulative_var,
            marker="o",
            color="red",
            linestyle="-",
            linewidth=2,
            label="Cumulative Variance",
        )
        plt.axhline(y=0.8, color="green", linestyle="--", label="80% Threshold")

        # Annotate heights
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.title(
            f"PCA Variance Explanation (n_components={args.n_components}, {n_comps} components selected)",
            fontsize=12,
        )
        plt.xlabel("Principal Component Index")
        plt.ylabel("Explained Variance Ratio")
        plt.xticks(components)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="best")
        plt.tight_layout()

        if args.plot:
            pca_save_path = parent / f"{base_stem}_pca.png"
            fig_pca.savefig(pca_save_path, dpi=150, bbox_inches="tight")
            plt.close(fig_pca)
            print(f"Saved PCA figure to: {pca_save_path}")
        else:
            plt.show()
    else:
        print("Skipping PCA (D_dim <= 1 or insufficient samples).")

    # Figure 3: Optional joint KDE plot of two dimensions
    if args.kde_dims is not None:
        dim1, dim2 = args.kde_dims
        if dim1 < 0 or dim1 >= D_dim or dim2 < 0 or dim2 >= D_dim:
            raise ValueError(
                f"kde_dims {args.kde_dims} out of bounds for D_dim={D_dim}"
            )

        print(f"Generating joint KDE plot for Latent Dim {dim1} and {dim2}...")
        df_kde = pd.DataFrame(
            {
                f"Latent Dim {dim1}": embeddings[:, dim1],
                f"Latent Dim {dim2}": embeddings[:, dim2],
            }
        )

        g = sns.jointplot(
            data=df_kde,
            x=f"Latent Dim {dim1}",
            y=f"Latent Dim {dim2}",
            kind="kde",
            fill=True,
            cmap="viridis",
            thresh=0.01,
            levels=50,
        )
        g.fig.suptitle(
            f"Joint KDE Plot: Latent Dim {dim1} vs {dim2}", y=1.02, fontsize=12
        )

        if args.plot:
            kde_save_path = parent / f"{base_stem}_joint_kde_{dim1}_{dim2}.png"
            g.savefig(kde_save_path, dpi=150, bbox_inches="tight")
            plt.close(g.fig)
            print(f"Saved joint KDE plot to: {kde_save_path}")
        else:
            plt.show()
