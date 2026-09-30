import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shap
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sensor_list import SENSOR_LIST

FEATURE_NAMES = ["Altitude", "MachNumber", "TRA"] + SENSOR_LIST


class GNNExplainer(nn.Module):
    def __init__(self, model, type_idx, device):
        super().__init__()
        self.model = model
        self.type_idx = type_idx
        self.device = device

    def explain_edges(self, x, dim_idx, steps=250, lr=0.05):
        self.model.train()  # cuDNN requirement
        B, L, N = x.shape
        triu_indices = np.triu_indices(N, k=1)
        mask_logits = nn.Parameter(
            torch.ones(N * (N - 1) // 2, device=self.device) * 0.5
        )
        optimizer = torch.optim.Adam([mask_logits], lr=lr)

        with torch.no_grad():
            self.model.eval()
            h_pool = self.model.forward_headless(
                x.to(self.device), self.type_idx.to(self.device)
            )
            target = h_pool[:, dim_idx]
            self.model.train()

        for _ in range(steps):
            optimizer.zero_grad()
            mask = torch.sigmoid(mask_logits)
            e_mask = torch.ones((N, N), device=self.device)
            e_mask[triu_indices[0], triu_indices[1]] = mask
            e_mask[triu_indices[1], triu_indices[0]] = mask
            out = self.model.forward_headless(
                x.to(self.device), self.type_idx, edge_mask=e_mask.view(1, 1, N, N)
            )
            out_dim = out[:, dim_idx]
            loss = (
                F.mse_loss(out_dim, target)
                + 0.01 * mask.sum()
                - 0.1
                * torch.mean(
                    mask * torch.log(mask + 1e-8)
                    + (1 - mask) * torch.log(1 - mask + 1e-8)
                )
            )
            self.model.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()
        res = np.zeros((N, N))
        res[triu_indices] = torch.sigmoid(mask_logits).detach().cpu().numpy()
        return res + res.T


class HybridExplainer:
    def __init__(self, model, type_idx, device, f_means):
        self.model = model
        self.type_idx = type_idx
        self.device = device
        self.f_means = f_means.to(device).view(1, 1, -1)
        self.N = type_idx.shape[0]
        self.gnn_explainer = GNNExplainer(self.model, type_idx, device)

    def explain_shap(self, x, dim_idx, num_samples, use_fft):
        """SHAP for spatio-temporal importance on a specific latent dimension."""
        B, L, N = x.shape
        x_delta = x.to(self.device) - self.f_means
        xf = torch.fft.rfft(x_delta, dim=1) if use_fft else None
        L_dim = xf.shape[1] if use_fft else L

        # 1. Deterministic Gradient Computation w.r.t embeddings
        self.model.train()

        if use_fft:
            xf_grad = xf.clone().detach().requires_grad_(True)
            x_reconstructed = torch.fft.irfft(xf_grad, n=L, dim=1) + self.f_means
            h_pool = self.model.forward_headless(x_reconstructed, self.type_idx)
            loss = h_pool[:, dim_idx].sum()
            loss.backward()
            grad = xf_grad.grad.abs().cpu().numpy()[0]  # (L_dim, N)
        else:
            x_grad = x.to(self.device).clone().detach().requires_grad_(True)
            h_pool = self.model.forward_headless(x_grad, self.type_idx)
            loss = h_pool[:, dim_idx].sum()
            loss.backward()
            grad = x_grad.grad.abs().cpu().numpy()[0]  # (L_dim, N)

        self.model.eval()

        # Add epsilon to prevent division by zero
        grad = grad + 1e-8
        temp_weights = grad / grad.sum(axis=0, keepdims=True)  # (L_dim, N)

        # 2. Robust Sensor-level SHAP
        num_feat = N  # 24 sensors as features

        def predict_func(masks):
            M = masks.shape[0]
            m = torch.tensor(masks, dtype=torch.float32).to(self.device).view(M, 1, N)

            if use_fft:
                xf_expanded = xf.expand(M, -1, -1)
                x_masked = torch.fft.irfft(xf_expanded * m, n=L, dim=1) + self.f_means
            else:
                x_in = x.to(self.device).expand(M, -1, -1)
                x_masked = (x_in * m.expand(-1, L, -1)) + (
                    self.f_means * (1.0 - m.expand(-1, L, -1))
                )

            with torch.no_grad():
                all_h = []
                for j in range(0, M, 128):
                    h = self.model.forward_headless(x_masked[j : j + 128], self.type_idx)
                    val = h[:, dim_idx]
                    all_h.append(val.cpu().numpy() if isinstance(val, torch.Tensor) else val)
                return np.concatenate(all_h)

        explainer = shap.KernelExplainer(predict_func, np.zeros((1, num_feat)))
        l1_reg = f"num_features({min(15, num_feat)})"

        shap_vals = explainer.shap_values(
            np.ones((1, num_feat)), nsamples=num_samples, l1_reg=l1_reg
        )
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = shap_vals.flatten()  # (N,)

        # 3. Distribute sensor-level SHAP values across temporal/spectral dimensions
        shap_matrix = temp_weights * shap_vals.reshape(1, N)  # (L_dim, N)

        base_val = (
            explainer.expected_value[0]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )
        return shap_matrix.flatten(), L_dim, base_val

    def run_single_dim(self, x, dim_idx, samples, use_fft, plot, unit_idx):
        B, L, N = x.shape
        if B > 1:
            print(f"Running RMS explanation for latent dim {dim_idx} across all {B} units...")
            total_shap_matrix_sq = None
            total_s_shap_sq = np.zeros(self.N)
            total_e_gnn = None
            total_base = 0.0

            for i in tqdm(range(B), desc="Averaging units"):
                x_i = x[i : i + 1]
                full_shap, L_dim, base = self.explain_shap(x_i, dim_idx, samples, use_fft)
                e_gnn = self.gnn_explainer.explain_edges(x_i, dim_idx)
                shap_matrix = full_shap.reshape(L_dim, self.N)
                s_shap_i = shap_matrix.sum(axis=0)

                if total_shap_matrix_sq is None:
                    total_shap_matrix_sq = shap_matrix**2
                    total_e_gnn = e_gnn
                else:
                    total_shap_matrix_sq += shap_matrix**2
                    total_e_gnn += e_gnn
                total_s_shap_sq += s_shap_i**2
                total_base += base

            shap_matrix = np.sqrt(total_shap_matrix_sq / B)
            e_gnn = total_e_gnn / B
            base = total_base / B
            s_shap = np.sqrt(total_s_shap_sq / B)
            data_val = x[:, -1].mean(dim=0).cpu().numpy()
        else:
            full_shap, L_dim, base = self.explain_shap(x, dim_idx, samples, use_fft)
            e_gnn = self.gnn_explainer.explain_edges(x, dim_idx)
            shap_matrix = full_shap.reshape(L_dim, self.N)
            s_shap = shap_matrix.sum(axis=0)
            data_val = x[0, -1].cpu().numpy()

        if plot:
            plot_path = Path(plot)
            base_stem = plot_path.stem
            parent = plot_path.parent

            # 1. Spatio-Temporal or Spectral Heatmap
            plt.figure(figsize=(12, 6))
            cmap_heatmap = "Reds" if B > 1 else "RdBu_r"
            im1 = plt.imshow(
                shap_matrix, aspect="auto", origin="lower", cmap=cmap_heatmap
            )
            plt.title(
                f"{'FFT' if use_fft else 'Temporal'} Heatmap SHAP for Latent Dim {dim_idx} (RMS)"
                if B > 1
                else f"{'FFT' if use_fft else 'Temporal'} Heatmap SHAP for Latent Dim {dim_idx} (Unit {unit_idx})"
            )
            plt.xticks(range(self.N), FEATURE_NAMES, rotation=90)
            plt.xlabel("Feature")
            if use_fft:
                ticks = [0, 1, 2, 5, 10, 25]
                labels = ["inf", "50", "25", "10", "5", "2"]
                plt.yticks(ticks, labels)
                plt.ylabel("Period (cycles)")
            else:
                plt.ylabel("Time (cycles)")
            plt.colorbar(im1)
            plt.tight_layout()
            heatmap_path = parent / f"{base_stem}_latent{dim_idx}_heatmap.png"
            plt.savefig(heatmap_path, dpi=150)
            plt.close()
            print(f"Heatmap saved to: {heatmap_path}")

            # 2. SHAP waterfall or bar plot
            fig = plt.figure(figsize=(12, 6))
            shap_exp = shap.Explanation(
                values=s_shap,
                base_values=float(base),
                data=data_val,
                feature_names=FEATURE_NAMES,
            )
            if B > 1:
                sorted_idx = np.argsort(np.abs(s_shap))
                sorted_shap = s_shap[sorted_idx]
                labels = [FEATURE_NAMES[i] for i in sorted_idx]
                plt.barh(
                    range(len(sorted_shap)), sorted_shap, color="#ff0051", height=0.7
                )
                plt.yticks(range(len(sorted_shap)), labels)
                plt.xlabel("SHAP Value (RMS)")
                plt.grid(axis="x", linestyle="--", alpha=0.5)
                max_abs_shap = max(np.abs(sorted_shap)) if len(sorted_shap) > 0 and max(np.abs(sorted_shap)) > 0 else 1.0
                for i, val in enumerate(sorted_shap):
                    if abs(val) > 0.001:
                        plt.text(
                            val + 0.01 * max_abs_shap,
                            i,
                            f"{val:.2f}",
                            va="center",
                            ha="left",
                            fontsize=9,
                        )
                plt.title(f"Aggregated Sensor SHAP for Latent Dim {dim_idx} - RMS")
            else:
                shap.plots.waterfall(shap_exp, show=False)
                plt.title(f"SHAP Waterfall for Latent Dim {dim_idx} (Unit {unit_idx})")
            waterfall_path = parent / f"{base_stem}_latent{dim_idx}_waterfall.png"
            plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Waterfall/Bar plot saved to: {waterfall_path}")

            # 3. GNN Edge Importance
            plt.figure(figsize=(8, 8))
            im3 = plt.imshow(e_gnn, cmap="hot", aspect="equal")
            plt.title(
                f"Edge importance - GNNExplainer for Latent Dim {dim_idx} (Mean)"
                if B > 1
                else f"Edge importance - GNNExplainer for Latent Dim {dim_idx} (Unit {unit_idx})"
            )
            plt.xticks(range(self.N), FEATURE_NAMES, rotation=90)
            plt.yticks(range(self.N), FEATURE_NAMES)
            plt.xlabel("Feature")
            plt.ylabel("Feature")
            plt.colorbar(im3)
            plt.tight_layout()
            edges_path = parent / f"{base_stem}_latent{dim_idx}_edges.png"
            plt.savefig(edges_path, dpi=150)
            plt.close()
            print(f"GNN Edge Importance saved to: {edges_path}")

        return s_shap, e_gnn, base

    def run_all_dims(self, x, D_dim, samples, use_fft, plot, unit_idx):
        B, L, N = x.shape
        print(f"Running general analysis across all {D_dim} latent dimensions...")
        latent_shap_matrix = np.zeros((D_dim, self.N))

        for dim_idx in range(D_dim):
            print(f"Analyzing Latent Space Dimension Index: {dim_idx} / {D_dim - 1}...")
            if B > 1:
                total_s_shap_sq = np.zeros(self.N)
                for i in range(B):
                    x_i = x[i : i + 1]
                    full_shap, L_dim, _ = self.explain_shap(x_i, dim_idx, samples, use_fft)
                    shap_matrix = full_shap.reshape(L_dim, self.N)
                    s_shap_i = shap_matrix.sum(axis=0)
                    total_s_shap_sq += s_shap_i**2
                s_shap = np.sqrt(total_s_shap_sq / B)
            else:
                full_shap, L_dim, _ = self.explain_shap(x, dim_idx, samples, use_fft)
                shap_matrix = full_shap.reshape(L_dim, self.N)
                s_shap = shap_matrix.sum(axis=0)

            latent_shap_matrix[dim_idx] = s_shap

        if plot:
            plot_path = Path(plot)
            base_stem = plot_path.stem
            parent = plot_path.parent

            plt.figure(figsize=(14, 8))
            im = plt.imshow(
                latent_shap_matrix, aspect="auto", origin="lower", cmap="coolwarm"
            )
            plt.title(
                "Latent Space Feature Importance Heatmap (RMS)"
                if B > 1
                else f"Latent Space Feature Importance Heatmap (Unit {unit_idx})"
            )
            plt.xticks(range(self.N), FEATURE_NAMES, rotation=90)
            plt.xlabel("Feature / Sensor")
            plt.yticks(range(D_dim))
            plt.ylabel("Latent Space Dimension Index")
            plt.colorbar(im)
            plt.tight_layout()
            general_heatmap_path = parent / f"{base_stem}_latent_general_heatmap.png"
            plt.savefig(general_heatmap_path, dpi=150)
            plt.close()
            print(f"General Latent Heatmap saved to: {general_heatmap_path}")

        return latent_shap_matrix


if __name__ == "__main__":
    from model import THGNN, build_model
    from preprocess import get_preprocessed_data, ACTIVE_SENSOR_TYPES

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--unit_idx", type=int, default=0)
    p.add_argument("--fft", action="store_true")
    p.add_argument("--plot")
    p.add_argument("--all_units", action="store_true")
    p.add_argument("--dim_idx", type=int, default=None, help="Explain specific dimension or all if None")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_idx = torch.tensor(ACTIVE_SENSOR_TYPES).to(dev)
    ds = ""
    for x_ds in ["FD001", "FD002", "FD003", "FD004"]:
        if x_ds in args.model_path:
            ds = x_ds
            break
    if not ds:
        raise ValueError(args.model_path, "dataset spec not found")
    
    X_tr, y_tr, _, _, X_te, y_te = get_preprocessed_data("../CMAPSSData", ds)
    if args.all_units:
        x = torch.tensor(X_te, dtype=torch.float32)
    else:
        x = torch.tensor(X_te[args.unit_idx : args.unit_idx + 1], dtype=torch.float32)
    means = torch.FloatTensor(X_tr.mean(axis=(0, 1)))

    with open(
        Path(args.model_path).with_name(
            Path(args.model_path).name.replace(".pth", ".meta").replace("-best", "")
        ),
        "rb",
    ) as f:
        meta = pickle.load(f)
    model = build_model(
        n_sensors=X_te.shape[2],
        n_types=len(set(ACTIVE_SENSOR_TYPES)),
        window_size=50,
        **meta,
    ).to(dev)
    model.load_state_dict(torch.load(args.model_path, map_location=dev), strict=False)
    model.eval()

    # Pass a dummy batch to model to find D
    with torch.no_grad():
        dummy_h = model.forward_headless(x[:1].to(dev), t_idx)
        D_dim = dummy_h.shape[1]

    exp = HybridExplainer(model, t_idx, dev, means)

    if args.dim_idx is not None:
        if args.dim_idx < 0 or args.dim_idx >= D_dim:
            raise ValueError(f"dim_idx {args.dim_idx} out of range [0, {D_dim-1}]")
        s, e, b = exp.run_single_dim(x, args.dim_idx, args.samples, args.fft, args.plot, args.unit_idx)
        print(f"Latent Dim {args.dim_idx} | Base: {b:.2f}")
        print("\nTop Sensors:")
        print(
            "\n ".join(
                [f"{FEATURE_NAMES[i]}: {s[i]:+.4f}" for i in np.argsort(np.abs(s))[::-1][:]]
            )
        )
    else:
        m = exp.run_all_dims(x, D_dim, args.samples, args.fft, args.plot, args.unit_idx)
        print(f"Successfully completed general latent analysis. Matrix shape: {m.shape}")
