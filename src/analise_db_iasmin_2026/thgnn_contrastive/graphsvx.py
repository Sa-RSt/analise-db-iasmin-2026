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

    def explain_edges(self, x, steps=250, lr=0.05):
        self.model.train()  # cuDNN requirement
        B, L, N = x.shape
        triu_indices = np.triu_indices(N, k=1)
        mask_logits = nn.Parameter(
            torch.ones(N * (N - 1) // 2, device=self.device) * 0.5
        )
        optimizer = torch.optim.Adam([mask_logits], lr=lr)

        with torch.no_grad():
            self.model.eval()
            target = self.model.forward_headless(
                x.to(self.device), self.type_idx.to(self.device)
            )
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
            loss = (
                F.mse_loss(out, target)
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
    def __init__(self, predictor, type_idx, device, f_means):
        self.predictor = predictor
        self.model = predictor.model
        self.type_idx = type_idx
        self.device = device
        self.f_means = f_means.to(device).view(1, 1, -1)
        self.N = type_idx.shape[0]
        self.gnn_explainer = GNNExplainer(self.model, type_idx, device)

    def explain_shap(self, x, num_samples, use_fft):
        """SHAP for spatio-temporal importance with robust Sensor-level grouping and Gradient Distribution."""
        B, L, N = x.shape
        x_delta = x.to(self.device) - self.f_means
        xf = torch.fft.rfft(x_delta, dim=1) if use_fft else None
        L_dim = xf.shape[1] if use_fft else L

        # 1. Deterministic Gradient Computation w.r.t embeddings
        # Set to train mode for RNN backward pass in cuDNN
        self.model.train()

        if use_fft:
            xf_grad = xf.clone().detach().requires_grad_(True)
            x_reconstructed = torch.fft.irfft(xf_grad, n=L, dim=1) + self.f_means
            h_pool = self.model.forward_headless(x_reconstructed, self.type_idx)
            loss = h_pool.sum()
            loss.backward()
            grad = xf_grad.grad.abs().cpu().numpy()[0]  # (L_dim, N)
        else:
            x_grad = x.to(self.device).clone().detach().requires_grad_(True)
            h_pool = self.model.forward_headless(x_grad, self.type_idx)
            loss = h_pool.sum()
            loss.backward()
            grad = x_grad.grad.abs().cpu().numpy()[0]  # (L_dim, N)

        self.model.eval()

        # Add epsilon to prevent division by zero
        grad = grad + 1e-8
        temp_weights = grad / grad.sum(axis=0, keepdims=True)  # (L_dim, N)

        # 2. Robust Sensor-level SHAP (treating each of the 24 sensors as 1 feature)
        num_feat = N  # 24 sensors as features

        def predict_func(masks):
            M = masks.shape[0]
            # masks: (M, 24). Expand along the sequence length (L)
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
                all_p = []
                for j in range(0, M, 128):
                    p = self.predictor(x_masked[j : j + 128], self.type_idx)
                    all_p.append(p.cpu().numpy() if isinstance(p, torch.Tensor) else p)
                return np.concatenate(all_p)

        print(
            f"Running Sensor-SHAP for {'FFT' if use_fft else 'Time-Steps'} ({num_samples} samples)..."
        )
        explainer = shap.KernelExplainer(predict_func, np.zeros((1, num_feat)))

        # Selecting top 15 features is extremely stable for 24 sensors
        l1_reg = f"num_features({min(15, num_feat)})"

        shap_vals = explainer.shap_values(
            np.ones((1, num_feat)), nsamples=num_samples, l1_reg=l1_reg
        )
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = shap_vals.flatten()  # (N,)

        # 3. Distribute sensor-level SHAP values across temporal/spectral dimensions
        # using the deterministic gradient weights
        shap_matrix = temp_weights * shap_vals.reshape(1, N)  # (L_dim, N)

        base_val = (
            explainer.expected_value[0]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )
        return shap_matrix.flatten(), L_dim, base_val

    def run(self, x, samples, use_fft, plot, unit_idx):
        B, L, N = x.shape
        if B > 1:
            print(
                f"Running global explanation averaged across all {B} units using RMS..."
            )
            total_shap_matrix_sq = None
            total_s_shap_sq = np.zeros(self.N)
            total_e_gnn = None
            total_base = 0.0

            for i in tqdm(range(B), desc="Averaging units"):
                x_i = x[i : i + 1]
                full_shap, L_dim, base = self.explain_shap(x_i, samples, use_fft)
                e_gnn = self.gnn_explainer.explain_edges(x_i)
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
            full_shap, L_dim, base = self.explain_shap(x, samples, use_fft)
            e_gnn = self.gnn_explainer.explain_edges(x)
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
                f"{'FFT' if use_fft else 'Temporal'} Heatmap SHAP (RMS)"
                if B > 1
                else f"{'FFT' if use_fft else 'Temporal'} Heatmap SHAP (Unit {unit_idx})"
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
            heatmap_path = parent / f"{base_stem}_heatmap.png"
            plt.savefig(heatmap_path, dpi=150)
            plt.close()
            print(f"Heatmap saved to: {heatmap_path}")

            # 2. Native SHAP Waterfall or Bar Plot
            # Create a dedicated tall figure so text labels and values never overlap
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
                max_abs_shap = (
                    max(np.abs(sorted_shap))
                    if len(sorted_shap) > 0 and max(np.abs(sorted_shap)) > 0
                    else 1.0
                )
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
                plt.title("Aggregated Sensor SHAP - RMS")
            else:
                shap.plots.waterfall(shap_exp, show=False)
                plt.title(f"SHAP Waterfall (Unit {unit_idx})")
            waterfall_path = parent / f"{base_stem}_waterfall.png"
            plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Waterfall/Bar plot saved to: {waterfall_path}")

            # 3. GNN Edge Importance Matrix (Perfect Square)
            plt.figure(figsize=(8, 8))
            im3 = plt.imshow(e_gnn, cmap="hot", aspect="equal")
            plt.title(
                "Edge importance - GNNExplainer (Mean)"
                if B > 1
                else f"Edge importance - GNNExplainer (Unit {unit_idx})"
            )
            plt.xticks(range(self.N), FEATURE_NAMES, rotation=90)
            plt.yticks(range(self.N), FEATURE_NAMES)
            plt.xlabel("Feature")
            plt.ylabel("Feature")
            plt.colorbar(im3)
            plt.tight_layout()
            edges_path = parent / f"{base_stem}_edges.png"
            plt.savefig(edges_path, dpi=150)
            plt.close()
            print(f"GNN Edge Importance saved to: {edges_path}")

        return s_shap, e_gnn, base


def parse_head_params(params_str):
    if not params_str or params_str == "{}":
        return {}
    import json

    try:
        return json.loads(params_str)
    except ValueError:
        pass
    d = {}
    for item in params_str.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
                elif v.lower() == "none":
                    v = None
            d[k] = v
    return d


if __name__ == "__main__":
    from model import THGNN, build_model
    from test import Predictor
    from preprocess import get_preprocessed_data, ACTIVE_SENSOR_TYPES

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--decap_kind", default="none")
    p.add_argument("--unit_idx", type=int, default=0)
    p.add_argument("--fft", action="store_true")
    p.add_argument("--plot")
    p.add_argument("--head_params", default="{}")
    p.add_argument("--pca", type=int, default=0)
    p.add_argument("--use_sidekick", action="store_true")
    p.add_argument("--all_units", action="store_true")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_idx = torch.tensor(ACTIVE_SENSOR_TYPES).to(dev)
    ds = ""
    for x in ["FD001", "FD002", "FD003", "FD004"]:
        if x in args.model_path:
            ds = x
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

    kwargs = parse_head_params(args.head_params)
    if args.decap_kind == "xgb" and "seed" not in kwargs:
        kwargs["seed"] = 42
    elif args.decap_kind == "rf" and "random_state" not in kwargs:
        kwargs["random_state"] = 42
    pred = Predictor(
        args.decap_kind,
        model,
        args.pca,
        args.use_sidekick,
        head_params=kwargs,
    )
    if args.decap_kind != "none":
        pred.train(
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)
                ),
                batch_size=50,
            ),
            t_idx,
        )

    with torch.no_grad():
        r = pred(x.to(dev), t_idx)
        if args.all_units:
            mean_pred = r.mean().item()
            mean_true = y_te.mean()
            print(
                f"All {x.shape[0]} units | Mean True {mean_true:.2f} | Mean Pred {mean_pred:.2f}"
            )
        else:
            print(
                f"Unit {args.unit_idx} | True {y_te[args.unit_idx]} | Pred {r.item():.2f}"
            )

    exp = HybridExplainer(pred, t_idx, dev, means)
    s, e, b = exp.run(x, args.samples, args.fft, args.plot, args.unit_idx)

    print(f"Base: {b:.2f}")
    print("\nTop Sensors (Aggregated SHAP):")
    print(
        "\n ".join(
            [f"{FEATURE_NAMES[i]}: {s[i]:+.4f}" for i in np.argsort(np.abs(s))[::-1][:]]
        )
    )

    print("\nTop Edges (GNNExplainer):")
    N = e.shape[0]
    triu_indices = np.triu_indices(N, k=1)
    edge_vals = e[triu_indices]
    top_indices = np.argsort(edge_vals)[::-1][:5]
    for idx in top_indices:
        u, v = triu_indices[0][idx], triu_indices[1][idx]
        print(f"{FEATURE_NAMES[u]} <-> {FEATURE_NAMES[v]}: {edge_vals[idx]:.4f}")
