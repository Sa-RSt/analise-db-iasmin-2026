import torch
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from model import THGNN, build_model
from test import Predictor
from preprocess import get_preprocessed_data, ACTIVE_SENSOR_TYPES
from graphsvx import HybridExplainer

def calculate_auc(y_values):
    """Normalize steps to [0, 1] and calculate Area Under Curve (trapezoidal rule)."""
    x_steps = np.linspace(0, 1, len(y_values))
    return np.trapz(y_values, x_steps)

def main():
    parser = argparse.ArgumentParser(description="Verify accuracy of SHAP and GNNExplainer explanations via Deletion & Insertion.")
    parser.add_argument("--model_path", required=True, help="Path to THGNN model checkpoint.")
    parser.add_argument("--dataset", default="FD001", help="CMAPSS dataset ID.")
    parser.add_argument("--samples", type=int, default=2000, help="SHAP coalition samples.")
    parser.add_argument("--decap_kind", default="xgb", help="Regression decap head.")
    parser.add_argument("--unit_idx", type=int, default=3, help="CMAPSS test unit index to verify.")
    parser.add_argument("--fft", action="store_true", help="Explain in frequency domain.")
    parser.add_argument("--plot", default="fidelity_curves.png", help="Path to save fidelity curves plot.")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_idx = torch.tensor(ACTIVE_SENSOR_TYPES).to(dev)
    
    # 1. Load Data
    X_tr, y_tr, _, _, X_te, y_te = get_preprocessed_data("../CMAPSSData", args.dataset)
    x = torch.tensor(X_te[args.unit_idx : args.unit_idx + 1], dtype=torch.float32)
    means = torch.FloatTensor(X_tr.mean(axis=(0, 1)))

    # 2. Reconstruct Model
    with open(
        Path(args.model_path)
        .with_stem(Path(args.model_path).stem.removesuffix("-best"))
        .with_suffix(".meta"),
        "rb",
    ) as f:
        meta = pickle.load(f)
    model = build_model(
        n_sensors=X_te.shape[2],
        n_types=len(set(ACTIVE_SENSOR_TYPES)),
        window_size=50,
        **meta,
    ).to(dev)
    model.load_state_dict(
        torch.load(args.model_path + ".pth", map_location=dev), strict=False
    )
    model.eval()

    # 3. Initialize & Train Predictor
    kwargs = {"random_state": 42} if args.decap_kind != "none" else {}
    pred = Predictor(
        args.decap_kind,
        model,
        0,
        False,
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

    # 4. Generate Predictions and Explanations
    with torch.no_grad():
        orig_pred = pred(x.to(dev), t_idx).item()
    print(f"\n[Verification] Target Unit: {args.unit_idx} | True RUL: {y_te[args.unit_idx]} | Pred: {orig_pred:.4f}")

    exp = HybridExplainer(pred, t_idx, dev, means)
    
    print("\n>>> STEP 1: Running GNNExplainer for Edge Importance...")
    e_gnn = exp.gnn_explainer.explain_edges(x)
    
    print(">>> STEP 2: Running SHAP for Sensor Importance...")
    shap_vals, L_dim, base_val = exp.explain_shap(x, args.samples, args.fft)
    
    # ==========================================
    # VERIFICATION PART I: SHAP (SENSORS)
    # ==========================================
    print("\n" + "="*50)
    print("          VERIFYING SHAP (SENSORS) FIDELITY")
    print("="*50)
    
    # Aggregate SHAP over temporal/spectral dimension
    s_shap = shap_vals.reshape(L_dim, exp.N).sum(axis=0)
    sorted_indices = np.argsort(np.abs(s_shap))[::-1]
    reverse_sorted_indices = sorted_indices[::-1]

    N = exp.N
    L = x.shape[1]

    # Helper to evaluate model with masked sensors
    def evaluate_sensor_mask(active_sensors):
        mask = torch.zeros((1, L, N), dtype=torch.float32).to(dev)
        for s_idx in active_sensors:
            mask[0, :, s_idx] = 1.0
        x_masked = (x.to(dev) * mask) + (means.to(dev).view(1, 1, -1) * (1.0 - mask))
        with torch.no_grad():
            return pred(x_masked, t_idx).item()

    # Deletion Curve
    del_top_preds = [orig_pred]
    for k in range(1, N + 1):
        del_top_preds.append(evaluate_sensor_mask(sorted_indices[k:]))
        
    del_bottom_preds = [orig_pred]
    for k in range(1, N + 1):
        del_bottom_preds.append(evaluate_sensor_mask(reverse_sorted_indices[k:]))
        
    np.random.seed(42)
    random_del_runs = []
    for _ in range(10):
        rand_order = np.random.permutation(N)
        run_preds = [orig_pred]
        for k in range(1, N + 1):
            run_preds.append(evaluate_sensor_mask(rand_order[k:]))
        random_del_runs.append(run_preds)
    del_random_preds = np.mean(random_del_runs, axis=0)

    del_top_error = np.abs(np.array(del_top_preds) - orig_pred)
    del_bottom_error = np.abs(np.array(del_bottom_preds) - orig_pred)
    del_random_error = np.abs(np.array(del_random_preds) - orig_pred)

    auc_del_top = calculate_auc(del_top_error)
    auc_del_bottom = calculate_auc(del_bottom_error)
    auc_del_random = calculate_auc(del_random_error)

    # Insertion Curve
    bg_pred = evaluate_sensor_mask([])
    ins_top_preds = [bg_pred]
    for k in range(1, N + 1):
        ins_top_preds.append(evaluate_sensor_mask(sorted_indices[:k]))
        
    ins_bottom_preds = [bg_pred]
    for k in range(1, N + 1):
        ins_bottom_preds.append(evaluate_sensor_mask(reverse_sorted_indices[:k]))
        
    random_ins_runs = []
    for _ in range(10):
        rand_order = np.random.permutation(N)
        run_preds = [bg_pred]
        for k in range(1, N + 1):
            run_preds.append(evaluate_sensor_mask(rand_order[:k]))
        random_ins_runs.append(run_preds)
    ins_random_preds = np.mean(random_ins_runs, axis=0)

    ins_top_proximity = np.maximum(0, np.abs(bg_pred - orig_pred) - np.abs(np.array(ins_top_preds) - orig_pred))
    ins_bottom_proximity = np.maximum(0, np.abs(bg_pred - orig_pred) - np.abs(np.array(ins_bottom_preds) - orig_pred))
    ins_random_proximity = np.maximum(0, np.abs(bg_pred - orig_pred) - np.abs(np.array(ins_random_preds) - orig_pred))

    auc_ins_top = calculate_auc(ins_top_proximity)
    auc_ins_bottom = calculate_auc(ins_bottom_proximity)
    auc_ins_random = calculate_auc(ins_random_proximity)

    # ==========================================
    # VERIFICATION PART II: GNNEXPLAINER (EDGES)
    # ==========================================
    print("\n" + "="*50)
    print("       VERIFYING GNNEXPLAINER (EDGES) FIDELITY")
    print("="*50)

    triu_indices = np.triu_indices(N, k=1)
    edge_scores = e_gnn[triu_indices[0], triu_indices[1]]
    
    # Sort the 276 edges by absolute importance
    sorted_edge_indices = np.argsort(np.abs(edge_scores))[::-1]
    reverse_sorted_edge_indices = sorted_edge_indices[::-1]
    num_edges = len(sorted_edge_indices)

    # Helper to evaluate model with custom edge mask
    def evaluate_edge_mask(active_edge_indices):
        e_mask = torch.eye(N, device=dev)  # keep self-loops
        u_idx = triu_indices[0][active_edge_indices]
        v_idx = triu_indices[1][active_edge_indices]
        e_mask[u_idx, v_idx] = 1.0
        e_mask[v_idx, u_idx] = 1.0
        with torch.no_grad():
            return pred(x.to(dev), t_idx, edge_mask=e_mask.view(1, 1, N, N)).item()

    # Deletion Curve (Edges)
    # Evaluating 276 steps can be fast-tracked by evaluating in steps of 10 edges to optimize runtime
    step_size = 10
    steps = list(range(0, num_edges + 1, step_size))
    if steps[-1] != num_edges:
        steps.append(num_edges)

    edge_del_top_preds = []
    for k in steps:
        edge_del_top_preds.append(evaluate_edge_mask(sorted_edge_indices[k:]))
        
    edge_del_bottom_preds = []
    for k in steps:
        edge_del_bottom_preds.append(evaluate_edge_mask(reverse_sorted_edge_indices[k:]))
        
    random_edge_del_runs = []
    for _ in range(5): # 5 runs for edges to save computation
        rand_order = np.random.permutation(num_edges)
        run_preds = []
        for k in steps:
            run_preds.append(evaluate_edge_mask(rand_order[k:]))
        random_edge_del_runs.append(run_preds)
    edge_del_random_preds = np.mean(random_edge_del_runs, axis=0)

    edge_del_top_error = np.abs(np.array(edge_del_top_preds) - orig_pred)
    edge_del_bottom_error = np.abs(np.array(edge_del_bottom_preds) - orig_pred)
    edge_del_random_error = np.abs(np.array(edge_del_random_preds) - orig_pred)

    auc_edge_del_top = calculate_auc(edge_del_top_error)
    auc_edge_del_bottom = calculate_auc(edge_del_bottom_error)
    auc_edge_del_random = calculate_auc(edge_del_random_error)

    # Insertion Curve (Edges)
    edge_bg_pred = evaluate_edge_mask([]) # Only self-loops active
    
    edge_ins_top_preds = []
    for k in steps:
        edge_ins_top_preds.append(evaluate_edge_mask(sorted_edge_indices[:k]))
        
    edge_ins_bottom_preds = []
    for k in steps:
        edge_ins_bottom_preds.append(evaluate_edge_mask(reverse_sorted_edge_indices[:k]))
        
    random_edge_ins_runs = []
    for _ in range(5):
        rand_order = np.random.permutation(num_edges)
        run_preds = []
        for k in steps:
            run_preds.append(evaluate_edge_mask(rand_order[:k]))
        random_edge_ins_runs.append(run_preds)
    edge_ins_random_preds = np.mean(random_edge_ins_runs, axis=0)

    edge_ins_top_proximity = np.maximum(0, np.abs(edge_bg_pred - orig_pred) - np.abs(np.array(edge_ins_top_preds) - orig_pred))
    edge_ins_bottom_proximity = np.maximum(0, np.abs(edge_bg_pred - orig_pred) - np.abs(np.array(edge_ins_bottom_preds) - orig_pred))
    edge_ins_random_proximity = np.maximum(0, np.abs(edge_bg_pred - orig_pred) - np.abs(np.array(edge_ins_random_preds) - orig_pred))

    auc_edge_ins_top = calculate_auc(edge_ins_top_proximity)
    auc_edge_ins_bottom = calculate_auc(edge_ins_bottom_proximity)
    auc_edge_ins_random = calculate_auc(edge_ins_random_proximity)

    # ==========================================
    # FINAL CONSOLIDATED REPORT
    # ==========================================
    print("\n" + "="*60)
    print("               CONSOLIDATED XAI FIDELITY REPORT")
    print("="*60)
    print(f"{'Method / Curve / Ordering':<35} | {'Area Under Curve (AUC)':<22}")
    print("-"*60)
    print(f"{'SHAP Deletion: Top First':<35} | {auc_del_top:<22.4f} (Ideal: Higher)")
    print(f"{'SHAP Deletion: Random':<35} | {auc_del_random:<22.4f}")
    print(f"{'SHAP Deletion: Bottom First':<35} | {auc_del_bottom:<22.4f} (Ideal: Lower)")
    print("-"*60)
    print(f"{'SHAP Insertion: Top First':<35} | {auc_ins_top:<22.4f} (Ideal: Higher)")
    print(f"{'SHAP Insertion: Random':<35} | {auc_ins_random:<22.4f}")
    print(f"{'SHAP Insertion: Bottom First':<35} | {auc_ins_bottom:<22.4f} (Ideal: Lower)")
    print("="*60)
    print(f"{'GNNEdge Deletion: Top First':<35} | {auc_edge_del_top:<22.4f} (Ideal: Higher)")
    print(f"{'GNNEdge Deletion: Random':<35} | {auc_edge_del_random:<22.4f}")
    print(f"{'GNNEdge Deletion: Bottom First':<35} | {auc_edge_del_bottom:<22.4f} (Ideal: Lower)")
    print("-"*60)
    print(f"{'GNNEdge Insertion: Top First':<35} | {auc_edge_ins_top:<22.4f} (Ideal: Higher)")
    print(f"{'GNNEdge Insertion: Random':<35} | {auc_edge_ins_random:<22.4f}")
    print(f"{'GNNEdge Insertion: Bottom First':<35} | {auc_edge_ins_bottom:<22.4f} (Ideal: Lower)")
    print("="*60)
    
    # Mathematical Validation Statements
    shap_deletion_valid = auc_del_top > auc_del_random > auc_del_bottom
    shap_insertion_valid = auc_ins_top > auc_ins_random > auc_ins_bottom
    gnn_deletion_valid = auc_edge_del_top > auc_edge_del_random > auc_edge_del_bottom
    gnn_insertion_valid = auc_edge_ins_top > auc_edge_ins_random > auc_edge_ins_bottom
    
    print("\nMathematical Sincerity Check:")
    print(f" [SHAP Sensors Deletion]  : {'PASS' if shap_deletion_valid else 'FAIL'}")
    print(f" [SHAP Sensors Insertion] : {'PASS' if shap_insertion_valid else 'FAIL'}")
    print(f" [GNN Edges Deletion]     : {'PASS' if gnn_deletion_valid else 'FAIL'}")
    print(f" [GNN Edges Insertion]    : {'PASS' if gnn_insertion_valid else 'FAIL'}")

    if shap_deletion_valid and shap_insertion_valid and gnn_deletion_valid and gnn_insertion_valid:
        print("\n>>> CONCLUSION: BOTH SHAP AND GNNEXPLAINER ARE FULLY VERIFIED AS FAITHFUL! <<<")
    else:
        print("\n>>> CONCLUSION: Partial verification complete. <<<")

    # Plotting 2x2 Subplots
    if args.plot:
        plt.figure(figsize=(12, 10))
        
        # 1. SHAP Deletion
        plt.subplot(2, 2, 1)
        plt.plot(del_top_error, label=f"Top-First (AUC: {auc_del_top:.2f})", color="crimson", linewidth=2.5)
        plt.plot(del_random_error, label=f"Random (AUC: {auc_del_random:.2f})", color="gray", linestyle="--", linewidth=2)
        plt.plot(del_bottom_error, label=f"Bottom-First (AUC: {auc_del_bottom:.2f})", color="forestgreen", linewidth=2.5)
        plt.title("Sensor Deletion Curve (Prediction Collapse)")
        plt.xlabel("Sensors Deleted")
        plt.ylabel("Absolute Prediction Change")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        
        # 2. SHAP Insertion
        plt.subplot(2, 2, 2)
        plt.plot(ins_top_proximity, label=f"Top-First (AUC: {auc_ins_top:.2f})", color="forestgreen", linewidth=2.5)
        plt.plot(ins_random_proximity, label=f"Random (AUC: {auc_ins_random:.2f})", color="gray", linestyle="--", linewidth=2)
        plt.plot(ins_bottom_proximity, label=f"Bottom-First (AUC: {auc_ins_bottom:.2f})", color="crimson", linewidth=2.5)
        plt.title("Sensor Insertion Curve (Prediction Recovery)")
        plt.xlabel("Sensors Restored")
        plt.ylabel("Prediction Recovery Proximity")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        
        # 3. GNNEdge Deletion
        plt.subplot(2, 2, 3)
        plt.plot(steps, edge_del_top_error, label=f"Top-First (AUC: {auc_edge_del_top:.2f})", color="crimson", linewidth=2.5)
        plt.plot(steps, edge_del_random_error, label=f"Random (AUC: {auc_edge_del_random:.2f})", color="gray", linestyle="--", linewidth=2)
        plt.plot(steps, edge_del_bottom_error, label=f"Bottom-First (AUC: {auc_edge_del_bottom:.2f})", color="forestgreen", linewidth=2.5)
        plt.title("GNN Edge Deletion Curve (Prediction Collapse)")
        plt.xlabel("Edges Deleted")
        plt.ylabel("Absolute Prediction Change")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        
        # 4. GNNEdge Insertion
        plt.subplot(2, 2, 4)
        plt.plot(steps, edge_ins_top_proximity, label=f"Top-First (AUC: {auc_edge_ins_top:.2f})", color="forestgreen", linewidth=2.5)
        plt.plot(steps, edge_ins_random_proximity, label=f"Random (AUC: {auc_edge_ins_random:.2f})", color="gray", linestyle="--", linewidth=2)
        plt.plot(steps, edge_ins_bottom_proximity, label=f"Bottom-First (AUC: {auc_edge_ins_bottom:.2f})", color="crimson", linewidth=2.5)
        plt.title("GNN Edge Insertion Curve (Prediction Recovery)")
        plt.xlabel("Edges Restored")
        plt.ylabel("Prediction Recovery Proximity")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(args.plot)
        print(f"\nAll fidelity curves saved to: {args.plot}")

if __name__ == "__main__":
    main()
