from time import perf_counter
import pickle
from warnings import warn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
from model import THGNN, build_model
from preprocess import get_preprocessed_data, ACTIVE_SENSOR_TYPES
from scoring import calculate_rmse, calculate_score

# Hyperparameters
WINDOW_SIZE = 50
DATA_PATH = "../CMAPSSData"

# Active sensor types
TYPE_IDX_LIST = ACTIVE_SENSOR_TYPES
N_TYPES = len(set(ACTIVE_SENSOR_TYPES))


def augment_batch(x):
    """
    Apply augmentations to a batch of time-series windows (B, L, N).
    """
    import torch

    # 1. Jitter (Gaussian noise)
    noise = torch.randn_like(x) * 0.02
    x_aug = x + noise

    # 2. Scaling
    scaling_factor = torch.randn(x.size(0), 1, x.size(2)).to(x.device) * 0.1 + 1.0
    x_aug = x_aug * scaling_factor

    return x_aug


def nt_xent_loss(z1, z2, temperature: float = 0.1):
    """
    Normalized Temperature-scaled Cross Entropy loss.
    """
    import torch

    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)

    N = z1.size(0)
    z = torch.cat((z1, z2), dim=0)  # (2N, D)

    sim_matrix = torch.matmul(z, z.T) / temperature  # (2N, 2N)

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat(
        [torch.arange(N, 2 * N, device=z1.device), torch.arange(N, device=z1.device)]
    )

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim_matrix.masked_fill_(mask, -1e4)

    loss = torch.nn.functional.cross_entropy(sim_matrix, labels)
    return loss


def supcon_loss(z, y, margin: float = 10.0, temperature: float = 0.1):
    """
    Supervised Contrastive Loss for regression (RUL).
    Samples with |y_i - y_j| <= margin are considered positive pairs.
    """
    import torch

    z = torch.nn.functional.normalize(z, dim=1)
    N = z.size(0)

    # Compute similarity matrix
    sim_matrix = torch.matmul(z, z.T) / temperature  # (N, N)

    # Compute distance matrix for labels
    # y shape (N,)
    label_dist = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))  # (N, N)

    # Positive pairs: |y_i - y_j| <= margin
    mask = (label_dist <= margin).to(z.dtype)

    # Mask out self-similarity
    diag_mask = torch.eye(N, device=z.device)
    mask = mask - diag_mask
    mask = torch.clamp(mask, 0.0, 1.0)

    # Numerical stability
    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()

    # Exclude diagonal from denominator
    exp_logits = torch.exp(logits) * (1.0 - diag_mask)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

    # Compute mean of log-likelihood over positives
    pos_count = mask.sum(1)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (pos_count + 1e-9)

    loss = -mean_log_prob_pos.mean()
    return loss


def sparsemax_np(v):
    """
    Numpy implementation of sparsemax.
    """
    v = np.nan_to_num(v, nan=0.0, posinf=10000.0, neginf=-10000.0)
    v_sorted = np.sort(v)[::-1]
    dim = len(v)
    cumsum = np.cumsum(v_sorted)
    indices = np.arange(1, dim + 1)
    cond = 1.0 + indices * v_sorted > cumsum

    where_cond = np.where(cond)[0]
    if len(where_cond) == 0:
        return np.ones_like(v) / dim

    k = np.max(where_cond) + 1
    tau = (cumsum[k - 1] - 1.0) / k
    return np.maximum(0.0, v - tau)


def supcon_l1_loss(z, y, weights, margin: float = 10.0, cov_scale: float = 1.0):
    """
    z: shape (N, D)
    y: shape (N,)
    weights: shape (D,)
    margin: float
    cov_scale: float
    """
    import torch

    N, D = z.shape

    # 1. Weighted L1 loss on positive pairs
    label_dist = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))  # (N, N)
    mask = (label_dist <= margin).to(z.dtype)
    diag_mask = torch.eye(N, device=z.device)
    mask = mask - diag_mask
    mask = torch.clamp(mask, 0.0, 1.0)

    pos_count = mask.sum()
    if pos_count > 0:
        # L1 distance: |z_i - z_j|
        diff = torch.abs(z.unsqueeze(1) - z.unsqueeze(0))  # (N, N, D)
        weighted_diff = diff * weights.view(1, 1, D)
        l1_dist_pairs = weighted_diff.sum(dim=-1)  # (N, N)
        loss_l1 = (mask * l1_dist_pairs).sum() / pos_count / D
    else:
        loss_l1 = torch.tensor(0.0, device=z.device)

    # 2. Off-diagonal covariance penalty
    # Center the representations
    mean = z.mean(dim=0)
    z_centered = z - mean
    # Compute covariance matrix
    # If N is 1, avoid division by zero
    denom = N - 1 if N > 1 else 1
    cov = (z_centered.T @ z_centered) / denom  # (D, D)

    # Off-diagonal elements
    diag = torch.eye(D, device=z.device)
    loss_cov = (cov * (1.0 - diag)).pow(2).sum() / D

    return z.mean().pow(2) + (loss_l1 + cov_scale * loss_cov) / (1 + cov_scale)


def supcon_l2_loss(z, y, margin: float = 10.0, cov_scale: float = 1.0):
    """
    z: shape (N, D)
    y: shape (N,)
    margin: float
    cov_scale: float
    """
    import torch

    N, D = z.shape

    # 1. L2 loss on positive pairs
    label_dist = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))  # (N, N)
    mask = (label_dist <= margin).to(z.dtype)
    diag_mask = torch.eye(N, device=z.device)
    mask = mask - diag_mask
    mask = torch.clamp(mask, 0.0, 1.0)

    pos_count = mask.sum()
    if pos_count > 0:
        # L2 distance: sqrt(sum((z_i - z_j)^2))
        diff_sq = (z.unsqueeze(1) - z.unsqueeze(0)).pow(2)  # (N, N, D)
        l2_dist_pairs = torch.sqrt(diff_sq.sum(dim=-1) + 1e-12)  # (N, N)
        loss_l2 = (mask * l2_dist_pairs).sum() / pos_count / D
    else:
        loss_l2 = torch.tensor(0.0, device=z.device)

    # 2. Off-diagonal covariance penalty
    # Center the representations
    mean = z.mean(dim=0)
    z_centered = z - mean
    # Compute covariance matrix
    # If N is 1, avoid division by zero
    denom = N - 1 if N > 1 else 1
    cov = (z_centered.T @ z_centered) / denom  # (D, D)

    # Off-diagonal elements
    diag = torch.eye(D, device=z.device)
    loss_cov = (cov * (1.0 - diag)).pow(2).sum() / D

    return loss_l2 + cov_scale * loss_cov


def evaluate_dt_performance(model, train_loader, val_loader, type_idx, device):
    import torch
    from sklearn.tree import DecisionTreeRegressor
    from scoring import calculate_rmse
    import numpy as np

    model.eval()
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    type_idx = type_idx.to(device)

    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            emb = model.forward_headless(batch_x, type_idx)
            X_train.append(emb.cpu().numpy())
            y_train.append(batch_y.cpu().numpy())

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            emb = model.forward_headless(batch_x, type_idx)
            X_val.append(emb.cpu().numpy())
            y_val.append(batch_y.cpu().numpy())

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    try:
        # Train Decision Tree
        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X_train, y_train)

        # Predict on validation set
        y_pred = dt.predict(X_val)

        # Calculate validation RMSE
        rmse = calculate_rmse(y_val, y_pred)
        if np.isnan(rmse) or np.isinf(rmse):
            return float("inf")
        return rmse
    except Exception as e:
        print(f"Error during DecisionTree evaluation: {e}")
        return float("inf")


def train(
    save_path,
    dataset_id,
    epochs,
    train_mode,
    alpha,
    margin,
    d,
    d_hidden,
    n_type_emb,
    ema_decay=0.99,
    fold=None,
    n_splits=5,
    cv_mode="kfold",
    loaded_data=None,
    cov_scale=1.0,
    simclr_scale=1.0,
    cnn_kernel_size=0,
    model_type="thgnn",
    gru_hidden_dim=64,
    gru_layers=1,
    lstm_hidden_dim=64,
    lstm_layers=1,
    cnn_layers=2,
):
    import torch
    import torch.optim as optim
    from torch.amp import GradScaler, autocast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    type_idx = torch.tensor(TYPE_IDX_LIST).to(device)

    BATCH_SIZE = {
        "simclr": 256,
        "multitask": 256,
        "supcon": 256,
        "supconL1": 256,
        "supconL1+simclr": 256,
        "supcon+simclr": 256,
        "default": 50,
        "jepa": 64,
    }[train_mode]
    LR = {
        "default": 0.0005,
        "simclr": 0.00075 * np.sqrt(BATCH_SIZE),
        "supcon": 0.00065 * np.sqrt(BATCH_SIZE),
        "supconL1": 0.00065 * np.sqrt(BATCH_SIZE),
        "supconL1+simclr": 0.00065 * np.sqrt(BATCH_SIZE),
        "supcon+simclr": 0.00065 * np.sqrt(BATCH_SIZE),
        "multitask": 0.00065 * np.sqrt(BATCH_SIZE),
        "jepa": 3e-4,
    }[train_mode]

    print(
        f"\n--- {train_mode.upper()} Training ({model_type.upper()}) on {dataset_id} ---"
    )
    if model_type == "gru":
        print(f"{d=} {gru_hidden_dim=} {gru_layers=}")
    elif model_type == "lstm":
        print(f"{d=} {lstm_hidden_dim=} {lstm_layers=}")
    elif model_type in ["cnn1d", "1dcnn", "cnn"]:
        print(f"{d=} {cnn_layers=} {cnn_kernel_size=}")
    else:
        print(f"{d=} {d_hidden=} {n_type_emb=} {cnn_kernel_size=}")
    if fold is not None:
        suffix = "f" if cv_mode == "kfold" else "loeo"
        print(f"{suffix.upper()}: {fold}")
        save_path = f"{save_path}-{suffix}{fold}"

    X_train, y_train, X_val, y_val, _, _ = get_preprocessed_data(
        DATA_PATH,
        dataset_id,
        window_size=WINDOW_SIZE,
        fold=fold,
        n_splits=n_splits,
        mode=cv_mode,
        loaded_data=loaded_data,
    )

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model
    model = build_model(
        n_sensors=X_train.shape[2],
        n_types=N_TYPES,
        window_size=WINDOW_SIZE,
        model_type=model_type,
        d=d,
        d_hidden=d_hidden,
        n_type_emb=n_type_emb,
        cnn_kernel_size=cnn_kernel_size,
        gru_hidden_dim=gru_hidden_dim,
        gru_layers=gru_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layers=lstm_layers,
        cnn_layers=cnn_layers,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion_mse = torch.nn.MSELoss()
    scaler = GradScaler("cuda")

    target_model = None
    if train_mode == "jepa":
        target_model = build_model(
            n_sensors=X_train.shape[2],
            n_types=N_TYPES,
            window_size=WINDOW_SIZE,
            model_type=model_type,
            d=d,
            d_hidden=d_hidden,
            n_type_emb=n_type_emb,
            cnn_kernel_size=cnn_kernel_size,
            gru_hidden_dim=gru_hidden_dim,
            gru_layers=gru_layers,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            cnn_layers=cnn_layers,
        ).to(device)
        target_model.load_state_dict(model.state_dict())
        for param in target_model.parameters():
            param.requires_grad = False

    best_val_loss = float("inf")
    best_path = f"{save_path}-best.pth"
    last_path = f"{save_path}-last.pth"

    with open(f"{save_path}.meta", "wb") as file:
        pickle.dump(
            dict(
                model_type=model_type,
                d=d,
                d_hidden=d_hidden,
                n_type_emb=n_type_emb,
                cnn_kernel_size=cnn_kernel_size,
                gru_hidden_dim=gru_hidden_dim,
                gru_layers=gru_layers,
                lstm_hidden_dim=lstm_hidden_dim,
                lstm_layers=lstm_layers,
                cnn_layers=cnn_layers,
            ),
            file,
        )

    with open(f"{save_path}.hyperparams", "wb") as file:
        pickle.dump(
            {
                "dataset_id": dataset_id,
                "epochs": epochs,
                "train_mode": train_mode,
                "alpha": alpha,
                "margin": margin,
                "ema_decay": ema_decay,
                "cov_scale": cov_scale,
                "simclr_scale": simclr_scale,
                "cnn_kernel_size": cnn_kernel_size,
                "d": d,
                "d_hidden": d_hidden,
                "n_type_emb": n_type_emb,
                "model_type": model_type,
                "gru_hidden_dim": gru_hidden_dim,
                "gru_layers": gru_layers,
                "lstm_hidden_dim": lstm_hidden_dim,
                "lstm_layers": lstm_layers,
                "cnn_layers": cnn_layers,
            },
            file,
        )

    t_start = perf_counter()

    patience = max_patience = 10
    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = (
                    batch_x.to(device, non_blocking=True),
                    batch_y.to(device, non_blocking=True),
                )

                # Augmented views
                x1 = augment_batch(batch_x)
                x2 = augment_batch(batch_x)

                optimizer.zero_grad()

                with autocast("cuda"):
                    if train_mode == "simclr":
                        z1 = model.forward_contrastive(x1, type_idx)
                        z2 = model.forward_contrastive(x2, type_idx)
                        loss = nt_xent_loss(z1, z2)
                    elif train_mode == "multitask":
                        z1 = model.forward_contrastive(x1, type_idx)
                        z2 = model.forward_contrastive(x2, type_idx)
                        loss_contrastive = nt_xent_loss(z1, z2)
                        y_pred = model(batch_x, type_idx)
                        loss_mse = criterion_mse(y_pred, batch_y)
                        loss = loss_contrastive + alpha * loss_mse
                    elif train_mode == "supcon":
                        z1 = model.forward_contrastive(x1, type_idx)
                        z2 = model.forward_contrastive(x2, type_idx)
                        z = torch.cat([z1, z2], dim=0)
                        y = torch.cat([batch_y, batch_y], dim=0)
                        loss = supcon_loss(z, y, margin=margin)
                    elif train_mode in ["supconL1", "supconL1+simclr"]:
                        z1 = model.forward_contrastive(x1, type_idx)
                        z2 = model.forward_contrastive(x2, type_idx)
                        z = torch.cat([z1, z2], dim=0)
                        y = torch.cat([batch_y, batch_y], dim=0)

                        # from sklearn.linear_model import Ridge

                        # X_np = z.detach().cpu().numpy()
                        # y_np = y.detach().cpu().numpy()
                        # reg = Ridge(alpha=1.0, fit_intercept=True).fit(X_np, y_np)
                        # stds = np.maximum(np.std(X_np, axis=0), 1e-2)
                        # scaled_coefs = np.abs(reg.coef_) / stds
                        # weights = torch.FloatTensor(sparsemax_np(scaled_coefs)).to(
                        #     device
                        # )

                        weights = torch.ones(z.shape[1]).to(z.device)
                        loss_base = supcon_l1_loss(
                            z, y, weights, margin=margin, cov_scale=cov_scale
                        )
                        raw_term = torch.mean(1.0 / (z.abs().pow(2) + 1.0))
                        term = torch.where(
                            raw_term < 0.2, torch.tensor(0.0, device=device), raw_term
                        )
                        loss = loss_base + alpha * term

                        if train_mode == "supconL1+simclr":
                            loss_simclr = nt_xent_loss(z1, z2)
                            loss = loss + simclr_scale * loss_simclr
                    elif train_mode == "supcon+simclr":
                        z1 = model.forward_contrastive(x1, type_idx)
                        z2 = model.forward_contrastive(x2, type_idx)
                        z = torch.cat([z1, z2], dim=0)
                        y = torch.cat([batch_y, batch_y], dim=0)

                        loss_base = supcon_l2_loss(
                            z, y, margin=margin, cov_scale=cov_scale
                        )
                        raw_term = torch.mean(1.0 / (z.abs().pow(2) + 1.0))
                        term = torch.where(
                            raw_term < 0.2, torch.tensor(0.0, device=device), raw_term
                        )
                        loss = loss_base + alpha * term

                        loss_simclr = nt_xent_loss(z1, z2)
                        loss = loss + simclr_scale * loss_simclr
                    elif train_mode == "jepa":
                        z1 = model.forward_contrastive(x1, type_idx)
                        p1 = model.predictor(z1)
                        with torch.no_grad():
                            z2 = target_model.forward_contrastive(x2, type_idx)
                        p1 = torch.nn.functional.normalize(p1, dim=1)
                        z2 = torch.nn.functional.normalize(z2, dim=1)
                        loss = torch.nn.functional.mse_loss(p1, z2)
                    elif train_mode == "default":
                        y_pred = model(batch_x, type_idx)
                        loss = criterion_mse(y_pred, batch_y)
                    else:
                        raise RuntimeError

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if train_mode == "jepa":
                    # EMA update
                    with torch.no_grad():
                        for param_q, param_k in zip(
                            model.parameters(), target_model.parameters()
                        ):
                            param_k.data = param_k.data * ema_decay + param_q.data * (
                                1.0 - ema_decay
                            )

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad(), autocast("cuda"):
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = (
                        batch_x.to(device, non_blocking=True),
                        batch_y.to(device, non_blocking=True),
                    )

                    if train_mode == "simclr":
                        x1 = augment_batch(batch_x)
                        x2 = augment_batch(batch_x)
                        z1 = model.forward_contrastive(x1, type_idx)
                        z2 = model.forward_contrastive(x2, type_idx)
                        loss = nt_xent_loss(z1, z2)
                    elif train_mode == "multitask":
                        x1 = augment_batch(batch_x)
                        x2 = augment_batch(batch_x)
                        z1 = model.forward_contrastive(x1, type_idx)
                        z2 = model.forward_contrastive(x2, type_idx)
                        loss_contrastive = nt_xent_loss(z1, z2)
                        y_pred = model(batch_x, type_idx)
                        loss_mse = criterion_mse(y_pred, batch_y)
                        loss = loss_contrastive + alpha * loss_mse
                    elif train_mode == "supcon":
                        x1 = augment_batch(batch_x)
                        x2 = augment_batch(batch_x)
                        z1 = model.forward_contrastive(x1, type_idx)
                        z2 = model.forward_contrastive(x2, type_idx)
                        z = torch.cat([z1, z2], dim=0)
                        y = torch.cat([batch_y, batch_y], dim=0)
                        loss = supcon_loss(z, y, margin=margin)
                    elif train_mode in ["supconL1", "supconL1+simclr"]:
                        x1 = augment_batch(batch_x)
                        x2 = augment_batch(batch_x)
                        z1 = model.forward_contrastive(x1, type_idx)
                        z2 = model.forward_contrastive(x2, type_idx)
                        z = torch.cat([z1, z2], dim=0)
                        y = torch.cat([batch_y, batch_y], dim=0)

                        from sklearn.linear_model import Ridge

                        X_np = z.detach().cpu().numpy()
                        y_np = y.detach().cpu().numpy()
                        reg = Ridge(alpha=1.0, fit_intercept=True).fit(X_np, y_np)
                        stds = np.maximum(np.std(X_np, axis=0), 1e-2)
                        scaled_coefs = np.abs(reg.coef_) / stds
                        weights = torch.FloatTensor(sparsemax_np(scaled_coefs)).to(
                            device
                        )

                        loss_base = supcon_l1_loss(
                            z, y, weights, margin=margin, cov_scale=cov_scale
                        )
                        raw_term = torch.mean(1.0 / (z.abs().pow(2) + 1.0))
                        term = torch.where(
                            raw_term < 0.2, torch.tensor(0.0, device=device), raw_term
                        )
                        loss = loss_base + alpha * term

                        if train_mode == "supconL1+simclr":
                            loss_simclr = nt_xent_loss(z1, z2)
                            loss = loss + simclr_scale * loss_simclr
                    elif train_mode == "supcon+simclr":
                        x1 = augment_batch(batch_x)
                        x2 = augment_batch(batch_x)
                        z1 = model.forward_contrastive(x1, type_idx)
                        z2 = model.forward_contrastive(x2, type_idx)
                        z = torch.cat([z1, z2], dim=0)
                        y = torch.cat([batch_y, batch_y], dim=0)

                        loss_base = supcon_l2_loss(
                            z, y, margin=margin, cov_scale=cov_scale
                        )
                        raw_term = torch.mean(1.0 / (z.abs().pow(2) + 1.0))
                        term = torch.where(
                            raw_term < 0.2, torch.tensor(0.0, device=device), raw_term
                        )
                        loss = loss_base + alpha * term

                        loss_simclr = nt_xent_loss(z1, z2)
                        loss = loss + simclr_scale * loss_simclr
                    elif train_mode == "jepa":
                        x1 = augment_batch(batch_x)
                        x2 = augment_batch(batch_x)
                        z1 = model.forward_contrastive(x1, type_idx)
                        p1 = model.predictor(z1)
                        with torch.no_grad():
                            z2 = target_model.forward_contrastive(x2, type_idx)
                        p1 = torch.nn.functional.normalize(p1, dim=1)
                        z2 = torch.nn.functional.normalize(z2, dim=1)
                        loss = torch.nn.functional.mse_loss(p1, z2)
                    elif train_mode == "default":
                        y_pred = model(batch_x, type_idx)
                        loss = criterion_mse(y_pred, batch_y)
                    else:
                        raise RuntimeError

                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_path)
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4e} - Val Loss: {avg_val_loss:.4e} (Saved to {best_path})"
                )
                patience = max_patience
            else:
                patience -= 1
                if (epoch + 1) % 5 == 0:
                    print(
                        f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4e} - Val Loss: {avg_val_loss:.4e}"
                    )
            if patience <= 0:
                break

    except KeyboardInterrupt:
        print("Training interrupted.")

    elapsed = perf_counter() - t_start
    with open(f"{save_path}-time.txt", "w") as file:
        file.write(str(elapsed))
        file.write("\n")
    torch.save(model.state_dict(), last_path)

    # Load best model checkpoint for evaluation
    try:
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
            print(f"Loaded best model from {best_path} for DT evaluation.")
        else:
            print(f"Best model path {best_path} not found. Evaluating last state.")
    except Exception as e:
        print(f"Error loading best model from {best_path}: {e}. Evaluating last state.")

    if train_mode != "default":
        dt_rmse = evaluate_dt_performance(
            model, train_loader, val_loader, type_idx, device
        )
        print(
            f"Training complete after {elapsed / 60:.2f} minutes. Best validation loss: {best_val_loss:.4f} - Validation DT RMSE: {dt_rmse:.4f}"
        )
        return dt_rmse
    else:
        return best_val_loss


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save", type=str, default="model", help="File name base for the model"
    )
    parser.add_argument(
        "--dataset", type=str, default="FD001", help="Dataset ID (e.g., FD001)"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--mode",
        type=str,
        default="simclr",
        choices=[
            "simclr",
            "multitask",
            "supcon",
            "supconL1",
            "supconL1+simclr",
            "supcon+simclr",
            "jepa",
            "default",
        ],
        help="Training mode",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Loss weight (default: 0.0 for supconL1 and supcon modes, 1e-3 for multitask)",
    )
    parser.add_argument(
        "--margin", type=float, default=10.0, help="RUL margin for supcon mode"
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.99, help="EMA decay for JEPA mode"
    )
    parser.add_argument(
        "--cov-scale",
        type=float,
        default=1.0,
        help="Covariance loss multiplier for supconL1/supcon modes",
    )
    parser.add_argument(
        "--simclr-scale",
        type=float,
        default=1.0,
        help="Scaling parameter for simclr loss in supconL1+simclr / supcon+simclr mode",
    )
    parser.add_argument(
        "--dim", type=int, default=16, help="Projection and GRU dimension"
    )
    parser.add_argument("--hdim", type=int, default=4, help="Hidden FiLM dimension")
    parser.add_argument("--n-type-emb", type=int, default=8, help="Embedding dimension")
    parser.add_argument(
        "--cnn-kernel-size",
        type=int,
        default=0,
        help="Kernel size for 1D CNN individually on each sensor (0 to disable)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="thgnn",
        choices=["thgnn", "gru", "lstm", "cnn1d"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--gru-hidden-dim",
        type=int,
        default=64,
        help="GRU hidden dimensions",
    )
    parser.add_argument(
        "--gru-layers",
        type=int,
        default=1,
        help="Number of GRU layers",
    )
    parser.add_argument(
        "--lstm-hidden-dim",
        type=int,
        default=64,
        help="LSTM hidden dimensions",
    )
    parser.add_argument(
        "--lstm-layers",
        type=int,
        default=1,
        help="Number of LSTM layers",
    )
    parser.add_argument(
        "--cnn-layers",
        type=int,
        default=2,
        help="Number of 1D CNN layers",
    )
    parser.add_argument("--fold", type=int, default=None, help="Fold index for K-fold")
    parser.add_argument("--n-folds", type=int, default=5, help="Total number of folds")
    parser.add_argument("--loeo", action="store_true", help="Leave-One-Engine-Out mode")
    args = parser.parse_args()

    mode = "loeo" if args.loeo else "kfold"

    alpha = args.alpha
    if alpha is None:
        alpha = (
            0.0
            if args.mode in ["supconL1", "supconL1+simclr", "supcon+simclr"]
            else 1e-3
        )

    train(
        save_path=args.save,
        dataset_id=args.dataset,
        epochs=args.epochs,
        train_mode=args.mode,
        alpha=alpha,
        margin=args.margin,
        d=args.dim,
        d_hidden=args.hdim,
        n_type_emb=args.n_type_emb,
        ema_decay=args.ema_decay,
        fold=args.fold,
        n_splits=args.n_folds,
        cv_mode=mode,
        cov_scale=args.cov_scale,
        simclr_scale=args.simclr_scale,
        cnn_kernel_size=args.cnn_kernel_size,
        model_type=args.model_type,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_layers=args.gru_layers,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        cnn_layers=args.cnn_layers,
    )
