import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
import sys
from typing import Annotated, Literal, NoReturn

import numpy as np
import torch
from rungrid.cli import CLI
from rungrid.experiment import (
    STATE_ARG_RECORD,
    Experiment,
    Sampler,
    Trial,
    VarNamespace,
    step_method,
)
from rungrid.version import version_from_git_commit_hash
from sklearn.tree import DecisionTreeRegressor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, TensorDataset

from analise_db_iasmin_2026.thgnn_contrastive.model import (
    THGNN,
    Baseline1DCNN,
    BaselineGRU,
    BaselineLSTM,
    build_model,
)
from analise_db_iasmin_2026.thgnn_contrastive.preprocess import (
    ACTIVE_SENSOR_TYPES,
    get_preprocessed_data,
)
from analise_db_iasmin_2026.thgnn_contrastive.scoring import (
    calculate_score,
)

TYPE_IDX_LIST = ACTIVE_SENSOR_TYPES
N_TYPES = len(set(ACTIVE_SENSOR_TYPES))

type ModelType = THGNN | Baseline1DCNN | BaselineGRU | BaselineLSTM


def evaluate_dt_performance(model, train_loader, val_loader, type_idx, device):
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

    # Train Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)

    # Predict on validation set
    y_pred = dt.predict(X_val)

    rmse = calculate_score(y_val, y_pred)
    if np.isnan(rmse) or np.isinf(rmse):
        return float("inf")
    return rmse


def augment_batch(x, trial: Trial):
    """
    Apply augmentations to a batch of time-series windows (B, L, N).
    """

    # 1. Jitter (Gaussian noise)
    noise = torch.randn_like(x) * trial.v.aug_jitter
    x_aug = x + noise

    # 2. Scaling
    scaling_factor = (
        torch.randn(x.size(0), 1, x.size(2)).to(x.device) * trial.v.aug_scaling + 1.0
    )
    x_aug = x_aug * scaling_factor

    return x_aug


def supcon_loss(
    z,
    y,
    margin: float,
    margin_mode: Literal["margin_relative", "margin_absolute"],
    temperature: float = 0.1,
):
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
    if margin_mode == "margin_absolute":
        label_dist = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))  # (N, N)
    elif margin_mode == "margin_relative":
        label_dist = torch.abs(
            torch.log1p(y.unsqueeze(0)) - torch.log1p(y.unsqueeze(1))
        )
    else:
        raise RuntimeError

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


class TrainModelExperiment(Experiment):
    @lru_cache
    def logger(self) -> logging.Logger:
        log = logging.getLogger(__name__)
        sh = logging.StreamHandler(sys.stderr)
        log.addHandler(self.get_logging_handler())
        log.addHandler(sh)
        log.setLevel(logging.DEBUG)
        sh.setFormatter(
            logging.Formatter(
                "%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S"
            )
        )
        return log

    def variables(self, v: VarNamespace, s: Sampler) -> None:
        opts = CLI.get_instance().get_options()
        arch = opts["arch"].lower()
        assert arch in ["thgnn", "gru", "lstm", "cnn"]

        v.arch = arch
        v.dim = s.power_of_two(3, 5)
        v.lr = s.log_uniform(5e-4, 5e-1)
        v.batch_size = 128
        v.patience = 10
        v.epochs = 50
        v.aug_jitter = 0.02
        v.aug_scaling = 0.1

        with s.branch(arch):
            match arch:
                case "thgnn":
                    v.hdim = s.power_of_two(1, 7)
                    v.n_type_emb = s.power_of_two(2, 4)
                    v.layers = 0
                    v.kernel_size = 0
                case "gru":
                    v.hdim = s.power_of_two(5, 7)
                    v.layers = s.integer(1, 3)
                    v.kernel_size = 0
                    v.n_type_emb = 0
                case "lstm":
                    v.hdim = s.power_of_two(5, 7)
                    v.layers = s.integer(1, 3)
                    v.kernel_size = 0
                    v.n_type_emb = 0
                case "cnn":
                    v.kernel_size = s.categorical([3, 5, 7])
                    v.layers = s.integer(1, 4)
                    v.n_type_emb = 0
                    v.hdim = 0

        v.margin_mode = s.categorical(["margin_absolute", "margin_relative"])
        with s.branch(v.margin_mode):  # type: ignore
            match v.margin_mode:
                case "margin_absolute":
                    v.margin = s.integer(3, 50)
                case "margin_relative":
                    v.margin = s.log_uniform(1e-2, 1e2)

    def get_datasets(
        self, fold: int, key: Literal["train_on", "test_on"]
    ) -> list[np.ndarray]:
        opts = CLI.get_instance().get_options()
        new_cmapss = "new_cmapss" in opts
        cmapss_data = Path(__file__).parent.parent.parent.parent / "CMAPSSData"
        assert not new_cmapss
        train_on = opts[key]
        return list(get_preprocessed_data(cmapss_data, train_on, fold=fold, n_splits=5))

    def get_train_datasets(
        self,
        fold: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return tuple(self.get_datasets(fold, "train_on")[:4])  # type: ignore

    @lru_cache
    def get_train_dataloaders(
        self, fold: int, batch_size: int
    ) -> tuple[DataLoader, DataLoader]:
        X_train, y_train, X_val, y_val = self.get_train_datasets(fold)
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        return train_loader, val_loader

    @lru_cache
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def first_step(self) -> Callable:
        return self.start_training

    def version(self) -> str:
        return version_from_git_commit_hash(type(self))

    @step_method()
    def start_training(self, trial: Trial) -> NoReturn:
        trial.add_tags([trial.v.arch, trial.v.margin_mode])
        self.exit_next_step(self.train_fold, fold=0)

    @step_method()
    def train_fold(self, trial: Trial, fold: int) -> NoReturn:
        X_train, *_ = self.get_train_datasets(fold)
        model = build_model(
            n_sensors=X_train.shape[2],
            n_types=N_TYPES,
            window_size=50,
            model_type=trial.v.arch,
            d=trial.v.dim,
            d_hidden=trial.v.hdim,
            n_type_emb=trial.v.n_type_emb,
            cnn_kernel_size=trial.v.kernel_size,
            gru_hidden_dim=trial.v.hdim,
            gru_layers=trial.v.layers,
            lstm_hidden_dim=trial.v.hdim,
            lstm_layers=trial.v.layers,
            cnn_layers=trial.v.layers,
        ).to(self.device())
        del X_train
        train_dl, val_dl = self.get_train_dataloaders(fold, trial.v.batch_size)

        self.exit_next_step(
            self.train_one_epoch,
            last=model,
            fold=fold,
            epoch=0,
            best=None,
            best_loss=float("inf"),
            train_loader=train_dl,
            val_loader=val_dl,
            patience=trial.v.patience,
            optimizer=AdamW(model.parameters(), lr=trial.v.lr),
        )

    @step_method()
    def train_one_epoch(
        self,
        trial: Trial,
        fold: Annotated[int, STATE_ARG_RECORD],
        last: ModelType,
        best_loss: Annotated[float, STATE_ARG_RECORD],
        epoch: Annotated[int, STATE_ARG_RECORD],
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        patience: int,
    ) -> NoReturn:
        best_path = f"/mnt/sghdd/adbiasminfilestore/{trial.uuid}.pth"
        type_idx = torch.tensor(TYPE_IDX_LIST).to(self.device())
        scaler = GradScaler("cuda")
        last.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = (
                batch_x.to(self.device(), non_blocking=True),
                batch_y.to(self.device(), non_blocking=True),
            )

            # Augmented views
            x1 = augment_batch(batch_x, trial)
            x2 = augment_batch(batch_x, trial)

            optimizer.zero_grad()

            with autocast("cuda"):
                z1 = last.forward_contrastive(x1, type_idx)
                z2 = last.forward_contrastive(x2, type_idx)
                z = torch.cat([z1, z2], dim=0)
                y = torch.cat([batch_y, batch_y], dim=0)
                loss = supcon_loss(z, y, trial.v.margin, trial.v.margin_mode)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        trial.append_trace("train_loss", float(train_loss), fold=fold, epoch=epoch)

        last.eval()
        val_loss = 0
        with torch.inference_mode(), autocast("cuda"):
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = (
                    batch_x.to(self.device(), non_blocking=True),
                    batch_y.to(self.device(), non_blocking=True),
                )

                x1 = augment_batch(batch_x, trial)
                x2 = augment_batch(batch_x, trial)
                z1 = last.forward_contrastive(x1, type_idx)
                z2 = last.forward_contrastive(x2, type_idx)
                z = torch.cat([z1, z2], dim=0)
                y = torch.cat([batch_y, batch_y], dim=0)
                loss = supcon_loss(z, y, trial.v.margin, trial.v.margin_mode)
                val_loss += loss.item()

        trial.append_trace("val_loss", float(val_loss), fold=fold, epoch=epoch)

        if val_loss < best_loss:
            torch.save(last, best_path)
            best_loss = val_loss
            self.logger().info(
                f"Epoch {epoch + 1}/{trial.v.epochs} - Train Loss: {train_loss:.4e} - Val Loss: {val_loss:.4e} (best!)"
            )
            patience = trial.v.patience
        else:
            patience -= 1
            if (epoch + 1) % 5 == 0:
                self.logger().info(
                    f"Epoch {epoch + 1}/{trial.v.epochs} - Train Loss: {train_loss:.4e} - Val Loss: {val_loss:.4e}"
                )

        if epoch >= trial.v.epochs - 1 or patience <= 0:
            self.exit_next_step(
                self.score_model_on_fold,
                fold=fold,
                train_loader=train_loader,
                val_loader=val_loader,
                model=torch.load(best_path, weights_only=False),
            )
        else:
            epoch += 1
            self.exit_next_step(self.train_one_epoch, fill_locals=True)

    @step_method()
    def score_model_on_fold(
        self,
        trial: Trial,
        model: Annotated[ModelType, STATE_ARG_RECORD],
        fold: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> NoReturn:
        type_idx = torch.tensor(TYPE_IDX_LIST).to(self.device())
        score = evaluate_dt_performance(
            model, train_loader, val_loader, type_idx, self.device()
        )
        trial.append_trace("dt_score", score, fold=fold)
        self.logger().info("fold=%s score=%.3f", fold, score)
        if fold in [0, 1, 2, 3]:
            self.exit_next_step(self.train_fold, fold=fold + 1)
        else:
            fold_scores = list(trial.trace_values("dt_score"))
            if len(fold_scores) != 5:
                self.logger().warning(
                    "Somehow less than 5 folds registered?",
                    list(trial.trace_all("dt_score")),
                )
            mean = sum(fold_scores) / len(fold_scores)
            self.logger().info(
                "%i folds, mean=%.3f, std=%.3f",
                len(fold_scores),
                mean,
                float(np.std(fold_scores)),
            )
            self.exit_done(mean)
