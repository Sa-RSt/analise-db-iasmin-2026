import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import numpy as np
from pandas import isna
import torch
from rungrid.experiment import (
    STATE_ARG_RECORD,
    Experiment,
    Sampler,
    Trial,
    VarNamespace,
    fixture,
    step_method,
)
from rungrid.version import version_from_git_commit_hash, version_from_most_recent_mtime
from sklearn.tree import DecisionTreeRegressor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, TensorDataset

from analise_db_iasmin_2026.environment import Environment
from analise_db_iasmin_2026.options import get_options
from analise_db_iasmin_2026.thgnn_contrastive.model import (
    THGNN,
    Baseline1DCNN,
    BaselineGRU,
    BaselineLSTM,
    build_model,
)
from analise_db_iasmin_2026.thgnn_contrastive.n_cmapss import (
    COLUMNS_TYPES,
    FoldLoader,
    get_n_cmapss_folds,
)
from analise_db_iasmin_2026.thgnn_contrastive.preprocess import (
    ACTIVE_SENSOR_TYPES,
    get_preprocessed_data,
)
from analise_db_iasmin_2026.thgnn_contrastive.scoring import (
    calculate_score,
)

TYPE_IDX_LIST_CMAPSS = ACTIVE_SENSOR_TYPES
N_TYPES_CMAPSS = len(set(ACTIVE_SENSOR_TYPES))
TYPE_IDX_LIST_NEW_CMAPSS = [x[1] for x in COLUMNS_TYPES]
N_TYPES_NEW_CMAPSS = len(set(TYPE_IDX_LIST_NEW_CMAPSS))

type ModelType = THGNN | Baseline1DCNN | BaselineGRU | BaselineLSTM


def evaluate_e2e_performance(model, val_loader, type_idx, device):
    model.eval()
    y_val = []
    y_pred = []

    type_idx = type_idx.to(device)

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            y_val.append(batch_y.cpu().numpy())
            y_pred.append(model(batch_x, type_idx).cpu().numpy())

    y_val = np.concatenate(y_val, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    s = calculate_score(y_val, y_pred)
    if np.isnan(s) or np.isinf(s):
        return float("inf")
    return s


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

    s = calculate_score(y_val, y_pred)
    if np.isnan(s) or np.isinf(s):
        return float("inf")
    return s


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
    def __hash__(self) -> int:
        return 0

    @fixture()
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
        opts = get_options()
        arch = opts["arch"].lower()
        archs = ["thgnn", "gru", "lstm", "cnn"]
        if arch == "random":
            arch = s.random(archs)
        v.arch = arch
        assert v.arch in archs

        mode = opts["mode"].lower()
        modes = ["e2e", "supcon"]
        if mode == "random":
            mode = s.random(modes)
        v.mode = mode
        assert v.mode in modes

        folding = opts.get("folds", "yes").lower()
        assert folding in ["yes", "no"]

        with s.branch(v.mode):
            v.folding = folding
            v.dim = s.power_of_two(3, 5)
            v.lr = s.log_uniform(1e-5, 1e-1)
            v.batch_size = 50 if mode == "e2e" else 128
            v.patience = 10
            v.epochs = int(opts.get("epochs", "") or 50)
            if mode == "supcon":
                v.aug_jitter = s.ordinal([0.0, 0.0002, 0.002, 0.02, 0.2])
                v.aug_scaling = s.ordinal([0.0, 0.0001, 0.001, 0.01, 0.1])
            else:
                v.aug_scaling = v.aug_jitter = 0
            if opts["train_on"] == "olden":
                v.ds_name = s.random(["FD001", "FD002", "FD003", "FD004"])
            else:
                v.ds_name = opts["train_on"]

            with s.branch(v.arch):
                match v.arch:
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
                        v.kernel_size = s.ordinal([3, 5, 7])
                        v.layers = s.integer(1, 4)
                        v.n_type_emb = 0
                        v.hdim = 0

            if v.mode == "supcon":
                v.margin_mode = s.random(
                    [
                        "margin_absolute",
                        # "margin_relative",
                    ]
                )
                with s.branch(v.margin_mode):  # type: ignore
                    match v.margin_mode:
                        case "margin_absolute":
                            v.margin = s.integer(2, 50)
                        case "margin_relative":
                            v.margin = s.log_uniform(1e-2, 1e2)
            else:
                v.margin_mode = "e2e"
                v.margin = 0

    def get_train_datasets(self, ds_name: str, fold: int) -> list[np.ndarray]:
        if self.is_training_on_new_cmapss(ds_name):
            raise RuntimeError
        else:
            cmapss_data = Path(__file__).parent.parent.parent.parent / "CMAPSSData"
            return list(
                get_preprocessed_data(cmapss_data, ds_name, fold=fold, n_splits=5)[:4]
            )

    @fixture()
    def get_n_cmapss_fold_loader(self, ds_name: str) -> FoldLoader:
        return get_n_cmapss_folds(ds_name, self.logger())

    @fixture()
    def get_fold_count(self, ds_name: str) -> int:
        if self.is_training_on_new_cmapss(ds_name):
            return self.get_n_cmapss_fold_loader(ds_name).fold_count()
        else:
            return 5

    def get_train_dataloaders(
        self, fold: int, batch_size: int, ds_name: str
    ) -> tuple[DataLoader, DataLoader]:
        if self.is_training_on_new_cmapss(ds_name):
            fold_loader = self.get_n_cmapss_fold_loader(ds_name)
            return fold_loader.load_train_val(fold, batch_size)
        else:
            X_train, y_train, X_val, y_val = self.get_train_datasets(ds_name, fold)
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train), torch.FloatTensor(y_train)
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
            )

            val_dataset = TensorDataset(
                torch.FloatTensor(X_val), torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
            )
            return train_loader, val_loader

    @fixture()
    def is_training_on_new_cmapss(self, ds_name: str) -> bool:
        return not ds_name.startswith("FD")

    @fixture()
    def get_sensor_type_count(self, ds_name: str) -> int:
        if self.is_training_on_new_cmapss(ds_name):
            return N_TYPES_NEW_CMAPSS
        return N_TYPES_CMAPSS

    @fixture()
    def get_sensor_type_indices_list(self, ds_name: str) -> list[int]:
        if self.is_training_on_new_cmapss(ds_name):
            return TYPE_IDX_LIST_NEW_CMAPSS
        return TYPE_IDX_LIST_CMAPSS

    @fixture()
    def get_sensor_count(self, ds_name: str) -> int:
        if self.is_training_on_new_cmapss(ds_name):
            return len(COLUMNS_TYPES)
        X_train, *_ = self.get_train_datasets(ds_name, 0)
        return X_train.shape[2]

    @fixture()
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def first_step(self) -> Callable:
        return self.start_training

    def version(self) -> str:
        try:
            return version_from_git_commit_hash(type(self))
        except Exception:
            return version_from_most_recent_mtime(type(self))

    @step_method()
    def start_training(self, trial: Trial) -> NoReturn:
        trial.add_tags(
            [trial.v.arch, trial.v.margin_mode, trial.v.ds_name, trial.v.mode]
        )
        if trial.v.mode != "e2e":
            trial.add_tag("not_e2e")
        self.logger().info(
            "starting experiment with %s",
            json.dumps({k: v for k, v in trial.v._variables_dict.items()}),
        )
        env = Environment.get_environment()
        fs = env.get_file_store_path()
        fs_zip = fs.parent / "zip"
        if not env.is_laptop() and len(list(fs.iterdir())) >= 90:
            zf = fs_zip / f"{trial.uuid}.zip"
            cmd = ["zip", "-r", str(zf.absolute()), fs]
            ret = subprocess.call(cmd)
            self.logger().info("ran command %s: %s", repr(cmd), str(ret))
            if ret == 0:
                shutil.rmtree(fs)
                fs.mkdir()
        self.exit_next_step(self.train_fold, fold=0)

    @step_method()
    def train_fold(self, trial: Trial, fold: int) -> NoReturn:
        model = build_model(
            n_sensors=self.get_sensor_count(trial.v.ds_name),
            n_types=self.get_sensor_type_count(trial.v.ds_name),
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

        self.exit_next_step(
            self.train_one_epoch,
            last=model,
            fold=fold,
            epoch=0,
            best_loss=float("inf"),
            patience=trial.v.patience,
            optimizer=AdamW(model.parameters(), lr=trial.v.lr),
        )

    @contextmanager
    def null_context(self):
        yield

    def calc_loss(
        self, trial: Trial, last: ModelType, batch_x, batch_y, type_idx, ctx_factory
    ) -> torch.Tensor:
        if trial.v.mode == "supcon":
            x1 = augment_batch(batch_x, trial)
            x2 = augment_batch(batch_x, trial)

            with ctx_factory():
                z1 = last.forward_contrastive(x1, type_idx)
                z2 = last.forward_contrastive(x2, type_idx)
                z = torch.cat([z1, z2], dim=0)
                y = torch.cat([batch_y, batch_y], dim=0)
                loss = supcon_loss(z, y, trial.v.margin, trial.v.margin_mode)
        elif trial.v.mode == "e2e":
            with ctx_factory():
                y_pred = last(batch_x, type_idx)
                loss = torch.nn.functional.mse_loss(y_pred, batch_y)
        else:
            raise RuntimeError
        return loss

    @step_method()
    def train_one_epoch(
        self,
        trial: Trial,
        fold: Annotated[int, STATE_ARG_RECORD],
        last: ModelType,
        best_loss: Annotated[float, STATE_ARG_RECORD],
        epoch: Annotated[int, STATE_ARG_RECORD],
        optimizer: Optimizer,
        patience: int,
    ) -> NoReturn:
        fs = Environment.get_environment().get_file_store_path()
        best_path = fs / f"{trial.uuid}.pth"
        type_idx = torch.tensor(self.get_sensor_type_indices_list(trial.v.ds_name)).to(
            self.device()
        )
        train_loader, val_loader = self.get_train_dataloaders(
            fold, trial.v.batch_size, trial.v.ds_name
        )
        scaler = GradScaler("cuda")
        last.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = (
                batch_x.to(self.device(), non_blocking=True),
                batch_y.to(self.device(), non_blocking=True),
            )
            optimizer.zero_grad()
            loss = self.calc_loss(
                trial, last, batch_x, batch_y, type_idx, lambda: autocast("cuda")
            )
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

                loss = self.calc_loss(
                    trial, last, batch_x, batch_y, type_idx, self.null_context
                )
                val_loss += loss.item()

        trial.append_trace("val_loss", float(val_loss), fold=fold, epoch=epoch)

        if not np.isfinite(float(train_loss)) or not np.isfinite(float(val_loss)):
            self.exit_prune("failed to converge")

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
                self.score_model_on_fold, fold=fold, model_path=best_path
            )
        else:
            epoch += 1
            self.exit_next_step(self.train_one_epoch, fill_locals=True)

    def evaluate_performance(
        self, trial: Trial, model: ModelType, train_loader, val_loader, type_idx, device
    ):
        if trial.v.mode == "supcon":
            return evaluate_dt_performance(
                model, train_loader, val_loader, type_idx, device
            )
        elif trial.v.mode == "e2e":
            return evaluate_e2e_performance(model, val_loader, type_idx, device)
        else:
            raise RuntimeError

    @step_method()
    def score_model_on_fold(
        self,
        trial: Trial,
        model_path: Annotated[Path, STATE_ARG_RECORD],
        fold: int,
    ) -> NoReturn:
        model = torch.load(model_path, weights_only=False)
        type_idx = torch.tensor(self.get_sensor_type_indices_list(trial.v.ds_name)).to(
            self.device()
        )
        train_loader, val_loader = self.get_train_dataloaders(
            fold, trial.v.batch_size, trial.v.ds_name
        )
        score = self.evaluate_performance(
            trial,
            model,
            train_loader,
            val_loader,
            type_idx,
            self.device(),
        )
        trial.append_trace("dt_score", score, fold=fold)
        self.logger().info("fold=%s score=%.3f", fold, score)
        if trial.v.folding == "yes" and fold < self.get_fold_count(trial.v.ds_name) - 1:
            self.exit_next_step(self.train_fold, fold=fold + 1)
        else:
            fold_scores = list(trial.trace_values("dt_score"))
            mean = sum(fold_scores) / len(fold_scores)
            self.logger().info(
                "%i folds, mean=%.3f, std=%.3f",
                len(fold_scores),
                mean,
                float(np.std(fold_scores) if len(fold_scores) > 1 else 0),
            )
            self.exit_done(mean)
