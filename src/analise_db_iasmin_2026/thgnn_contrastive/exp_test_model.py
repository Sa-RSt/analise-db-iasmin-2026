import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, NoReturn

import numpy as np
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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from analise_db_iasmin_2026.options import get_options
from analise_db_iasmin_2026.thgnn_contrastive.preprocess import (
    ACTIVE_SENSOR_TYPES,
    get_preprocessed_data,
)
from analise_db_iasmin_2026.thgnn_contrastive.scoring import (
    calculate_brier_score,
    calculate_cd_auc,
    calculate_concordance_index,
    calculate_mae,
    calculate_map,
    calculate_r2,
    calculate_rmse,
    calculate_score,
)

TYPE_IDX_LIST_CMAPSS = ACTIVE_SENSOR_TYPES


def resolve_backbone_config(opts: dict[str, str]) -> tuple[str, str, str]:
    """
    Resolve (arch, mode, ds_name) from options dictionary.
    Supports either:
      - 'backbone' or 'model' option in format '<ARCHITECTURE>-<MODE>-<DATASET>'
      - separate 'arch', 'mode', and 'dataset'/'ds_name'/'train_on' options
    """
    raw_backbone = opts.get("backbone") or opts.get("model") or opts.get("model_path")
    if raw_backbone:
        stem = Path(raw_backbone).stem
        parts = stem.split("-")
        if len(parts) == 3:
            arch, mode, ds = parts
        else:
            raise ValueError(
                f"Invalid backbone '{raw_backbone}'. Expected format: '<ARCHITECTURE>-<MODE>-<DATASET>'"
            )
    else:
        arch = opts.get("arch", "thgnn")
        mode = opts.get("mode", "e2e")
        ds = (
            opts.get("dataset")
            or opts.get("train_on")
            or opts.get("ds_name")
            or "FD001"
        )

    arch = arch.lower()
    mode = mode.lower()
    if mode == "default":
        mode = "e2e"
    ds = ds.upper()

    valid_archs = ["thgnn", "cnn", "lstm", "gru"]
    valid_modes = ["e2e", "supcon"]
    valid_datasets = ["FD001", "FD002", "FD003", "FD004"]

    assert arch in valid_archs, f"Unknown arch '{arch}'. Expected one of {valid_archs}"
    assert mode in valid_modes, f"Unknown mode '{mode}'. Expected one of {valid_modes}"
    assert ds in valid_datasets, (
        f"Unknown dataset '{ds}'. Expected one of {valid_datasets}"
    )

    return arch, mode, ds


class Predictor:
    """
    Prediction pipeline supporting original end-to-end head or decapped ML heads.
    """

    def __init__(
        self,
        decap_kind: str,
        model: torch.nn.Module,
        pca: int = 0,
        head_params: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        self.decap_kind = decap_kind
        self.model = model
        self.head = None
        self.head_params = head_params or {}
        self.pca_components = pca
        self.pca = None
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def train(self, train_loader: DataLoader, type_idx: torch.Tensor) -> None:
        if self.decap_kind == "none":
            return

        assert self.head is None
        X = []
        y = []
        type_idx = type_idx.to(self.device)
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                emb = self.model.forward_headless(batch_x, type_idx)
                X.append(emb.cpu().numpy())
                y.append(batch_y.cpu().numpy())

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0).flatten()

        if self.pca_components > 0:
            n_components = min(self.pca_components, X.shape[1])
            self.pca = PCA(n_components=n_components)
            X = self.pca.fit_transform(X)

        if self.decap_kind == "rf":
            self.head = RandomForestRegressor(**self.head_params)
        elif self.decap_kind == "svm":
            self.head = SVR(**self.head_params)
        elif self.decap_kind == "knn":
            self.head = KNeighborsRegressor(**self.head_params)
        elif self.decap_kind == "xgb":
            self.head = XGBRegressor(**self.head_params)
        elif self.decap_kind == "dt":
            self.head = DecisionTreeRegressor(**self.head_params)
        else:
            raise ValueError(f"Unknown decap_kind: {self.decap_kind}")

        self.head.fit(X, y)

    def predict(
        self, loader: DataLoader, type_idx: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        type_idx = type_idx.to(self.device)
        self.model.eval()
        y_true = []

        if self.decap_kind == "none":
            y_pred = []
            with torch.no_grad():
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device)
                    out = self.model(batch_x, type_idx)
                    y_pred.append(out.cpu().numpy())
                    y_true.append(batch_y.cpu().numpy())
            y_pred = np.concatenate(y_pred, axis=0).flatten()
            y_true = np.concatenate(y_true, axis=0).flatten()
            return y_true, y_pred
        else:
            X = []
            with torch.no_grad():
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device)
                    emb = self.model.forward_headless(batch_x, type_idx)
                    X.append(emb.cpu().numpy())
                    y_true.append(batch_y.cpu().numpy())
            X = np.concatenate(X, axis=0)
            y_true = np.concatenate(y_true, axis=0).flatten()

            if self.pca is not None:
                X = self.pca.transform(X)

            assert self.head is not None
            y_pred = self.head.predict(X).flatten()
            return y_true, y_pred


def compute_metrics(
    y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    score = calculate_score(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mae = calculate_mae(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    mape = calculate_map(y_true, y_pred)
    c_index = calculate_concordance_index(y_true, y_pred)
    brier = calculate_brier_score(y_train, y_true, y_pred)
    cda = calculate_cd_auc(y_train, y_true, y_pred)

    safe_score = (
        float(score) if not (np.isnan(score) or np.isinf(score)) else float("inf")
    )

    return {
        "score": safe_score,
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "map": float(mape),
        "cindex": float(c_index),
        "brier": float(brier) if not np.isnan(brier) else None,
        "cdauc": float(cda) if not np.isnan(cda) else None,
    }


class TestModelExperiment(Experiment):
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

    @fixture()
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @fixture()
    def models_dir(self) -> Path:
        p = Path(__file__).resolve().parent.parent.parent.parent / "models"
        if not p.exists():
            p = Path("models").resolve()
        return p

    @fixture()
    def cmapss_data_dir(self) -> Path:
        p = Path(__file__).resolve().parent.parent.parent.parent / "CMAPSSData"
        if not p.exists():
            p = Path("CMAPSSData").resolve()
        return p

    def version(self) -> str:
        try:
            return version_from_git_commit_hash(type(self))
        except Exception:
            return version_from_most_recent_mtime(type(self))

    def first_step(self) -> Callable:
        return self.start_testing

    def variables(self, v: VarNamespace, s: Sampler) -> None:
        opts = get_options()
        arch, mode, ds_name = resolve_backbone_config(opts)

        # Backbone variables fixed from options, NOT suggested by optimizer
        v.arch = arch
        v.mode = mode
        v.ds_name = ds_name
        v.backbone = f"{arch}-{mode}-{ds_name}"

        # Folding configuration
        folding = opts.get("folds", opts.get("folding", "no")).lower()
        assert folding in ["yes", "no"], f"Invalid folds option: '{folding}'"
        v.folding = folding

        v.batch_size = int(opts.get("batch_size", 128))
        v.transfer_ds = opts.get("transfer_ds", "")

        # Head / decap_kind selection
        if "decap_kind" in opts:
            v.decap_kind = opts["decap_kind"].lower()
            if v.mode == "supcon" and v.decap_kind == "none":
                raise ValueError("decap_kind 'none' is invalid for supcon backbone")
        else:
            if v.mode == "e2e":
                v.decap_kind = s.categorical(["none", "dt", "rf", "svm", "knn", "xgb"])
            else:
                v.decap_kind = s.categorical(["dt", "rf", "svm", "knn", "xgb"])

        # PCA components
        if v.decap_kind == "none":
            v.pca = 0
        elif "pca" in opts:
            v.pca = int(opts["pca"])
        else:
            v.pca = s.categorical([0, 2, 4, 8, 16])

        # Head hyperparameters
        with s.branch(v.decap_kind):
            match v.decap_kind:
                case "none":
                    pass
                case "dt":
                    v.dt_max_depth = (
                        int(opts["dt_max_depth"])
                        if "dt_max_depth" in opts
                        else s.categorical([-1, 3, 5, 10, 20])
                    )
                    v.dt_min_samples_split = (
                        int(opts["dt_min_samples_split"])
                        if "dt_min_samples_split" in opts
                        else s.integer(2, 10)
                    )
                case "rf":
                    v.rf_n_estimators = (
                        int(opts["rf_n_estimators"])
                        if "rf_n_estimators" in opts
                        else s.categorical([50, 100, 200])
                    )
                    v.rf_max_depth = (
                        int(opts["rf_max_depth"])
                        if "rf_max_depth" in opts
                        else s.categorical([-1, 5, 10, 20])
                    )
                case "svm":
                    v.svm_c = (
                        float(opts["svm_c"])
                        if "svm_c" in opts
                        else s.log_uniform(1e-2, 1e2)
                    )
                    v.svm_kernel = (
                        opts["svm_kernel"]
                        if "svm_kernel" in opts
                        else s.categorical(["rbf", "linear"])
                    )
                case "knn":
                    v.knn_k = (
                        int(opts["knn_k"]) if "knn_k" in opts else s.integer(3, 15)
                    )
                    v.knn_metric = (
                        opts["knn_metric"]
                        if "knn_metric" in opts
                        else s.categorical(["minkowski", "manhattan"])
                    )
                case "xgb":
                    v.xgb_n_estimators = (
                        int(opts["xgb_n_estimators"])
                        if "xgb_n_estimators" in opts
                        else s.categorical([50, 100, 200])
                    )
                    v.xgb_max_depth = (
                        int(opts["xgb_max_depth"])
                        if "xgb_max_depth" in opts
                        else s.integer(3, 10)
                    )
                    v.xgb_lr = (
                        float(opts["xgb_lr"])
                        if "xgb_lr" in opts
                        else s.log_uniform(1e-3, 0.3)
                    )

    def get_datasets(
        self, ds_name: str, fold: int | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return get_preprocessed_data(
            self.cmapss_data_dir(), ds_name, fold=fold, n_splits=5
        )

    def get_dataloaders(
        self, ds_name: str, fold: int | None, batch_size: int
    ) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
        X_train, y_train, X_val, y_val, X_test, y_test = self.get_datasets(
            ds_name, fold
        )
        pin = torch.cuda.is_available()

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin,
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin,
        )
        return train_loader, val_loader, test_loader, y_train

    def build_head_params(self, trial: Trial) -> dict[str, Any]:
        match trial.v.decap_kind:
            case "dt":
                max_depth = trial.v.dt_max_depth if trial.v.dt_max_depth != -1 else None
                return {
                    "max_depth": max_depth,
                    "min_samples_split": trial.v.dt_min_samples_split,
                    "random_state": 42,
                }
            case "rf":
                max_depth = trial.v.rf_max_depth if trial.v.rf_max_depth != -1 else None
                return {
                    "n_estimators": trial.v.rf_n_estimators,
                    "max_depth": max_depth,
                    "random_state": 42,
                }
            case "svm":
                return {
                    "C": trial.v.svm_c,
                    "kernel": trial.v.svm_kernel,
                }
            case "knn":
                return {
                    "n_neighbors": trial.v.knn_k,
                    "metric": trial.v.knn_metric,
                }
            case "xgb":
                return {
                    "n_estimators": trial.v.xgb_n_estimators,
                    "max_depth": trial.v.xgb_max_depth,
                    "learning_rate": trial.v.xgb_lr,
                    "random_state": 42,
                    "verbosity": 0,
                }
            case _:
                return {}

    @step_method()
    def start_testing(self, trial: Trial) -> NoReturn:
        trial.add_tags(
            [
                "testing",
                trial.v.arch,
                trial.v.mode,
                trial.v.ds_name,
                trial.v.decap_kind,
            ]
        )
        if trial.v.pca:
            trial.add_tag(f"pca{trial.v.pca}")
        self.logger().info(
            "Starting testing experiment with variables: %s",
            json.dumps({k: v for k, v in trial.v._variables_dict.items()}),
        )
        self.exit_next_step(self.evaluate_fold, fold=0)

    @step_method()
    def evaluate_fold(
        self,
        trial: Trial,
        fold: Annotated[int, STATE_ARG_RECORD],
    ) -> NoReturn:
        opts = get_options()
        if trial.v.folding == "yes":
            fold_idx = fold
        elif "fold" in opts:
            fold_idx = int(opts["fold"])
        else:
            fold_idx = None

        target_ds = trial.v.transfer_ds if trial.v.transfer_ds else trial.v.ds_name

        model_path = (
            self.models_dir() / f"{trial.v.arch}-{trial.v.mode}-{trial.v.ds_name}.pth"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Backbone model not found at {model_path}")

        model = torch.load(
            model_path, map_location=self.device(), weights_only=False
        ).to(self.device())
        model.eval()

        train_loader, val_loader, test_loader, y_train = self.get_dataloaders(
            target_ds, fold_idx, trial.v.batch_size
        )
        type_idx = torch.tensor(TYPE_IDX_LIST_CMAPSS, device=self.device())

        head_params = self.build_head_params(trial)
        predictor = Predictor(
            decap_kind=trial.v.decap_kind,
            model=model,
            pca=trial.v.pca,
            head_params=head_params,
            device=self.device(),
        )

        self.logger().info(
            "Training predictor (head=%s, pca=%s) on fold %s...",
            trial.v.decap_kind,
            trial.v.pca,
            fold,
        )
        predictor.train(train_loader, type_idx)

        # Validation set evaluation
        y_val_true, y_val_pred = predictor.predict(val_loader, type_idx)
        val_metrics = compute_metrics(y_train, y_val_true, y_val_pred)
        val_score = val_metrics["score"]

        # Test set evaluation
        y_test_true, y_test_pred = predictor.predict(test_loader, type_idx)
        test_metrics = compute_metrics(y_train, y_test_true, y_test_pred)
        test_score = test_metrics["score"]

        # Record validation score and test score in trial traces
        trial.append_trace("val_score", val_score, fold=fold, **val_metrics)
        trial.append_trace("test_score", test_score, fold=fold, **test_metrics)

        self.logger().info(
            "Fold %d: val_score=%.3f, test_score=%.3f, val_rmse=%.3f, test_rmse=%.3f",
            fold,
            val_score,
            test_score,
            val_metrics["rmse"],
            test_metrics["rmse"],
        )

        if trial.v.folding == "yes" and fold < 4:
            self.exit_next_step(self.evaluate_fold, fold=fold + 1)
        else:
            val_scores = list(trial.trace_values("val_score"))
            mean_val_score = float(np.mean(val_scores))

            test_scores = list(trial.trace_values("test_score"))
            mean_test_score = float(np.mean(test_scores))

            self.logger().info(
                "Completed %d fold(s): mean_val_score=%.3f, mean_test_score=%.3f",
                len(val_scores),
                mean_val_score,
                mean_test_score,
            )
            # The result of the experiment is the score on the validation set
            self.exit_done(mean_val_score)
