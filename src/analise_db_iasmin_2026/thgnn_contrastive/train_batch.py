import datetime
import os
import shutil
import sys
import pickle
import shlex
from pathlib import Path
from random import Random
import pandas as pd
import numpy as np
import subprocess as sp
from uuid import uuid4

from preprocess import load_data, ACTIVE_SENSOR_TYPES

DATA_PATH = "../CMAPSSData"
here = Path(__file__).parent


def load_all_cmapss_data():
    data = {}
    for dataset_id in ["FD001", "FD002", "FD003", "FD004"]:
        data[dataset_id] = load_data(DATA_PATH, dataset_id)
    return data


def run_optuna_training(
    dataset_id,
    mode,
    all_data,
    n_folds,
    loeo,
    unit_counts,
    epochs,
    trials,
    og_dims=False,
    model_type="thgnn",
    gru_hidden_dim=64,
    gru_layers=1,
    lstm_hidden_dim=64,
    lstm_layers=1,
    cnn_layers=2,
    cnn_kernel_size=3,
):
    import optuna
    import optuna.storages.journal
    from train import train

    iters = range(unit_counts[dataset_id]) if loeo else range(n_folds)
    cv_mode = "loeo" if loeo else "kfold"

    def objective(trial):
        # Sample hyperparameters
        if og_dims:
            dim = 16
            hdim = 4
            n_type_emb = 8
        else:
            dim = trial.suggest_categorical("dim", [8, 16, 32])
            hdim = trial.suggest_categorical("hdim", [2, 4, 8])
            n_type_emb = trial.suggest_categorical("n_type_emb", [4, 8, 16])

        if model_type in ["cnn1d", "1dcnn", "cnn"]:
            if og_dims:
                c_k_size = cnn_kernel_size if cnn_kernel_size > 0 else 3
            else:
                c_k_size = trial.suggest_categorical("cnn_kernel_size", [3, 5, 7])
        else:
            c_k_size = trial.suggest_categorical("cnn_kernel_size", [0, 3, 5, 7])

        if model_type == "gru":
            if og_dims:
                g_hdim = gru_hidden_dim
                g_layers = gru_layers
            else:
                g_hdim = trial.suggest_categorical("gru_hidden_dim", [32, 64, 128])
                g_layers = trial.suggest_categorical("gru_layers", [1, 2, 3])
        else:
            g_hdim = gru_hidden_dim
            g_layers = gru_layers

        if model_type == "lstm":
            if og_dims:
                l_hdim = lstm_hidden_dim
                l_layers = lstm_layers
            else:
                l_hdim = trial.suggest_categorical("lstm_hidden_dim", [32, 64, 128])
                l_layers = trial.suggest_categorical("lstm_layers", [1, 2, 3])
        else:
            l_hdim = lstm_hidden_dim
            l_layers = lstm_layers

        if model_type in ["cnn1d", "1dcnn", "cnn"]:
            if og_dims:
                c_layers = cnn_layers
            else:
                c_layers = trial.suggest_categorical("cnn_layers", [1, 2, 3, 4])
        else:
            c_layers = cnn_layers

        alpha = 1e-3
        if mode == "multitask":
            alpha = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
        elif mode in ["supconL1", "supconL1+simclr", "supcon+simclr"]:
            alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)

        margin = 10.0
        if mode in ["supcon", "supconL1", "supconL1+simclr", "supcon+simclr"]:
            margin = trial.suggest_float("margin", 5.0, 50.0)

        ema_decay = 0.99
        if mode == "jepa":
            ema_decay = trial.suggest_float("ema_decay", 0.9, 0.999)

        cov_scale = 1.0
        if mode in ["supconL1", "supconL1+simclr", "supcon+simclr"]:
            cov_scale = trial.suggest_float("cov_scale", 0.01, 10.01, log=True) - 0.01

        simclr_scale = 1.0
        if mode in ["supconL1+simclr", "supcon+simclr"]:
            simclr_scale = (
                trial.suggest_float("simclr_scale", 0.01, 10.01, log=True) - 0.01
            )

        fold_losses = []
        for fold in iters:
            # Descriptive name for the model
            time_format = "%Y-%m-%d-%H-%M-%S"
            save_name = f"optuna-{model_type}-{mode}-{dataset_id}-{datetime.datetime.now().strftime(time_format)}-trial{trial.number}"
            models_folder = here / "models"
            models_folder.mkdir(exist_ok=True)
            save_path = models_folder / save_name
            if save_path.is_file():
                print(save_path, "exists, skipping")
                continue
            else:
                print("Training", save_path)

            # Fork for isolation
            pid = os.fork()
            if pid == 0:
                # CHILD
                try:
                    loss = train(
                        save_path=str(save_path),
                        dataset_id=dataset_id,
                        epochs=epochs,
                        train_mode=mode,
                        alpha=alpha,
                        margin=margin,
                        d=dim,
                        d_hidden=hdim,
                        n_type_emb=n_type_emb,
                        ema_decay=ema_decay,
                        fold=fold,
                        n_splits=n_folds,
                        cv_mode=cv_mode,
                        loaded_data=all_data[dataset_id],
                        cov_scale=cov_scale,
                        simclr_scale=simclr_scale,
                        cnn_kernel_size=c_k_size,
                        model_type=model_type,
                        gru_hidden_dim=g_hdim,
                        gru_layers=g_layers,
                        lstm_hidden_dim=l_hdim,
                        lstm_layers=l_layers,
                        cnn_layers=c_layers,
                    )
                    with open(f"/tmp/optuna_loss_{os.getpid()}.txt", "w") as f:
                        f.write(str(loss))
                except Exception as e:
                    print(f"Error in training: {e}")
                    import traceback

                    traceback.print_exc()
                if len(list(models_folder.iterdir())) >= 200:
                    if (
                        sp.call(
                            shlex.join(
                                [
                                    "zip",
                                    "-r",
                                    f"models-{uuid4()}.zip",
                                    str(models_folder),
                                ]
                            ),
                            shell=True,
                        )
                        == 0
                    ):
                        shutil.rmtree(models_folder)
                    models_folder.mkdir(exist_ok=True)
                os._exit(0)
            else:
                # PARENT
                _, exit_code = os.waitpid(pid, 0)
                if exit_code != 0:
                    # Trial failed
                    return float("inf")

                loss_file = f"/tmp/optuna_loss_{pid}.txt"
                if os.path.exists(loss_file):
                    with open(loss_file, "r") as f:
                        loss = float(f.read())
                    os.remove(loss_file)
                    fold_losses.append(loss)
                else:
                    return float("inf")

        return np.mean(fold_losses)

    study_name = f"train-study-{dataset_id}-{mode}-{cv_mode}-{model_type}"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend("optuna_journal_train")
    )
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=trials)

    print(f"Best trial for {dataset_id} {mode} {cv_mode}:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")

    # Run a final training with best params and a standard name
    # best_params = study.best_params
    # dim = best_params["dim"]
    # hdim = best_params["hdim"]
    # n_type_emb = best_params["n_type_emb"]
    # alpha = best_params.get("alpha", 0.0 if mode in ["supconL1", "supconL1+simclr", "supcon+simclr"] else 1e-3)
    # margin = best_params.get("margin", 10.0)
    # ema_decay = best_params.get("ema_decay", 0.99)
    # cov_scale = best_params.get("cov_scale", 1.0)
    # simclr_scale = best_params.get("simclr_scale", 1.0)

    # for fold in iters:
    #     save_p = f"best-{mode}-{dataset_id}-dim{dim}-hdim{hdim}-nte{n_type_emb}"
    #     if mode == "multitask":
    #         save_p += f"-a{alpha:.3e}"
    #     elif mode in ["supcon", "supconL1", "supconL1+simclr", "supcon+simclr"]:
    #         save_p += f"-m{margin:.1f}"
    #         if mode in ["supconL1", "supconL1+simclr", "supcon+simclr"]:
    #             save_p += f"-c{cov_scale:.3f}-a{alpha:.3e}"
    #             if mode in ["supconL1+simclr", "supcon+simclr"]:
    #                 save_p += f"-s{simclr_scale:.3f}"
    #     elif mode == "jepa":
    #         save_p += f"-ed{ema_decay:.3f}"

    #     save_path = here / "models" / save_p

    #     pid = os.fork()
    #     if pid == 0:
    #         try:
    #             train(
    #                 save_path=str(save_path),
    #                 dataset_id=dataset_id,
    #                 epochs=epochs,
    #                 train_mode=mode,
    #                 alpha=alpha,
    #                 margin=margin,
    #                 d=dim,
    #                 d_hidden=hdim,
    #                 n_type_emb=n_type_emb,
    #                 ema_decay=ema_decay,
    #                 fold=fold,
    #                 n_splits=n_folds,
    #                 cv_mode=cv_mode,
    #                 loaded_data=all_data[dataset_id],
    #                 cov_scale=cov_scale,
    #                 simclr_scale=simclr_scale,
    #             )
    #         except Exception as e:
    #             print(f"Error in final training: {e}")
    #         os._exit(0)
    #     else:
    #         os.waitpid(pid, 0)

    return study.best_params


def main(
    n_folds,
    loeo,
    target_dataset,
    target_mode,
    epochs,
    trials,
    og_dims,
    model_type,
    gru_hidden_dim,
    gru_layers,
    lstm_hidden_dim,
    lstm_layers,
    cnn_layers,
    cnn_kernel_size,
):
    all_data = load_all_cmapss_data()

    # Unit counts for CMAPSS datasets
    unit_counts = {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 249}

    datasets = [f"FD00{x}" for x in range(1, 5)]
    if target_dataset:
        datasets = [target_dataset]

    modes = [
        "simclr",
        "multitask",
        "supcon",
        "supconL1",
        "supconL1+simclr",
        "supcon+simclr",
        "jepa",
        "default",
    ]
    if target_mode:
        modes = [target_mode]

    for dataset in datasets:
        for mode in modes:
            print(f"Starting Optuna study for {dataset} {mode}...")
            run_optuna_training(
                dataset,
                mode,
                all_data,
                n_folds,
                loeo,
                unit_counts,
                epochs,
                trials,
                og_dims=og_dims,
                model_type=model_type,
                gru_hidden_dim=gru_hidden_dim,
                gru_layers=gru_layers,
                lstm_hidden_dim=lstm_hidden_dim,
                lstm_layers=lstm_layers,
                cnn_layers=cnn_layers,
                cnn_kernel_size=cnn_kernel_size,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument(
        "--loeo", action="store_true", help="Enable Leave-One-Engine-Out mode"
    )
    parser.add_argument("--dataset", type=str, help="Target dataset (e.g. FD001)")
    parser.add_argument("--mode", type=str, help="Target mode (e.g. simclr)")
    parser.add_argument("--epochs", type=int, default=50, help="Epoch count")
    parser.add_argument("--trials", type=int, default=15, help="Optuna trial count")
    parser.add_argument(
        "--og-dims",
        action="store_true",
        help="Use default dimensions for d, d_hidden, and n_type_emb",
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
    parser.add_argument(
        "--cnn-kernel-size",
        type=int,
        default=3,
        help="1D CNN kernel size",
    )
    args = parser.parse_args()

    main(
        args.n_folds,
        args.loeo,
        args.dataset,
        args.mode,
        args.epochs,
        args.trials,
        args.og_dims,
        args.model_type,
        args.gru_hidden_dim,
        args.gru_layers,
        args.lstm_hidden_dim,
        args.lstm_layers,
        args.cnn_layers,
        args.cnn_kernel_size,
    )
