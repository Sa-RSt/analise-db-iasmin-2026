import itertools
import random
from collections.abc import Callable

import optuna
from rungrid.experiment import Experiment, Sampler, VarNamespace

from analise_db_iasmin_2026.options import get_options, pop_options, push_options


class RedoExperimentsOct2026Select:
    def __init__(self, study: optuna.Study, exp: Experiment) -> None:
        self._study = study
        datasets = ["FD001", "FD002", "FD003", "FD004"]
        self._kinds = [
            self.fd001_best_thgnn_supcon,
            self.fd002_best_thgnn_supcon,
            self.fd003_best_thgnn_supcon,
            self.fd004_best_thgnn_supcon,
        ] + [self.curry_thgnn_e2e(x) for x in datasets]
        for arch in ["cnn", "lstm", "gru"]:
            for mode in ["e2e", "supcon"]:
                for ds in datasets:
                    self._kinds.append(self.curry_arch_mode(ds, mode, arch))

        random.shuffle(self._kinds)
        self._iter = itertools.cycle(self._kinds)
        self._exp = exp

    def fd001_best_thgnn_supcon(self, v: VarNamespace, s: Sampler) -> None:
        v.mode = "supcon"
        v.arch = "thgnn"
        v.dim = 16
        v.hdim = 4
        v.n_type_emb = 16
        v.lr = s.log_uniform(1e-5, 1e-1)
        v.batch_size = 128
        v.patience = 10
        v.epochs = int(get_options().get("epochs", "") or 50)
        v.aug_jitter = 0.02
        v.aug_scaling = 0.1
        v.layers = 0
        v.kernel_size = 0
        v.margin_mode = "margin_absolute"
        v.margin = s.integer(3, 50)
        v.ds_name = "FD001"

    def fd002_best_thgnn_supcon(self, v: VarNamespace, s: Sampler) -> None:
        v.mode = "supcon"
        v.arch = "thgnn"
        v.dim = 8
        v.hdim = 4
        v.n_type_emb = 8
        v.lr = s.log_uniform(1e-5, 1e-1)
        v.batch_size = 128
        v.patience = 10
        v.epochs = int(get_options().get("epochs", "") or 50)
        v.aug_jitter = 0.02
        v.aug_scaling = 0.1
        v.layers = 0
        v.kernel_size = 0
        v.margin_mode = "margin_absolute"
        v.margin = 27
        v.ds_name = "FD002"

    def fd003_best_thgnn_supcon(self, v: VarNamespace, s: Sampler) -> None:
        v.mode = "supcon"
        v.arch = "thgnn"
        v.dim = 16
        v.hdim = 8
        v.n_type_emb = 4
        v.lr = s.log_uniform(1e-5, 1e-1)
        v.batch_size = 128
        v.patience = 10
        v.epochs = int(get_options().get("epochs", "") or 50)
        v.aug_jitter = 0.02
        v.aug_scaling = 0.1
        v.layers = 0
        v.kernel_size = 0
        v.margin_mode = "margin_absolute"
        v.margin = 13
        v.ds_name = "FD003"

    def fd004_best_thgnn_supcon(self, v: VarNamespace, s: Sampler) -> None:
        v.mode = "supcon"
        v.arch = "thgnn"
        v.dim = 32
        v.hdim = 4
        v.n_type_emb = 16
        v.lr = s.log_uniform(1e-5, 1e-1)
        v.batch_size = 128
        v.patience = 10
        v.epochs = int(get_options().get("epochs", "") or 50)
        v.aug_jitter = 0.02
        v.aug_scaling = 0.1
        v.layers = 0
        v.kernel_size = 0
        v.margin_mode = "margin_absolute"
        v.margin = 10
        v.ds_name = "FD004"

    def curry_thgnn_e2e(self, ds_name: str) -> Callable[[VarNamespace, Sampler], None]:
        def fn(v: VarNamespace, s: Sampler):
            v.mode = "e2e"
            v.arch = "thgnn"
            v.dim = 32
            v.hdim = 4
            v.n_type_emb = 8
            v.lr = 5e-4
            v.batch_size = 50
            v.patience = 10
            v.epochs = int(get_options().get("epochs", "") or 50)
            v.aug_jitter = 0.02
            v.aug_scaling = 0.1
            v.layers = 0
            v.kernel_size = 0
            v.margin_mode = "e2e"
            v.margin = 0
            v.ds_name = ds_name

        return fn

    def curry_arch_mode(
        self, ds_name: str, mode: str, arch: str
    ) -> Callable[[VarNamespace, Sampler], None]:
        def fn(v: VarNamespace, s: Sampler):
            opts = get_options().copy()
            try:
                opts["arch"] = arch
                opts["mode"] = mode
                opts["train_on"] = ds_name
                push_options(opts)
                self._exp.variables(v, s)
            finally:
                pop_options()

        return fn

    def __call__(self, v: VarNamespace, s: Sampler) -> None:
        next(self._iter)(v, s)


class RedoExperimentsOct2026Selected:
    def __init__(self, study: optuna.Study, exp: Experiment) -> None:
        self._study = study
        self._exp = exp
        self._kinds = [
            self.fd001_best_thgnn_supcon_lr_margin(0.008543046441516457, 16),
            self.fd001_best_thgnn_supcon_lr_margin(5e-4, 20),
            self.fd002_best_thgnn_supcon_lr(0.005162067891140875),
            self.fd002_best_thgnn_supcon_lr(5e-4),
            self.fd003_best_thgnn_supcon_lr(0.0105816834169473),
            self.fd003_best_thgnn_supcon_lr(5e-4),
            self.fd004_best_thgnn_supcon_lr(0.005670064419893825),
            self.fd004_best_thgnn_supcon_lr(5e-4),
        ]

    def fd001_best_thgnn_supcon_lr_margin(
        self, lr, margin
    ) -> Callable[[VarNamespace, Sampler]]:
        def curry(v: VarNamespace, s: Sampler):
            v.folding = "no"
            v.mode = "supcon"
            v.arch = "thgnn"
            v.dim = 16
            v.hdim = 4
            v.n_type_emb = 16
            v.lr = lr
            v.batch_size = 128
            v.patience = 10
            v.epochs = int(get_options().get("epochs", "") or 50)
            v.aug_jitter = 0.02
            v.aug_scaling = 0.1
            v.layers = 0
            v.kernel_size = 0
            v.margin_mode = "margin_absolute"
            v.margin = margin
            v.ds_name = "FD001"

        return curry

    def fd002_best_thgnn_supcon_lr(self, lr) -> Callable[[VarNamespace, Sampler]]:
        def curry(v: VarNamespace, s: Sampler):
            v.folding = "no"
            v.mode = "supcon"
            v.arch = "thgnn"
            v.dim = 8
            v.hdim = 4
            v.n_type_emb = 8
            v.lr = lr
            v.batch_size = 128
            v.patience = 10
            v.epochs = int(get_options().get("epochs", "") or 50)
            v.aug_jitter = 0.02
            v.aug_scaling = 0.1
            v.layers = 0
            v.kernel_size = 0
            v.margin_mode = "margin_absolute"
            v.margin = 27
            v.ds_name = "FD002"

        return curry

    def fd003_best_thgnn_supcon_lr(self, lr) -> Callable[[VarNamespace, Sampler]]:
        def curry(v: VarNamespace, s: Sampler):
            v.folding = "no"
            v.mode = "supcon"
            v.arch = "thgnn"
            v.dim = 16
            v.hdim = 8
            v.n_type_emb = 4
            v.lr = lr
            v.batch_size = 128
            v.patience = 10
            v.epochs = int(get_options().get("epochs", "") or 50)
            v.aug_jitter = 0.02
            v.aug_scaling = 0.1
            v.layers = 0
            v.kernel_size = 0
            v.margin_mode = "margin_absolute"
            v.margin = 13
            v.ds_name = "FD003"

        return curry

    def fd004_best_thgnn_supcon_lr(self, lr) -> Callable[[VarNamespace, Sampler]]:
        def curry(v: VarNamespace, s: Sampler):
            v.folding = "no"
            v.mode = "supcon"
            v.arch = "thgnn"
            v.dim = 32
            v.hdim = 4
            v.n_type_emb = 16
            v.lr = lr
            v.batch_size = 128
            v.patience = 10
            v.epochs = int(get_options().get("epochs", "") or 50)
            v.aug_jitter = 0.02
            v.aug_scaling = 0.1
            v.layers = 0
            v.kernel_size = 0
            v.margin_mode = "margin_absolute"
            v.margin = 10
            v.ds_name = "FD004"

        return curry
