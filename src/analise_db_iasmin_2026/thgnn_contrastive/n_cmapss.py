import enum
from functools import lru_cache
import logging
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import requests
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from analise_db_iasmin_2026.environment import Environment

# Sensor type mapping (0: Temp, 1: Press, 2: Speed, 3: Fuel, 4, 5, 6: OpSettings)
COLUMNS_TYPES = [
    ("alt", 4),
    ("Mach", 5),
    ("TRA", 6),
    ("T2", 0),
    ("T24", 0),
    ("T30", 0),
    ("T48", 0),
    ("T50", 0),
    ("P15", 1),
    ("P2", 1),
    ("P21", 1),
    ("P24", 1),
    ("Ps30", 1),
    ("P40", 1),
    ("P50", 1),
    ("Nf", 2),
    ("Nc", 2),
    ("Wf", 3),
    ("T40", 0),
    ("P30", 1),
]

DOWNLOAD_URLS = {  # TODO: Actually add the URLs
    "N-CMAPSS_DS01-005": "",
    "N-CMAPSS_DS02-006": "https://drive.usercontent.google.com/download?id=1pSFG6TwXzNRvLc1pE4iWGnCOOrjPAmcJ&export=download&authuser=0&confirm=t&uuid=c1696abc-d290-4729-a6d3-21ba4d11c256&at=AMrWOn1-fKsxKF81XdFFXQVGL0Ko:1791167426494",
    "N-CMAPSS_DS03-012": "",
    "N-CMAPSS_DS04": "",
    "N-CMAPSS_DS05": "",
    "N-CMAPSS_DS06": "",
    "N-CMAPSS_DS07": "",
    "N-CMAPSS_DS08a-009": "",
    "N-CMAPSS_DS08c-008": "",
}


class MultiTensorDataset(Dataset):
    def __init__(self, ds: list[TensorDataset]) -> None:
        super().__init__()
        self._ds = ds

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        if index > len(self):
            raise IndexError

        for ds in self._ds:
            if index < len(ds):
                return ds[index]
            else:
                index -= len(ds)

        raise RuntimeError("unreachable code")

    def __len__(self) -> int:
        return sum(len(x) for x in self._ds)


class FoldLoader:
    def __init__(self, dir_base: Path) -> None:
        self.dir_base = dir_base
        train_dir = self.dir_base / "train"
        samples, labels = load_part_array_unmerged(
            sorted(train_dir.iterdir(), key=lambda x: str(x.name))
        )
        self.datasets = []
        for i, (sample, label) in enumerate(zip(samples, labels)):
            td = TensorDataset(FloatTensor(sample), FloatTensor(label))
            self.datasets.append(td)

    def load_train_val(
        self, fold: int, batch_size: int
    ) -> tuple[DataLoader, DataLoader]:
        train_ds = [x for i, x in enumerate(self.datasets) if i != fold]
        val_ds = self.datasets[fold]
        train_loader = DataLoader(
            MultiTensorDataset(train_ds),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        return train_loader, val_loader

    def fold_count(self) -> int:
        return len(self.datasets)


def load_part_array_unmerged(npz_units) -> tuple[list[np.ndarray], list[np.ndarray]]:
    sample_array_lst = []
    label_array_lst = []
    for npz_unit in npz_units:
        loaded = np.load(npz_unit)
        sample_array_lst.append(loaded["sample"].transpose(2, 0, 1))
        label_array_lst.append(loaded["label"])
    return sample_array_lst, label_array_lst


def merge_parts(
    sample: list[np.ndarray], label: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    sample_array = np.concatenate(sample)
    label_array = np.concatenate(label)
    return sample_array, label_array


def folds_from_files(files: Path) -> FoldLoader:
    return FoldLoader(files)


def get_n_cmapss_folds(dataset_id: str, log: logging.Logger) -> FoldLoader:
    env = Environment.get_environment()
    if env.is_laptop():
        base = Path("/mnt/sghdd/N-CMAPSS_DL/N-CMAPSS") / dataset_id
        log.info("will load N-CMAPSS from %s", base)
        return folds_from_files(base)
    elif env.is_kaggle():
        ds_path = Path(
            "/kaggle/input/datasets/sasharodelasteidle/ds02-006/N-CMAPSS_DS02-006"
        )
        return folds_from_files(ds_path)
    else:
        download_dir = Path("download")
        download_dir.mkdir(parents=True, exist_ok=True)
        the_dir = download_dir / dataset_id
        if the_dir.is_dir():
            return folds_from_files(the_dir)
        if env.is_colab():
            file_name = "/content/drive/MyDrive/N-CMAPSS_DS02-006.zip"
        else:
            file_name = the_dir.with_suffix(".zip")
            url = DOWNLOAD_URLS[dataset_id]
            log.info("will download N-CMAPSS from URL %s to file %s", url, file_name)
            resp = requests.get(url, stream=True)
            with open(file_name, "wb") as file:
                file.writelines(resp.iter_content(8192, decode_unicode=False))
        with zipfile.ZipFile(file_name, "r") as zf:
            zf.extractall(download_dir)
            return folds_from_files(the_dir)
