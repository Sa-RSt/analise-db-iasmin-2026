from pathlib import Path

import optuna.storages.journal
import torch
from rungrid.plumbing import ResultPodiumSink, StateSink
from rungrid.plumbing.storage import BucketFileStorage

from analise_db_iasmin_2026.thgnn_contrastive.exp_train_model import (
    TrainModelExperiment,
)

here = Path(__file__).parent
file_store_base = Path("/mnt/sghdd/adbiasminfilestore/")
file_store = BucketFileStorage(
    file_store_base / ".rg-finished", file_store_base / ".rg-unfinished"
)


def study_factory(name: str) -> optuna.Study:
    name_alnum = "".join(x for x in name if x.isalnum())
    journal_file = here / f"optuna_journal_{name_alnum}.log"
    backend = optuna.storages.journal.JournalFileBackend(str(journal_file.resolve()))
    return optuna.create_study(
        study_name=name,
        storage=optuna.storages.JournalStorage(backend),
        load_if_exists=True,
        direction="maximize",
    )
