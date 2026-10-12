import sys
from pathlib import Path

import optuna.storages.journal
from rungrid.cli import CLI
from rungrid.optimize import OptimizingSampler
from rungrid.plumbing import CopyingMultiSink
from rungrid.plumbing.storage import BucketFileStorage, CSVSink

from analise_db_iasmin_2026.environment import Environment
from analise_db_iasmin_2026.options import get_options, push_options
from analise_db_iasmin_2026.optuna_server import TrialClient, TrialServer
from analise_db_iasmin_2026.thgnn_contrastive.exp_train_model import (
    TrainModelExperiment,
)
from analise_db_iasmin_2026.thgnn_contrastive.redo_experiments import (
    RedoExperimentsOct2026,
)

push_options(CLI.get_instance().get_options())
env = Environment.get_environment()
here = Path(__file__).parent
file_store_base = env.get_file_store_path()
file_store = BucketFileStorage(
    file_store_base / ".rg-finished", file_store_base / ".rg-unfinished"
)
trial_client = TrialClient(get_options().get("trial-client-url", ""))


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


def run_trial_server():
    print("REDOING THE EXPERIMENTS!", file=sys.stderr)
    exp = TrainModelExperiment.get_instance()
    study = study_factory(exp.identifier + "-server")
    redo = RedoExperimentsOct2026(study, exp)
    optimizer = OptimizingSampler(exp, study, redo)
    file_no = 0
    file_name = Path("/")
    while file_name.exists():
        file_no += 1
        file_name = Path(f"report-{file_no}.csv")
    with file_name.open("w") as f:
        sink = CopyingMultiSink([optimizer, CSVSink(f)])
        ts = TrialServer(optimizer, sink)
        ts.serve()


tme = TrainModelExperiment.get_instance()
redo_study = study_factory(tme.identifier + "-redo")
redo = RedoExperimentsOct2026(redo_study, tme)
redo_optimizer = OptimizingSampler(tme, redo_study, redo)
