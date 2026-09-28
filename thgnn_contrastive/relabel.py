from pathlib import Path
import os
import shutil
from tqdm import tqdm
from hashlib import sha1
import pickle


def test(model_path, dataset_id, plots, decap_kind, pca, validation, use_sidekick):
    destination_dir = Path(__file__).parent / "log"
    destination_dir.mkdir(exist_ok=True)
    hashing_blob = ",".join(
        map(
            str,
            [
                dataset_id,
                pca,
                Path(model_path).name,
                decap_kind,
                validation,
            ],
        )
    )
    if not use_sidekick:
        hashing_blob += ",without_sidekick"
    blob_hash = sha1(hashing_blob.encode()).hexdigest()
    destination_file = destination_dir / f"{blob_hash}.csv"
    content_there = destination_file.read_text()
    lines = content_there.splitlines()
    lines[0] += ",sidekick"
    lines[1] += [",0", ",1"][int(use_sidekick)]
    destination_file.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="model.pth", help="Path to the saved model"
    )
    parser.add_argument(
        "--dataset", type=str, default="FD001", help="Dataset ID (e.g., FD001)"
    )
    parser.add_argument(
        "--plots", type=str, default="dontsavetheplots", help="Plot output dir"
    )
    parser.add_argument(
        "--pca",
        type=float,
        default=0,
        help="PCA components, default is no PCA/full dimensionality",
    )
    parser.add_argument(
        "--decap-kind", type=str, default="none", help="Head replacement"
    )
    parser.add_argument(
        "--final", action="store_true", help="Use test set instead of validation"
    )
    parser.add_argument(
        "--use-sidekick", action="store_true", help="Include sidekick LR model"
    )
    args = parser.parse_args()

    test(
        model_path=args.model,
        dataset_id=args.dataset,
        plots=args.plots,
        decap_kind=args.decap_kind,
        pca=args.pca,
        validation=not args.final,
        use_sidekick=args.use_sidekick,
    )
