"""
Combines individual subject BerpDatasets into a single BerpDataset
with average-pooled EEG response.
"""

from argparse import ArgumentParser
from pathlib import Path
import pickle

from berp.datasets import BerpDataset
from berp.datasets.base import average_datasets


p = ArgumentParser()
p.add_argument("dataset_paths", nargs="+", type=Path)
p.add_argument("-n", "--name", default="average")
p.add_argument("-o", "--output_path", type=Path, required=True)
p.add_argument("--subset_sensors", type=str)
p.add_argument("--average_strategy", choices=["macro", "micro"], default="micro",
               help="Macro: Average across sensors within-subject before averaging across subjects")


def main(args):
    datasets = []
    subset_sensors = None
    if args.subset_sensors:
        subset_sensors = [s.strip() for s in args.subset_sensors.split(",")]

    for path in args.dataset_paths:
        with path.open("rb") as f:
            ds = pickle.load(f).ensure_torch()
            if subset_sensors:
                ds = ds.subset_sensors(subset_sensors)
            if args.average_strategy == "macro":
                ds = ds.average_sensors()
            datasets.append(ds)
    dataset = average_datasets(datasets, name=args.name)

    with args.output_path.open("wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    args = p.parse_args()
    main(args)