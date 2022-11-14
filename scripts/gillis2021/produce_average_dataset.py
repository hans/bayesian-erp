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
p.add_argument("--subset_sensors", nargs="+")
p.add_argument("--macro_average", action="store_true",
               help="Average across sensors within-subject before averaging across subjects")


def main(args):
    datasets = []
    for path in args.dataset_paths:
        with path.open("rb") as f:
            ds = pickle.load(f).ensure_torch()
            if args.subset_sensors:
                ds = ds.subset_sensors(args.subset_sensors)
            if args.macro_average:
                ds = ds.average_sensors()
            datasets.append(ds)
    dataset = average_datasets(datasets, name=args.name)

    with args.output_path.open("wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    args = p.parse_args()
    main(args)