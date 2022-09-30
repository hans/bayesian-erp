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


def main(args):
    datasets = []
    for path in args.dataset_paths:
        with path.open("rb") as f:
            datasets.append(pickle.load(f).ensure_torch())
    dataset = average_datasets(datasets, name=args.name)

    with args.output_path.open("wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    args = p.parse_args()
    main(args)