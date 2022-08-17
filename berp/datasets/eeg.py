import pickle
from typing import *

from hydra.utils import to_absolute_path
import mne

from berp.datasets.base import BerpDataset, NestedBerpDataset


def load_eeg_dataset(paths: List[str], montage_name: str,
                     subset_sensors: Optional[List[str]] = None) -> NestedBerpDataset:
    datasets = []
    for dataset in paths:
        with open(to_absolute_path(dataset), "rb") as f:
            datasets.append(pickle.load(f).ensure_torch())

    dataset = NestedBerpDataset(datasets)
    montage = mne.channels.make_standard_montage(montage_name)

    if subset_sensors is not None:
        sensor_idxs = [montage.ch_names.index(s) for s in subset_sensors]
        dataset = dataset.subset_sensors(sensor_idxs)

    return dataset