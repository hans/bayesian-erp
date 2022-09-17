import pickle
from typing import *

from hydra.utils import to_absolute_path
import mne
import torch

from berp.datasets.base import BerpDataset, NestedBerpDataset


def load_eeg_dataset(paths: List[str], montage_name: str,
                     subset_sensors: Optional[List[str]] = None,
                     normalize_X_ts: bool = True,
                     normalize_X_variable: bool = True,
                     normalize_Y: bool = True) -> NestedBerpDataset:
    datasets = []
    for dataset in paths:
        with open(to_absolute_path(dataset), "rb") as f:
            datasets.append(pickle.load(f).ensure_torch())

    dataset = NestedBerpDataset(datasets)

    def norm_ts(tensor):
        return (tensor - tensor.mean(dim=0, keepdim=True)) / tensor.std(dim=0, keepdim=True)

    if normalize_X_ts or normalize_X_variable or normalize_Y:
        for ds in dataset.datasets:
            if normalize_X_ts:
                ds.X_ts = norm_ts(ds.X_ts)
            if normalize_X_variable:
                # Don't normalize intercept columns the same way.
                mask = ~(ds.X_variable == 1).all(dim=0)
                ds.X_variable[:, mask] = norm_ts(ds.X_variable[:, mask])
                
                # Scale intercept column
                # HACK HACK, steal from word_onset. Should really do the add_zeros trick again
                f_idx = ds.ts_feature_names.index("word onsets_0")
                scale = ds.X_ts[:, f_idx].nonzero()[0, 0]
                ds.X_variable[:, ~mask] *= scale
            if normalize_Y:
                ds.Y = norm_ts(ds.Y)
    
    if subset_sensors is not None:
        montage = mne.channels.make_standard_montage(montage_name)

        sensor_idxs = [montage.ch_names.index(s) for s in subset_sensors]
        dataset = dataset.subset_sensors(sensor_idxs)

    return dataset