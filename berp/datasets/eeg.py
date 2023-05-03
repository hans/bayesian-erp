import logging
import pickle
from typing import *

from hydra.utils import to_absolute_path
import torch
from torch._utils import _get_all_device_indices, _get_device_index

from berp.datasets import NaturalLanguageStimulus
from berp.datasets.base import BerpDataset, NestedBerpDataset

L = logging.getLogger(__name__)


def load_eeg_dataset(paths: List[str],
                     subset_sensors: Optional[List[str]] = None,
                     normalize_X_ts: bool = True,
                     normalize_X_variable: bool = True,
                     normalize_Y: bool = True,
                     special_normalize_variable_intercept: bool = False,
                     stimulus_paths: Optional[Dict[str, str]] = None,
                     device: Optional[str] = None,
                     dtype: Optional[str] = "float32",
                     ts_dtype: Optional[str] = "float32",
                     **kwargs) -> NestedBerpDataset:
    if dtype not in ["float16", "float32", "float64"]:
        raise ValueError(f"Invalid dtype: {dtype}")
    if ts_dtype not in ["float16", "float32", "float64"]:
        raise ValueError(f"Invalid ts_dtype: {ts_dtype}")
    dtype = getattr(torch, dtype)
    ts_dtype = getattr(torch, ts_dtype)

    # If stimulus data is stored separately, load this first.
    stimulus_data: Dict[str, NaturalLanguageStimulus] = {}
    if stimulus_paths is not None:
        for name, path in stimulus_paths.items():
            with open(to_absolute_path(path), "rb") as f:
                stimulus_data[name] = pickle.load(f)

    datasets = []
    for i, dataset in enumerate(paths):
        with open(to_absolute_path(dataset), "rb") as f:
            ds_device = device
            if ds_device == "cuda":
                # Distribute across all available GPUs.
                available_devices = _get_all_device_indices()
                ds_device_idx = available_devices[i % len(available_devices)]
                ds_device = str(torch.device("cuda", _get_device_index(ds_device_idx, True)))

            ds: BerpDataset = pickle.load(f).ensure_torch(
                device=ds_device, dtype=dtype, ts_dtype=ts_dtype)
            if ds.stimulus_name in stimulus_data:
                ds.add_stimulus(stimulus_data[ds.stimulus_name])
            if subset_sensors is not None:
                ds.subset_sensors(list(subset_sensors), on_missing="warn")
            datasets.append(ds)

    dataset = NestedBerpDataset(datasets)

    def norm_ts(tensor, add_zeros=None):
        if add_zeros is None:
            ref_tensor = tensor
        else:
            ref_tensor = torch.cat([tensor, torch.zeros(add_zeros, *tensor.shape[1:], dtype=tensor.dtype)], dim=0)
        return (tensor - ref_tensor.mean(dim=0, keepdim=True)) / ref_tensor.std(dim=0, keepdim=True)

    if normalize_X_ts or normalize_X_variable or normalize_Y:
        for ds in dataset.datasets:
            if normalize_X_ts:
                ds.X_ts = norm_ts(ds.X_ts)
            if normalize_X_variable:
                # Don't normalize intercept columns the same way.
                mask = ~(ds.X_variable == 1).all(dim=0)
                ds.X_variable[:, mask] = norm_ts(ds.X_variable[:, mask])

                if special_normalize_variable_intercept and (~mask).any():
                    # Scale intercept columns so that the resulting design matrix
                    # column after scattering has mean 0 and std 1.
                    intercept_col_idx = torch.where(~mask)[0][0]
                    n_add_zeros = ds.X_ts.shape[0] - ds.X_variable[:, intercept_col_idx].sum().int().item()
                    ds.X_variable[:, ~mask] = norm_ts(ds.X_variable[:, ~mask],
                                                        add_zeros=n_add_zeros)
            if normalize_Y:
                ds.Y = norm_ts(ds.Y)

    return dataset
