import logging
from typing import Optional
from typing_extensions import TypeAlias

import numpy as np
import pandas as pd

import torch
from torchtyping import TensorType  # type: ignore

from berp.datasets import BerpDataset

L = logging.getLogger(__name__)


EpochDataFrame: TypeAlias = pd.DataFrame


def epoch_ts(data: TensorType["n_samples", "n_sensors", float],
             epoch_points: TensorType["n_epochs", torch.long],
             epoch_shift_left: int, epoch_shift_right: int,
             baseline=True,
             baseline_window_left: int = 0,
             baseline_window_right: int = 10) -> pd.DataFrame:
    """
    Epoch a dataset. NB that inputs to this function are all in sample units,
    not time units.
    """

    epoch_n_samples = epoch_shift_right - epoch_shift_left
    n_sensors = data.shape[1]

    epochs = np.empty((len(epoch_points), epoch_n_samples, n_sensors))
    for i, samp_i in enumerate(epoch_points):
        if samp_i > data.shape[0]:
            raise ValueError(f"epoch point {samp_i} (at idx {i}) is out of bounds")

        # Make sure we don't provide a negative index in left slice.
        epoch_data_left_idx = samp_i + epoch_shift_left
        epoch_data = data[max(0, epoch_data_left_idx):samp_i + epoch_shift_right].clone()

        # Pad
        if epoch_data.shape[0] < epoch_n_samples:
            # pad with NaNs
            pad_left = max(0, -epoch_data_left_idx)
            pad_right = epoch_n_samples - (epoch_data.shape[0] + pad_left)
            epoch_data = np.pad(epoch_data, ((pad_left, pad_right), (0, 0)),
                                mode="constant", constant_values=np.nan)

        # Baseline
        if baseline:
            # print("Baseline", np.nanmean(epoch_data[baseline_window_left:baseline_window_right], axis=0))
            epoch_data -= np.nanmean(epoch_data[baseline_window_left:baseline_window_right], axis=0)

        epochs[i] = epoch_data

    # Convert to DataFrame.
    epochs_df = pd.DataFrame([
        (i, sample, sensor, epochs[i, sample, sensor])
        for i, sample, sensor in np.ndindex(*epochs.shape)
    ], columns=["epoch", "sample", "sensor_idx", "value"])
    return epochs_df


def make_epochs(dataset: BerpDataset, epoch_points: TensorType[float],
                tmin=-0.1, tmax=0.7,
                baseline=True,
                baseline_tmin=-0.1, baseline_tmax=0.0,
                ts: Optional[TensorType["n_samples", "n_sensors", float]] = None,
                ) -> pd.DataFrame:
    """
    Generate an epoched time series dataset from a BerpDataset.
    """
    # DEV for now assume things cut cleanly w.r.t. sample rate
    sample_rate = dataset.sample_rate
    assert int(tmin * sample_rate) == tmin * sample_rate
    assert int(tmax * sample_rate) == tmax * sample_rate

    if baseline:
        assert int(baseline_tmin * sample_rate) == baseline_tmin * sample_rate
        assert int(baseline_tmax * sample_rate) == baseline_tmax * sample_rate

    epoch_idxs = torch.arange(len(epoch_points))
    epoch_samples = (epoch_points * sample_rate).long()

    epoch_shift_left = int(tmin * sample_rate)
    epoch_shift_right = int(tmax * sample_rate)

    ts = dataset.Y if ts is None else ts

    # Detect epoch points that exceed time series bounds. Drop and warn.
    out_of_bounds_mask = epoch_samples > ts.shape[0]
    if out_of_bounds_mask.any():
        L.warning(f"dropping {out_of_bounds_mask.sum()} epochs that exceed time series bounds")

        # Track idxs of retained epochs so that they can be matched up
        # with events.
        epoch_idxs = epoch_idxs[~out_of_bounds_mask]
        epoch_samples = epoch_samples[~out_of_bounds_mask]

    epochs_df = epoch_ts(
        ts, epoch_samples,
        epoch_shift_left, epoch_shift_right,
        baseline=baseline,
        baseline_window_left=int((baseline_tmin - tmin) * sample_rate),
        baseline_window_right=int((baseline_tmax - tmin) * sample_rate))

    epochs_df["epoch_time"] = epochs_df["sample"] / sample_rate + tmin

    # Map epoch idx onto originating epoch idx
    epochs_df["epoch"] = epochs_df.epoch.map(dict(enumerate(epoch_idxs.numpy())))
    epochs_df = epochs_df.set_index(["epoch", "sample", "sensor_idx"])

    return epochs_df


def make_word_onset_epochs(
    dataset: BerpDataset,
    word_idxs: Optional[TensorType[torch.long]] = None,
    **kwargs):
    """
    Generate an epoched dataset where t=0 <=> word onset.
    """

    if word_idxs is None:
        word_idxs = torch.arange(dataset.n_words)
    epoch_points = dataset.word_onsets[word_idxs]

    ret = make_epochs(dataset, epoch_points, **kwargs)
    ret["word_idx"] = ret.index.get_level_values("epoch") \
        .map(dict(enumerate(word_idxs.numpy())))

    return ret


def make_word_recognition_epochs(
    dataset: BerpDataset,
    recognition_points: TensorType["batch", torch.long],
    recognition_times: TensorType["batch", float],
    word_idxs: Optional[TensorType["batch", torch.long]] = None,
    **kwargs):
    """
    Generate an epoched dataset where t=0 <=> word recognition time.
    """

    ret = make_epochs(dataset, recognition_times, **kwargs)

    if word_idxs is None:
        word_idxs = torch.arange(dataset.n_words)
    ret["word_idx"] = ret.index.get_level_values("epoch") \
        .map(dict(enumerate(word_idxs.numpy())))
    ret["recognition_point"] = ret.word_idx.map(dict(zip(word_idxs.numpy(), recognition_points.numpy())))
    ret["recognition_time"] = ret.word_idx.map(dict(zip(word_idxs.numpy(), recognition_times.numpy())))

    return ret