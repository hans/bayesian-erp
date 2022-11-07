from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torchtyping import TensorType  # type: ignore

from berp.datasets import BerpDataset


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
    for i, samp_i in enumerate(tqdm(epoch_points)):
        epoch_data = data[samp_i + epoch_shift_left:samp_i + epoch_shift_right].clone()

        # Baseline
        if baseline:
            epoch_data -= epoch_data[baseline_window_left:baseline_window_right].mean(axis=0)
        
        if epoch_data.shape[0] < epoch_n_samples:
            # pad with NaNs
            epoch_data = np.concatenate(
                [epoch_data,
                 np.empty((epoch_n_samples - epoch_data.shape[0], n_sensors)) * np.nan],
                axis=0)

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
    assert int(baseline_tmin * sample_rate) == baseline_tmin * sample_rate
    assert int(baseline_tmax * sample_rate) == baseline_tmax * sample_rate

    epoch_samples = (epoch_points * sample_rate).long()

    epoch_shift_left = int(tmin * sample_rate)
    epoch_shift_right = int(tmax * sample_rate)

    ts = dataset.Y if ts is None else ts
    epochs_df = epoch_ts(
        ts, epoch_samples,
        epoch_shift_left, epoch_shift_right,
        baseline=baseline,
        baseline_window_left=int((baseline_tmin - tmin) * sample_rate),
        baseline_window_right=int((baseline_tmax - tmin) * sample_rate))

    epochs_df["epoch_time"] = epochs_df["sample"] / sample_rate + tmin
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