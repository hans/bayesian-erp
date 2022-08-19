from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import logging
from typing import List, Optional, Callable, Dict, Tuple, Union

import numpy as np
import torch
from torch.nn.functional import pad
from torchtyping import TensorType
from tqdm.auto import tqdm, trange
from typeguard import typechecked

from berp.typing import DIMS, is_log_probability

L = logging.getLogger(__name__)

# Type variables
B, N_W, N_C, N_F, N_F_T, N_P, V_W = \
    DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_F_T, DIMS.N_P, DIMS.V_W
T, S = DIMS.T, DIMS.S

# Type aliases
Phoneme = str
intQ = Optional[Union[int, np.integer]]

def default_phonemizer(string) -> List[Phoneme]:
    return list(string)


@typechecked
@dataclass
class BerpDataset:
    """
    Defines a time series dataset for reindexing regression.

    The predictors are stored in two groups:

    1. `X_ts`: Time-series predictors, which are sampled at the same rate as `Y`.
    2. `X_variable`: Latent-onset predictors, `batch` many whose onset is to be inferred
       by the model.

    All tensors are padded on the N_P axis on the right to the maximum word length.
    """

    name: str

    sample_rate: int

    phonemes: List[str]
    """
    Phoneme vocabulary.
    """

    p_word: TensorType[B, N_C, is_log_probability]
    """
    Predictive distribution over expected candidate words at each time step,
    derived from a language model.
    """

    word_lengths: TensorType[B, int]
    """
    Length of ground-truth words in phonemes. Can be used to unpack padded
    ``N_P`` axes.
    """

    candidate_phonemes: TensorType[B, N_C, N_P, int]
    """
    Phoneme ID sequence for each word and alternate candidate set.
    """

    word_onsets: TensorType[B, float]
    """
    Onset of each word in seconds, relative to the start of the sequence.
    """

    phoneme_onsets: TensorType[B, N_P, float]
    """
    Onset of each phoneme within each word in seconds, relative to the start of
    the corresponding word. Column axis should be padded with 0s.
    """

    X_ts: TensorType[T, N_F_T, float]

    X_variable: TensorType[B, N_F, float]
    """
    Word-level features whose onset is to be determined by the model.
    """

    Y: TensorType[T, S, float]
    """
    Response data.
    """

    def __len__(self):
        return self.Y.shape[0]

    @property
    def n_samples(self):
        return len(self)

    @property
    def n_total_features(self):
        return self.X_variable.shape[1] + self.X_ts.shape[1]

    @property
    def n_sensors(self):
        return self.Y.shape[1]

    def __getitem__(self, key) -> "BerpDataset":
        """
        Extract a number of samples from the dataset.
        The resulting dataset has adjusted times to match the new sample start point.
        """
        if isinstance(key, slice):
            if key.step is not None:
                raise ValueError("Step size not supported.")

            start_sample = key.start or 0
            end_sample = key.stop or self.n_samples

            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate

            # Find which word indices should be retained for these time boundaries.
            # TODO add some slack on end? we don't want to include words exactly at the right boundary
            word_mask = (self.word_onsets >= start_time) & (self.word_onsets <= end_time)
            keep_word_indices = torch.where(word_mask)[0]

            # Subset word-level features.
            word_onsets = self.word_onsets[keep_word_indices]
            phoneme_onsets = self.phoneme_onsets[keep_word_indices]
            X_variable = self.X_variable[keep_word_indices]

            # Subtract onset data so that t=0 -> sample 0.
            word_onsets = word_onsets - start_time
            phoneme_onsets = phoneme_onsets - start_time

            ret = dataclasses.replace(self,
                name=f"{self.name}/slice:{start_sample}:{end_sample}",

                p_word=self.p_word[keep_word_indices],
                word_lengths=self.word_lengths[keep_word_indices],
                candidate_phonemes=self.candidate_phonemes[keep_word_indices],

                word_onsets=word_onsets,
                phoneme_onsets=phoneme_onsets,

                X_ts=self.X_ts[key],
                X_variable=X_variable,

                Y=self.Y[key],
            )

            return ret

        return super().__getitem__(key)

    def ensure_torch(self, dtype=torch.float32) -> BerpDataset:
        """
        Convert all tensors to torch tensors.
        """
        self.p_word = torch.as_tensor(self.p_word, dtype=torch.float32)
        self.word_lengths = torch.as_tensor(self.word_lengths)
        self.candidate_phonemes = torch.as_tensor(self.candidate_phonemes)
        self.word_onsets = torch.as_tensor(self.word_onsets, dtype=torch.float32)
        self.phoneme_onsets = torch.as_tensor(self.phoneme_onsets, dtype=torch.float32)
        self.X_ts = torch.as_tensor(self.X_ts, dtype=torch.float32)
        self.X_variable = torch.as_tensor(self.X_variable, dtype=torch.float32)
        self.Y = torch.as_tensor(self.Y, dtype=torch.float32)

        return self

    def subset_sensors(self, sensors: List[int]) -> BerpDataset:
        """
        Subset sensors in response variable. Returns a copy.
        """
        return dataclasses.replace(self, Y=self.Y[:, sensors])


class NestedBerpDataset(object):
    """
    Represents a grouped Berp dataset as a list of time series intervals.
    This makes the data amenable to cross validation by the standard sklearn
    API -- can index via integer values.

    Each element in the resulting dataset corresponds to a fraction of an
    original subject's sub-dataset, `1/n_splits` large.
    """

    def __init__(self, datasets: List[BerpDataset], n_splits=2):
        # Shape checks. Everything but batch axis should match across
        # subjects. Batch axis should match within-subject between
        # X and Y.
        for dataset in datasets:
            ds0 = datasets[0]
            assert dataset.X_ts.shape[1:] == ds0.X_ts.shape[1:]
            assert dataset.X_variable.shape[1:] == ds0.X_variable.shape[1:]
            assert dataset.Y.shape[1:] == ds0.Y.shape[1:]
            assert dataset.X_ts.shape[0] == dataset.Y.shape[0]

        self.datasets = datasets
        self.n_datasets = len(datasets)
        self.set_n_splits(n_splits)

    def set_n_splits(self, n_splits):
        self.n_splits = n_splits

        # Maps integer indices on this dataset into slices of individual
        # sub-datasets.
        self.flat_idxs: List[Tuple[int, slice]] = []
        for i, dataset in enumerate(self.datasets):
            split_size = int(np.ceil(len(dataset) / self.n_splits))
            for split_offset in range(0, len(dataset), split_size):
                self.flat_idxs.append(
                    (i, slice(split_offset, split_offset + split_size)))

    @property
    def shape(self):
        # Define shape property so that sklearn indexing thinks we're an
        # ndarray, and will index with an ndarray of indices rather than
        # scalars+concatenate.
        # Then we can make sure the output is still a NestedBerpDataset :)
        return (len(self),)

    @property
    def n_total_features(self):
        ds = self.datasets[0]
        return ds.X_variable.shape[1] + ds.X_ts.shape[1]

    @property
    def n_sensors(self):
        return self.datasets[0].n_sensors

    # TODO will be super slow to always typecheck. remove once we know this
    # works
    @typechecked
    def __getitem__(self, key: Union[int, np.integer, np.ndarray]
                    ) -> Union[BerpDataset, NestedBerpDataset]:
        if isinstance(key, (int, np.integer)):
            dataset, split = self.flat_idxs[key]
            return self.datasets[dataset][split]
        elif isinstance(key, np.ndarray):
            return NestedBerpDataset([self[i] for i in key],
                                     n_splits=self.n_splits)
        else:
            raise NotImplementedError(f"Unsupported key type {type(key)}")

    def __len__(self):
        return len(self.flat_idxs)

    def iter_datasets(self):
        return iter(self.datasets)

    def subset_sensors(self, sensors: List[int]) -> NestedBerpDataset:
        """
        Subset sensors in response variable. Returns a copy.
        """
        return NestedBerpDataset(
            [dataset.subset_sensors(sensors) for dataset in self.datasets],
            n_splits=self.n_splits)