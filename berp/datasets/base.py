from __future__ import annotations
import dataclasses
from dataclasses import dataclass, replace
import logging
from pprint import pformat
from typing import List, Optional, Callable, Dict, Tuple, Union, Iterator, cast

import numpy as np
import torch
from torchtyping import TensorType
from typeguard import typechecked

from berp.datasets import NaturalLanguageStimulus
from berp.typing import DIMS, is_log_probability, is_positive, is_nonnegative

L = logging.getLogger(__name__)

# Type variables
B, N_W, N_C, N_F, N_F_T, N_P, V_W = \
    DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_F_T, DIMS.N_P, DIMS.V_W
T, S = DIMS.T, DIMS.S

# Type aliases
Phoneme = str
intQ = Optional[Union[int, np.integer]]


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
    stimulus_name: str

    sample_rate: int

    word_onsets: TensorType[B, float, is_nonnegative]
    """
    Onset of each word in seconds, relative to the start of the sequence.
    """

    word_offsets: TensorType[B, float, is_nonnegative]
    """
    Offset of each word in seconds, relative to the start of the sequence.
    """

    phoneme_onsets: TensorType[B, N_P, float, is_nonnegative]
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

    sensor_names: Optional[List[str]] = None
    """
    Names of sensors (columns of `Y`).
    """

    ### Stimulus data shared between subjects, may be saved separately

    phonemes: Optional[List[str]] = None
    """
    Phoneme vocabulary.
    """

    p_candidates: Optional[TensorType[B, N_C, is_log_probability]] = None
    """
    Predictive distribution over expected candidate words at each time step,
    derived from a language model.
    """

    word_lengths: Optional[TensorType[B, int]] = None
    """
    Length of ground-truth words in phonemes. Can be used to unpack padded
    ``N_P`` axes.
    """

    candidate_phonemes: Optional[TensorType[B, N_C, N_P, int]] = None
    """
    Phoneme ID sequence for each word and alternate candidate set.
    """

    ### Dynamic data

    global_slice_indices: Optional[Tuple[int, int]] = None
    """
    If this dataset corresponds to a slice of a larger time series dataset,
    store the index of this dataset's onset and offset within that dataset
    (in samples). Otherwise is `None`, indicating that this dataset is not
    a slice of a larger time series.
    """

    ts_feature_names: Optional[List[str]] = None
    variable_feature_names: Optional[List[str]] = None

    def __post_init__(self):
        self.check_shapes()

    @property
    def base_name(self):
        """
        Get the name of the base dataset associated with this data, prior to
        slicing.
        """
        if "slice:" in self.name:
            return self.name[:self.name.index("/slice:")]
        return self.name

    def __len__(self):
        return self.Y.shape[0]

    @property
    def dtype(self):
        return self.Y.dtype

    @property
    def n_samples(self):
        return len(self)

    @property
    def n_words(self):
        return self.word_onsets.shape[0]

    @property
    def n_ts_features(self):
        return self.X_ts.shape[1]

    @property
    def n_variable_features(self):
        return self.X_variable.shape[1]

    @property
    def n_total_features(self):
        return self.n_ts_features + self.n_variable_features

    @property
    def n_sensors(self):
        return self.Y.shape[1]

    @property
    def n_phonemes(self):
        """Number of available phoneme types"""
        return len(self.phonemes)

    @property
    def max_n_phonemes(self):
        """
        The maximum length of a word/candidate in the representation, in
        phonemes.
        """
        return self.phoneme_onsets.shape[1]

    @property
    def n_candidates(self):
        """
        Number of represented candidate completions for each context.
        """
        return self.p_candidates.shape[1]
    
    @property
    def phoneme_onsets_global(self) -> TensorType[B, N_P, float, is_nonnegative]:
        """
        Onset of each phoneme within each word in seconds, relative to the start of
        the time series.
        """
        return self.word_onsets[:, None] + self.phoneme_onsets

    @property
    def phoneme_offsets(self) -> TensorType[B, N_P, float, is_nonnegative]:
        """
        Offset of each phoneme within each word in seconds, relative to the onset of
        the word.
        """
        return torch.cat([
            self.phoneme_onsets[:, 1:],
            self.word_offsets[:, None] - self.word_onsets[:, None],
        ], 1)

    @property
    def phoneme_offsets_global(self) -> TensorType[B, N_P, float, is_nonnegative]:
        """
        Offset of each phoneme within each word in seconds, relative to the start of
        the time series.
        """
        return torch.cat([
            self.phoneme_onsets_global[:, 1:],
            self.word_offsets[:, None]
        ], 1)

    @typechecked
    def __getitem__(self, key) -> "BerpDataset":
        """
        Extract a number of samples from the dataset.
        The resulting dataset has adjusted times to match the new sample start point.
        """
        if isinstance(key, slice):
            if key.step is not None:
                raise ValueError("Step size not supported.")
            if key.start < 0 or key.stop < 0:
                raise ValueError("Negative indices not supported.")

            start_sample = int(key.start) or 0
            end_sample = int(key.stop) or self.n_samples

            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate

            # Find which word indices should be retained for these time boundaries.
            # TODO add some slack on end? we don't want to include words exactly at the right boundary
            word_mask = (self.word_onsets >= start_time) & (self.word_onsets <= end_time)
            keep_word_indices = torch.where(word_mask)[0]

            # Subset word-level features.
            word_onsets = self.word_onsets[keep_word_indices]
            word_offsets = self.word_offsets[keep_word_indices]
            phoneme_onsets = self.phoneme_onsets[keep_word_indices]
            X_variable = self.X_variable[keep_word_indices]

            # Subtract onset data so that t=0 -> sample 0.
            # NB phoneme_onsets is relative to word onset, so we don't subtract here.
            word_onsets = word_onsets - start_time
            word_offsets = word_offsets - start_time

            # Retain global slicing data matching the result's samples to samples in the
            # original dataset.
            if self.global_slice_indices is None:
                global_slice_indices = (start_sample, end_sample)
            else:
                orig_start, orig_end = self.global_slice_indices
                global_slice_indices = (
                    orig_start + start_sample,
                    orig_start + start_sample + (end_sample - start_sample)
                )

            ret = dataclasses.replace(self,
                name=f"{self.name}/slice:{start_sample}:{end_sample}",

                p_candidates=self.p_candidates[keep_word_indices],
                word_lengths=self.word_lengths[keep_word_indices],
                candidate_phonemes=self.candidate_phonemes[keep_word_indices],

                word_onsets=word_onsets,
                word_offsets=word_offsets,
                phoneme_onsets=phoneme_onsets,

                X_ts=self.X_ts[key],
                X_variable=X_variable,

                Y=self.Y[key],

                global_slice_indices=global_slice_indices,
            )

            return ret

        return super().__getitem__(key)

    def check_shapes(self):
        """
        Check that all data arrays have the expected shape.
        """
        assert self.word_onsets.shape == (self.n_words,)
        assert self.word_offsets.shape == (self.n_words,)
        assert self.phoneme_onsets.shape == (self.n_words, self.max_n_phonemes)
        assert self.X_ts.shape == (self.n_samples, self.n_ts_features)
        assert self.X_variable.shape == (self.n_words, self.n_variable_features)
        assert self.Y.shape == (self.n_samples, self.n_sensors)

        if self.sensor_names is not None:
            assert self.Y.shape[1] == len(self.sensor_names)

        if self.p_candidates is not None:
            assert self.p_candidates.shape == (self.n_words, self.n_candidates)
            assert self.word_lengths.shape == (self.n_words,)
            assert self.candidate_phonemes.shape == (self.n_words, self.n_candidates, self.max_n_phonemes)

        if self.ts_feature_names is not None:
            assert len(self.ts_feature_names) == self.n_ts_features
        if self.variable_feature_names is not None:
            assert len(self.variable_feature_names) == self.n_variable_features

    def ensure_torch(self, device: Optional[str] = None, dtype=torch.float32) -> BerpDataset:
        """
        Convert all tensors to torch tensors.
        """
        self.word_onsets = torch.as_tensor(self.word_onsets, dtype=dtype).to(device)
        self.word_offsets = torch.as_tensor(self.word_offsets, dtype=dtype).to(device)
        self.phoneme_onsets = torch.as_tensor(self.phoneme_onsets, dtype=dtype).to(device)
        self.X_ts = torch.as_tensor(self.X_ts, dtype=dtype).to(device)
        self.X_variable = torch.as_tensor(self.X_variable, dtype=dtype).to(device)
        self.Y = torch.as_tensor(self.Y, dtype=dtype).to(device)

        if self.p_candidates is not None:
            self.p_candidates = torch.as_tensor(self.p_candidates, dtype=dtype).to(device)
            self.word_lengths = torch.as_tensor(self.word_lengths).to(device)
            self.candidate_phonemes = torch.as_tensor(self.candidate_phonemes).to(device)

        return self

    def add_stimulus(self, stimulus: NaturalLanguageStimulus) -> None:
        """
        Add stimulus information to the dataset in-place.
        """
        if self.p_candidates is not None:
            raise RuntimeError("Dataset already has stimulus information. Stop.")
        elif self.stimulus_name != stimulus.name:
            raise ValueError(f"Stimulus name does not match. {self.stimulus_name} != {stimulus.name}.")

        self.phonemes = stimulus.phonemes

        # Reference tensor for dtype/device
        ref_tensor = self.word_onsets

        self.p_candidates = stimulus.p_candidates.to(ref_tensor)
        self.word_lengths = stimulus.word_lengths.to(ref_tensor.device)
        self.candidate_phonemes = stimulus.candidate_phonemes

    def subset_sensors(self, sensors: Union[List[int], List[str]]) -> None:
        """
        Subset sensors in response variable. Operates in place.
        """
        sensor_idxs, sensor_names = [], []
        for sensor in sensors:
            if isinstance(sensor, int):
                if sensor > self.n_sensors:
                    raise ValueError(f"Sensor index {sensor} out of range.")

                sensor_idxs.append(sensor)
                sensor_names.append(self.sensor_names[sensor])
            elif isinstance(sensor, str):
                if self.sensor_names is None:
                    raise ValueError(
                        "Dataset has no sensor names but string sensor identifiers passed.")

                try:
                    sensor_idx = self.sensor_names.index(sensor)
                except ValueError:
                    raise ValueError(f"Sensor name {sensor} not found.")
                else:
                    sensor_idxs.append(self.sensor_names.index(sensor))
                    sensor_names.append(sensor)

        self.Y = self.Y[:, sensor_idxs]
        self.sensor_names = sensor_names

    def average_sensors(self) -> None:
        """
        Average across sensor values per sample, creating a univariate
        response. Operates in place.
        """
        # TODO outliers?
        sensor_names = ["average"] if self.sensor_names is None \
            else ["average_%s" % "_".join(self.sensor_names)]

        self.Y = self.Y.mean(dim=1, keepdim=True)
        self.sensor_names = sensor_names

    def _check_feature_spec(self,
                            ts: Optional[Union[List[int], List[str]]] = None,
                            variable: Optional[Union[List[int], List[str]]] = None
                            ) -> Tuple[List[int], List[int]]:
        """
        Check and convert the given feature lists to a list of column indices into
        `X_ts` and `X_variable`.
        """
        if ts is None:
            ts = list(range(self.n_ts_features))
        elif ts and isinstance(ts[0], str):
            if self.ts_feature_names is None:
                raise ValueError("Dataset has no time-series feature names but string identifiers passed.")
            try:
                ts = [self.ts_feature_names.index(f) for f in ts]
            except ValueError:
                L.error("Available time-series features in dataset:")
                L.error(pformat(self.ts_feature_names))
                raise
        if variable is None:
            variable = list(range(self.n_variable_features))
        elif variable and isinstance(variable[0], str):
            if self.variable_feature_names is None:
                raise ValueError("Dataset has no variable feature names but string identifiers passed.")
            try:
                variable = [self.variable_feature_names.index(f) for f in variable]
            except ValueError:
                L.error("Available variable features in dataset:")
                L.error(pformat(self.variable_feature_names))
                raise
        return ts, variable

    def get_features(self,
                     ts: Optional[Union[List[int], List[str]]] = None,
                     variable: Optional[Union[List[int], List[str]]] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the given time-series and variable features as tensors.

        Returns:
            ts_tensor:
            variable_tensor:
        """
        ts, variable = self._check_feature_spec(ts, variable)
        return self.X_ts[:, ts], self.X_variable[:, variable]

    def select_features(self,
                        ts: Optional[Union[List[int], List[str]]] = None,
                        variable: Optional[Union[List[int], List[str]]] = None) -> None:
        """
        Subset with the given ordered time-series and variable features.
        Operates in place.
        """
        ts, variable = self._check_feature_spec(ts, variable)

        self.X_ts = self.X_ts[:, ts]
        self.X_variable = self.X_variable[:, variable]
        self.ts_feature_names = [self.ts_feature_names[i] for i in ts] \
            if self.ts_feature_names is not None else None
        self.variable_feature_names = [self.variable_feature_names[i] for i in variable] \
            if self.variable_feature_names is not None else None

    # Avoid saving stimulus data in pickle data
    def __getstate__(self):
        state = self.__dict__.copy()
        for k in ["phonemes", "p_candidates", "word_lengths", "candidate_phonemes"]:
            state[k] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class NestedBerpDataset(object):
    """
    Represents a grouped Berp dataset as a list of time series intervals.
    This makes the data amenable to cross validation by the standard sklearn
    API -- can index via integer values.

    Each element in the resulting dataset corresponds to a fraction of an
    original subject's sub-dataset, `1/n_splits` large.
    """

    def __init__(self, datasets: List[BerpDataset], n_splits=2):
        for dataset in datasets:
            assert_compatible(dataset, datasets[0])

        self.datasets = datasets
        self.n_datasets = len(datasets)
        self.set_n_splits(n_splits)

    def set_n_splits(self, n_splits):
        self.n_splits = n_splits

        # Maps integer indices on this dataset into slices of individual
        # sub-datasets (slices represented as int start:end).
        flat_idxs: List[Tuple[int, int, int]] = []
        for i, dataset in enumerate(self.datasets):
            split_size = int(np.ceil(len(dataset) / self.n_splits))
            for split_offset in range(0, len(dataset), split_size):
                flat_idxs.append(
                    (i, split_offset, min(len(dataset), split_offset + split_size)))
        self.flat_idxs = np.array(flat_idxs)

    def order_by_dataset(self) -> "NestedBerpDataset":
        """
        Order the dataset by dataset, then by time.
        """
        ordered = np.lexsort(self.flat_idxs[:, [1, 0]].T)
        self.flat_idxs = self.flat_idxs[ordered]
        return self

    def order_by_time(self) -> "NestedBerpDataset":
        """
        Reorder splits across dataset by time, then by dataset.
        """
        ordered = np.lexsort(self.flat_idxs[:, [0, 1]].T)
        self.flat_idxs = self.flat_idxs[ordered]
        return self

    @property
    def dtype(self):
        return self.datasets[0].dtype

    @property
    def sample_rate(self):
        return self.datasets[0].sample_rate

    @property
    def phonemes(self):
        return self.datasets[0].phonemes

    @property
    def n_phonemes(self):
        return self.datasets[0].n_phonemes

    @property
    def shape(self):
        # Define shape property so that sklearn indexing thinks we're an
        # ndarray, and will index with an ndarray of indices rather than
        # scalars+concatenate.
        # Then we can make sure the output is still a NestedBerpDataset :)
        return (len(self),)

    @property
    def n_ts_features(self):
        return self.datasets[0].n_ts_features

    @property
    def n_variable_features(self):
        return self.datasets[0].n_variable_features

    @property
    def n_total_features(self):
        return self.datasets[0].n_total_features

    @property
    def n_sensors(self):
        return self.datasets[0].n_sensors

    @property
    def ts_feature_names(self):
        return self.datasets[0].ts_feature_names

    @property
    def variable_feature_names(self):
        return self.datasets[0].variable_feature_names

    # TODO will be super slow to always typecheck. remove once we know this
    # works
    @typechecked
    def __getitem__(self, key: Union[int, np.integer, np.ndarray]
                    ) -> Union[BerpDataset, NestedBerpDataset]:
        """
        Note that the order of the keys will not be respected in the case
        where an ndarray is used for indexing. Instead, datasets will be
        returned in an order that maximizes contiguity of time series.
        """
        if isinstance(key, (int, np.integer)):
            dataset, split_start, split_end = self.flat_idxs[key]
            return self.datasets[dataset][split_start:split_end]
        elif isinstance(key, np.ndarray):
            flat_idxs = self.flat_idxs[key]

            # Sort by dataset, then by start time
            flat_idxs = flat_idxs[np.lexsort(flat_idxs[:, [1, 0]].T)]

            # Slice in a way that keeps subdatasets maximally contiguous.
            # We only want to slice when 1) the subdataset index changes, or
            # 2) the time series indices are not contiguous.
            grouped_idxs = np.split(
                flat_idxs,
                np.where((np.diff(flat_idxs[:, 0]) != 0) |
                         (flat_idxs[:-1, 2] != flat_idxs[1:, 1]))[0] + 1)

            ret = []
            for group in grouped_idxs:
                dataset, split_start, _ = group[0]
                split_end = group[-1][2]

                if split_start == 0 and split_end >= len(self.datasets[dataset]):
                    # No need to slice. We are using the whole subdataset.
                    ret.append(self.datasets[dataset])
                else:
                    ret.append(self.datasets[dataset][split_start:split_end])

            return NestedBerpDataset(ret, n_splits=self.n_splits)
        else:
            raise NotImplementedError(f"Unsupported key type {type(key)}")

    def __len__(self):
        return len(self.flat_idxs)

    def __iter__(self) -> Iterator[BerpDataset]:
        return (self[i] for i in range(len(self)))

    def iter_datasets(self):
        return iter(self.datasets)

    @property
    def names(self):
        return [ds.name for ds in self.datasets]

    def __repr__(self):
        """
        Return a summary of all the datasets contained in this dataset.
        Merge contiguous subdatasets in this summary.
        """

        descriptions = [(x.name, x.global_slice_indices, len(x)) for x in self]
        merged = []
        for name, slice_bounds, length in descriptions:
            if slice_bounds is None:
                merged.append((name, (0, length)))
                continue

            name = name.split("/")[0]
            start, end = slice_bounds
            if merged and merged[-1][0] == name and merged[-1][1][1] == start:
                merged[-1] = (name, (merged[-1][1][0], end))
            else:
                merged.append((name, (start, end)))

        return f"NestedBerpDataset({', '.join(f'{name}[{start}:{end}]' for name, (start, end) in merged)})"

    def subset_sensors(self, sensors: List[int]) -> None:
        """
        Subset sensors in response variable. Operates in place.
        """
        for ds in self.datasets:
            ds.subset_sensors(sensors)

    def average_sensors(self) -> None:
        """
        Average sensors in response variable. Operates in place.
        """
        for ds in self.datasets:
            ds.average_sensors()

    def select_features(self,
                        ts: Optional[Union[List[int], List[str]]] = None,
                        variable: Optional[Union[List[int], List[str]]] = None) -> None:
        """
        Subset with the given ordered time-series and variable features.
        Operates in place.
        """
        for ds in self.datasets:
            ds.select_features(ts, variable)


def assert_compatible(ds1: BerpDataset, ds2: BerpDataset):
    """
    Assert that the two datasets are compatible for concatenation.
    This is possible when they have the same feature set, same number of sensors.
    They DO NOT need to be in response to the same stimulus.
    """

    # Shape checks. Everything but batch axis should match across
    # subjects. Batch axis should match within-subject between
    # X and Y.
    assert ds1.dtype == ds2.dtype
    assert ds1.sample_rate == ds2.sample_rate
    assert ds1.X_ts.shape[1:] == ds2.X_ts.shape[1:]
    assert ds1.X_variable.shape[1:] == ds2.X_variable.shape[1:]
    assert ds1.Y.shape[1:] == ds2.Y.shape[1:]
    assert ds1.X_ts.shape[0] == ds1.Y.shape[0]

    # These may be none
    assert ds1.sensor_names == ds2.sensor_names

    if ds2.ts_feature_names is not None:
        assert ds2.ts_feature_names == ds1.ts_feature_names
    if ds2.variable_feature_names is not None:
        assert ds2.variable_feature_names == ds1.variable_feature_names


def average_datasets(datasets: List[BerpDataset], name="average"):
    """
    Create a dataset whose response time series is the average of the given datasets'
    response time series.
    Asserts that all other features match between datasets.
    """

    for ds in datasets[1:]:
        ds0 = datasets[0]
        assert_compatible(ds, ds0)

        # We need more than compatibility -- the presentations should 
        # all be matched in order to allow for averaging.
        assert torch.allclose(ds.word_onsets, ds0.word_onsets)
        assert torch.allclose(ds.word_offsets, ds0.word_offsets)
        assert torch.allclose(ds.phoneme_onsets, ds0.phoneme_onsets)
        assert torch.allclose(ds.X_ts, ds0.X_ts)
        assert torch.allclose(ds.X_variable, ds0.X_variable)

    Y_avg = torch.stack([ds.Y for ds in datasets]).mean(0)
    return replace(datasets[0], Y=Y_avg, name=name)