"""
Module glossary:
    - feature: a dataset feature (e.g. surprisal)
    - predictor: corresponds to a column in a design matrix. may be directly
      read from a feature value, or may be a derived value (e.g. an interaction
      between two features).

"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import singledispatchmethod
import logging
import pickle
import re
from typing import Optional, List, Dict, Union, Tuple, \
    TypeVar, Generic, Iterator, Type, cast, Hashable

from hydra.utils import to_absolute_path
import numpy as np
from optuna.distributions import BaseDistribution, UniformDistribution
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ShuffleSplit
import torch
from torchtyping import TensorType  # type: ignore
from tqdm.auto import tqdm, trange
from typeguard import typechecked

from berp.config import FeatureConfig
from berp.cv import EarlyStopException
from berp.datasets import BerpDataset, NestedBerpDataset, assert_concatenatable
from berp.models.reindexing_regression import \
    predictive_model, recognition_point_model, recognition_points_to_times, \
    PartiallyObservedModelParameters, ModelParameters
from berp.models.trf import TemporalReceptiveField, TRFPredictors, \
    TRFDesignMatrix, TRFResponse, TRFDelayer
from berp.tensorboard import Tensorboard
from berp.typing import is_probability, DIMS
from berp.util import time_to_sample, cat

L = logging.getLogger(__name__)

# Type variables
B, N_C, N_P, N_F, N_F_T, V_P, T, S = \
    DIMS.B, DIMS.N_C, DIMS.N_P, DIMS.N_F, DIMS.N_F_T, DIMS.V_P, DIMS.T, DIMS.S
P = "num_params"
Responsibilities = TensorType[P, is_probability]
MaskArray = TensorType[torch.bool]


@typechecked
def scatter_add(design_matrix: TRFDesignMatrix,
                target_samples: TensorType[B, torch.long],
                target_values: TensorType[B, "n_target_features"],
                lag_mask: Optional[TensorType["n_delays", torch.bool]] = None,
                add=True) -> None:
    """
    Scatter-add or update values to the given samples of the TRF design matrix,
    maintaining lag structure (i.e. duplicating target values at different lags).

    By default, the number of lags added is just the same as the size of the
    design matrix's final axis. If `lag_mask` is provided, just those lags for
    which `lag_mask[i]` is `True` will be updated. This allows imposing e.g.
    a custom lag structure for different modifications.

    If `add` is `True`, scatter-add to existing values; otherwise replace with
    given values.

    Operates in-place.
    """

    assert target_samples.shape[0] == target_values.shape[0]
    assert target_values.shape[1] == design_matrix.shape[1]
    assert target_values.dtype == design_matrix.dtype

    # NB no copy.
    out = design_matrix

    delay_idxs = torch.where(lag_mask)[0] if lag_mask is not None else range(out.shape[2])
    for delay in delay_idxs:
        # Mask out items which, with this delay, would exceed the right edge
        # of the time series.
        mask = target_samples + delay < out.shape[0]

        if add:
            out[target_samples[mask] + delay, :, delay] += target_values[mask]
        else:
            out[target_samples[mask] + delay, :, delay] = target_values[mask]


@typechecked
def scatter_variable(dataset: BerpDataset,
                     times: TensorType[B, float],
                     out: TRFDesignMatrix,
                     out_weight: float = 1.,
                     variable_features: Optional[Union[List[int], List[str]]] = None,
                     lag_mask: Optional[TensorType["n_delays", torch.bool]] = None,
                     ) -> None:
    """
    Scatter variable-onset predictors into design matrix, which is a
    lagged time series. Operates in place.

    Args:
        dataset:
        times:
        out: scatter-add to this tensor
        out_weight: apply this weight to the scatter-add.
        lag_mask: if provided, only scatter-add to lags for which this mask is
            `True`.
    """

    _, values = dataset.get_features(variable=variable_features)
    assert len(times) == values.shape[0]
    feature_start_idx = dataset.n_ts_features

    # Compute recognition onset times and convert to sample representation.
    samples = time_to_sample(times, dataset.sample_rate)

    # Scatter-add, lagging over delay axis.
    to_add = out_weight * values
    target = torch.narrow(out, 1, feature_start_idx, values.shape[1])
    scatter_add(target, samples, to_add, lag_mask)


def _make_validation_mask(dataset: BerpDataset, validation_fraction: float,
                          random_state=None) -> TensorType[torch.bool]:
    y = dataset.Y

    n_samples = y.shape[0]
    validation_mask = torch.zeros(n_samples).bool()

    # TODO valid sample boundaries?
    cv = ShuffleSplit(
        test_size=validation_fraction, random_state=random_state
    )
    idx_train, idx_val = next(cv.split(np.zeros(shape=(y.shape[0], 1)), y))
    if idx_train.shape[0] == 0 or idx_val.shape[0] == 0:
        raise ValueError(
            "Splitting %d samples into a train set and a validation set "
            "with validation_fraction=%r led to an empty set (%d and %d "
            "samples). Please either change validation_fraction, increase "
            "number of samples, or disable early_stopping."
            % (
                n_samples,
                validation_fraction,
                idx_train.shape[0],
                idx_val.shape[0],
            )
        )

    validation_mask[idx_val] = True
    return validation_mask


@dataclass
class ForwardPipelineCache:
    """
    For each dataset (subject--story), persists data that should be shared and can
    be efficiently reused across all training/evaluation calls.

    Individual uses of the pipeline will apply slices of any such dataset. This
    class provides easy affordances for slicing the relevant cached data to
    match the provided dataset slice.
    """

    cache_key: str
    design_matrix: TRFDesignMatrix

    n_variable_predictors: int
    ts_feature_names: List[str]
    variable_feature_names: List[str]
    """
    List of variable features drawn from the dataset used to produce
    the design matrix. NB these are not necessarily one-to-one with
    `n_variable_predictors`, because there may be interaction
    features/derived features produced from the features in this list.
    """

    validation_mask: np.ndarray

    @property
    def n_variable_features(self): return len(self.variable_feature_names)

    def check_compatible(self, dataset: BerpDataset):
        assert dataset.name.startswith(self.cache_key)
        
        n_samples, n_features = self.design_matrix.shape[:2]

        for feature_list, label in [(dataset.ts_feature_names, "ts"), (dataset.variable_feature_names, "variable")]:
            unknown_features = set(getattr(self, f"{label}_feature_names")) - set(feature_list)
            if unknown_features:
                raise ValueError(f"Unknown {label} features, missing from dataset {dataset.name}: {unknown_features}")
        
        assert len(dataset) <= n_samples
        if dataset.global_slice_indices is not None:
            start, stop = dataset.global_slice_indices
            assert stop <= n_samples

        # assert self.validation_mask.shape[0] == dataset.n_samples

    def get_cache_for(self, dataset: BerpDataset) -> "ForwardPipelineCache":
        self.check_compatible(dataset)

        if dataset.global_slice_indices is None:
            return self

        start, end = dataset.global_slice_indices
        return replace(
            self,
            design_matrix=self.design_matrix[start:end],
            validation_mask=self.validation_mask[start:end]
        )


@typechecked
def make_dummy_design_matrix(dataset: BerpDataset, delayer: TRFDelayer,
                             ts_predictor_names: Optional[List[str]] = None,
                             n_variable_predictors=None) -> TRFDesignMatrix:
    """
    Prepare a design matrix with time series values inserted, leaving variable-onset values
    zero. Should then be combined with `scatter_variable`.
    """
    if n_variable_predictors is None:
        n_variable_predictors = dataset.n_variable_features

    ts_features, _ = dataset.get_features(ts_predictor_names)

    # Design matrix will be colocated on the same device as the dataset predictors.
    dummy_variable_predictors: TRFPredictors = \
        torch.zeros(dataset.n_samples, n_variable_predictors) \
            .to(ts_features)

    dummy_predictors = torch.concat([ts_features, dummy_variable_predictors], dim=1)
    design_matrix, _ = delayer.transform(dummy_predictors)
    return design_matrix


# Global cache container for all pipeline forwards.
# This allows us to reuse the cache between e.g. different instantiations of full
# pipelines with different hparams, but with the same predictors.
#
# But NB caches are not shared across different implementations (e.g. Berp vs vanilla)
# or same implementation with different predictors, because they may have different
# data representations.
CacheKey = Tuple[Type["GroupTRFForwardPipeline"], Tuple[str], Tuple[str]]
GLOBAL_CACHE: Dict[CacheKey, Dict[str, ForwardPipelineCache]] = {}


clones = [0]
def clone_count(x):
    clones[0] += 1
    # print("=====+++CLONE", clones[0])
    return x.clone()


class ScatterParamsMixin(BaseEstimator):

    scatter_key: Optional[str] = None
    scatter_targets: Optional[List[BaseEstimator]] = None

    def _prepare_params_scatter(self, scatter_key: str, scatter_targets: List[BaseEstimator]):
        self.scatter_key = scatter_key
        self.scatter_targets = scatter_targets

    def set_params(self, **params):
        if self.scatter_key is None or self.scatter_targets is None:
            return super().set_params(**params)
        if not params:
            return self

        to_pop = []
        for orig_key, value in params.items():
            key, delim, sub_key = orig_key.partition("__")
            if key == self.scatter_key:
                to_pop.append(orig_key)

                for target in self.scatter_targets:
                    target.set_params(**{sub_key: value})

        for orig_key in to_pop:
            params.pop(orig_key)

        return super().set_params(**params)


Encoder = TypeVar("Encoder", bound=TemporalReceptiveField)
Dataset = TypeVar("Dataset", bound=Union[BerpDataset, NestedBerpDataset])
class GroupTRFForwardPipeline(ScatterParamsMixin, BaseEstimator, Generic[Encoder]):

    """
    Abstract class. Jointly estimates many TRF encoders, optionally storing
    group-level parameters as well (e.g. latent onset model parameters).

    NOT THREAD SAFE. Updates in-place on `ForwardPipelineCache` instances.
    """

    def __init__(self, encoder: Encoder,
                 ts_feature_names: List[str],
                 variable_feature_names: List[str],
                 encoder_key_re: Union[str, re.Pattern[str]],
                 device: Optional[str] = None,
                 **kwargs):
        """
        Args:
            encoder: The encoder model to use for each subject.
                Will be cloned for each subject.
            ts_feature_names: Names of time series features to draw from
                dataset.
            variable_feature_names: Names of variable onset features to draw
                from dataset.
            encoder_key_re: Regular expression to match subject IDs in dataset.
                Datasets with matching IDs will be sent to the same encoder.
                The regular expression should have a single capture group.
            device: Torch device
        """
        self.encoder = encoder

        # Names of distinct time series / variable onset features in the
        # dataset. The encoder may derive more features from these dataset
        # features (e.g. interaction features).
        self.ts_feature_names = ts_feature_names
        self.variable_feature_names = variable_feature_names

        self.encoder_key_re = re.compile(encoder_key_re)

        self.device = device

        self.delayer = TRFDelayer(encoder.tmin, encoder.tmax, encoder.sfreq)
        self.encoders_: Dict[str, Encoder] = {}

        if kwargs:
            L.warning(f"Unused kwargs: {kwargs}")

    @property
    def n_ts_features(self): return len(self.ts_feature_names)
    @property
    def n_variable_features(self): return len(self.variable_feature_names)

    @property
    def encoder_predictor_names(self) -> Tuple[List[str], List[str]]:
        """
        Get the names of the predictors in the encoder design matrix, grouped by
        time-series- and variable-onset- predictors.
        """
        return (
            self.ts_feature_names,
            [f"var_{x}" for x in self.variable_feature_names]
        )

    #region caching and weight sharing logic

    @property
    def _encoder_cache_key(self) -> CacheKey:
        """
        Class instances with the same cache key can safely share `ForwardPipelineCache`s.
        """
        ts_names, variable_names = self.encoder_predictor_names
        return (self.__class__, tuple(ts_names), tuple(variable_names))

    @property
    def _cache(self) -> Dict[str, ForwardPipelineCache]:
        key = self._encoder_cache_key
        if key not in GLOBAL_CACHE:
            GLOBAL_CACHE[key] = {}
        return GLOBAL_CACHE[key]

    def _build_cache_for_dataset(self, dataset: BerpDataset) -> ForwardPipelineCache:
        ts_predictor_names, variable_predictor_names = self.encoder_predictor_names

        design_matrix = make_dummy_design_matrix(dataset, self.delayer,
                ts_predictor_names=ts_predictor_names,
                n_variable_predictors=len(variable_predictor_names))

        return ForwardPipelineCache(
            cache_key=dataset.name,
            design_matrix=design_matrix,
            n_variable_predictors=len(variable_predictor_names),
            ts_feature_names=self.ts_feature_names,
            variable_feature_names=self.variable_feature_names,
            validation_mask=_make_validation_mask(dataset, 0.1),  # TODO magic number
        )

    def _get_cache_for_dataset(self, dataset: BerpDataset) -> ForwardPipelineCache:
        key = dataset.base_name
        try:
            cache = self._cache[key]
        except KeyError as e:
            raise KeyError(f"Pipeline was never primed for dataset key {key}.") from e
        else:
            return cache.get_cache_for(dataset)

    def prime(self, dataset: Dataset):
        """
        Prepare caching pipelines for each subdataset of the given dataset.
        """
        datasets = dataset.datasets if isinstance(dataset, NestedBerpDataset) else [dataset]
        for d in datasets:
            if d.global_slice_indices is not None:
                raise ValueError(
                    f"Dataset {d} is already sliced. Pipeline priming should "
                     "be invoked before any data slicing.")

            L.info(f"Preparing cache for dataset {d.name}...")
            # Prepare dataset forward cache.
            self._cache[d.name] = self._build_cache_for_dataset(d)

            # Ensure we have an encoder ready for the relevant dataset group.
            self._get_or_create_encoder(d)

    def _get_encoder_key(self, dataset: BerpDataset) -> str:
        """
        Compute a key by which this dataset should be mapped with others to
        a single encoder. (most intuitively this should be e.g. a subject ID.)
        """
        match = self.encoder_key_re.match(dataset.name)
        if match is None:
            raise RuntimeError(dataset.name)
        return match.group(1)

    def _get_encoder(self, dataset: BerpDataset) -> Encoder:
        """
        Get the relevant encoder for making predictions on this dataset.
        """
        key = self._get_encoder_key(dataset)
        try:
            return self.encoders_[key]
        except KeyError as e:
            raise KeyError(f"encoder not found for key {key}. Did you prime this model first?") from e

    def _set_encoder(self, key: str, encoder: Encoder):
        self.encoders_[key] = encoder
        self._prepare_params_scatter("encoder", [self.encoder] + list(self.encoders_.values()))

    def _get_or_create_encoder(self, dataset: BerpDataset) -> Encoder:
        try:
            return self._get_encoder(dataset)
        except KeyError:
            key = self._get_encoder_key(dataset)
            L.debug(f"Creating encoder for key {key}")
            enc: Encoder = cast(Encoder, clone(self.encoder))
            enc.set_name(key)
            enc.output_names = dataset.sensor_names
            self._set_encoder(key, enc)

            return self.encoders_[key]

    def _get_encoders(self, dataset: NestedBerpDataset) -> List[Tuple[BerpDataset, Encoder]]:
        return [(d, self._get_encoder(d)) for d in dataset.datasets]

    def _get_or_create_encoders(self, dataset: NestedBerpDataset) -> List[Tuple[BerpDataset, Encoder]]:
        return [(d, self._get_or_create_encoder(d)) for d in dataset.datasets]

    def reset_early_stopping(self):
        """
        Reset early stopping tracker for all encoders.
        """
        for encoder in self.encoders_.values():
            encoder.optim.reset_early_stopping()

    #endregion

    #region domain logic

    

    def pre_transform(self, dataset: BerpDataset) -> Tuple[TRFDesignMatrix, np.ndarray]:
        """
        Run a forward pass of the pre-transformation step, averaging out any
        other model parameters.
        """
        raise NotImplementedError()

    def pre_transform_expanded(self, dataset: BerpDataset) -> Iterator[Tuple[TRFDesignMatrix, np.ndarray]]:
        """
        Run a forward pass of the pre-transformation step, generating a design
        matrix for each latent parameter setting.
        
        NB, the same memory yielded in each iterator step will be reused at the
        next iterator step. If you want to store the yielded memory, you must
        clone it in order to keep it safe.
        """
        # By default, if there are no latent parameters, just yield the only
        # design matrix we have.
        yield self.pre_transform(dataset)
    
    #endregion

    #region scikit API

    def _select_features(self, dataset: Dataset) -> Dataset:
        """
        Modify the dataset in-place to have the relevant set of features
        for this model.
        """
        dataset.select_features(
            ts=self.ts_feature_names, variable=self.variable_feature_names)
        return dataset

    def _get_features(self, dataset: BerpDataset) -> Tuple[TensorType[float], TensorType[float]]:
        """
        Get the relevant features from the dataset for this model.
        """
        return dataset.get_features(
            ts=self.ts_feature_names, variable=self.variable_feature_names)

    def _prepare_trf_Xy(self, datasets: List[BerpDataset]) -> Tuple[TRFDesignMatrix, TRFResponse]:
        design_matrices, Ys = [], []
        for d in datasets:
            assert_concatenatable(d, datasets[0])

            design_matrix, _ = self.pre_transform(d)
            design_matrices.append(design_matrix)
            Ys.append(d.Y)

        design_matrix = cat(design_matrices, dim=0)
        Y = cat(Ys, dim=0)

        return design_matrix, Y

    def _fit(self, encoder: Encoder, datasets: List[BerpDataset]):
        design_matrix, Y = self._prepare_trf_Xy(datasets)
        
        with torch.no_grad():
            encoder.fit(design_matrix, Y)

    def fit(self, dataset: NestedBerpDataset, y=None) -> "GroupTRFForwardPipeline":
        # Group datasets by encoder and fit jointly.
        grouped_datasets = defaultdict(list)
        for ds_i in dataset.datasets:
            key = self._get_encoder_key(ds_i)
            grouped_datasets[key].append(ds_i)

        for key, datasets in tqdm(grouped_datasets.items(), desc="Exact fit"):
            enc = self._get_or_create_encoder(datasets[0])
            self._fit(enc, datasets)

        return self

    def _partial_fit(self, encoder: Encoder, dataset: BerpDataset):
        design_matrix, validation_mask = self.pre_transform(dataset)
        encoder.partial_fit(design_matrix, dataset.Y, validation_mask=validation_mask,
                            dataset_tag=dataset.name)

    def partial_fit(self, dataset: NestedBerpDataset, y=None) -> "GroupTRFForwardPipeline":
        n_early_stops = 0

        encs = self._get_or_create_encoders(dataset)
        for d, enc in encs:
            try:
                self._partial_fit(enc, d)
            except EarlyStopException:
                n_early_stops += 1

        # Only raise EarlyStop if all encoder children have reached an early stop.
        if n_early_stops == len(encs):
            raise EarlyStopException()

        return self

    @typechecked
    def predict(self, dataset: BerpDataset) -> TRFResponse:
        enc = self._get_or_create_encoder(dataset)
        design_matrix, _ = self.pre_transform(dataset)
        return enc.predict(design_matrix)

    @singledispatchmethod
    def score(self, dataset, y=None) -> float:
        raise NotImplementedError

    @score.register
    @typechecked
    def _(self, dataset: BerpDataset, y=None) -> float:
        enc = self._get_or_create_encoder(dataset)
        design_matrix, _ = self.pre_transform(dataset)
        return enc.score(design_matrix, dataset.Y)

    @score.register
    @typechecked
    def _(self, dataset: NestedBerpDataset, y=None) -> float:
        scores = [self.score(d) for d in dataset.datasets]
        return np.mean(scores)

    @singledispatchmethod
    def score_multidimensional(self, dataset, y=None) -> np.ndarray:
        raise NotImplementedError

    @score_multidimensional.register
    @typechecked
    def _(self, dataset: BerpDataset, y=None) -> np.ndarray:
        enc = self._get_or_create_encoder(dataset)
        design_matrix, _ = self.pre_transform(dataset)
        return enc.score_multidimensional(design_matrix, dataset.Y)

    @score_multidimensional.register
    @typechecked
    def _(self, dataset: NestedBerpDataset, y=None) -> np.ndarray:
        scores = [self.score_multidimensional(d) for d in dataset.datasets]
        return np.array(scores)

    @singledispatchmethod
    def log_likelihood(self, dataset):
        raise NotImplementedError

    @log_likelihood.register
    @typechecked
    def _(self, dataset: BerpDataset) -> TensorType[B, torch.float]:
        enc = self._get_or_create_encoder(dataset)
        design_matrix, _ = self.pre_transform(dataset)
        return enc.log_likelihood(design_matrix, dataset.Y).sum()

    @log_likelihood.register
    @typechecked
    def _(self, dataset: NestedBerpDataset) -> List[TensorType[B, torch.float]]:
        return [self.log_likelihood(d) for d in dataset.datasets]

    @singledispatchmethod
    def log_likelihood_expanded(self, dataset):
        raise NotImplementedError

    @log_likelihood_expanded.register
    @typechecked
    def _(self, dataset: BerpDataset) -> TensorType["param_grid", torch.float]:
        enc = self._get_or_create_encoder(dataset)
        ret = []
        for design_matrix, _ in self.pre_transform_expanded(dataset):
            ret.append(enc.log_likelihood(design_matrix, dataset.Y).sum())
        return torch.stack(ret)

    @log_likelihood_expanded.register
    @typechecked
    def _(self, dataset: NestedBerpDataset
          ) -> Union[List[TensorType["param_grid", torch.float]],
                     List[torch.Tensor]]:
        return [self.log_likelihood_expanded(d)
                for d in dataset.datasets]

    #endregion


class GroupBerpTRFForwardPipeline(GroupTRFForwardPipeline):

    def __init__(self, encoder: Encoder,
                 ts_feature_names: List[str],
                 variable_feature_names: List[str],
                 encoder_key_re: Union[str, re.Pattern[str]],

                 params: List[ModelParameters],
                 param_weights: Optional[Responsibilities] = None,

                 scatter_point: float = 0.0,
                 prior_scatter_index: int = 0,
                 prior_scatter_point: float = 0.0,

                 variable_trf_zero_left: int = 0,
                 variable_trf_zero_right: int = 0,
                 **kwargs):
        """
        Args:
            encoder:
            params:
            param_weights:
            scatter_point: Describes expected recognition time logic when
                words are recognized given some perceptual input.
                See `reindexing_regression.recognition_points_to_times`.
            prior_scatter_point: Describes expected recognition time logic when words are
                recognized prior to perceptual input.
                See `reindexing_regression.recognition_points_to_times`
            variable_trf_zero_left: Constrain this many samples starting from the left edge
                of the learned encoder time series to have zero values, for those encoder
                features which are variable-onset.
            variable_trf_zero_right: Constrain this many samples starting from the right edge
                of the learned encoder time series to have zero values, for those encoder
                features which are variable-onset.
        """
        super().__init__(encoder, ts_feature_names, variable_feature_names, encoder_key_re,
                         **kwargs)

        self.params = params
        self.param_weights = param_weights if param_weights is not None else \
            torch.ones(len(self.params), dtype=torch.float) / len(self.params)
        
        self.scatter_point = scatter_point
        self.prior_scatter_index = prior_scatter_index
        self.prior_scatter_point = prior_scatter_point

        # If possible, validate that these are compatible with the encoder's
        # temporal window.
        if hasattr(encoder, "tmin") and hasattr(encoder, "tmax") and hasattr(encoder, "sfreq"):
            encoder_num_samples = int((encoder.tmax - encoder.tmin) * encoder.sfreq)
            if encoder_num_samples < variable_trf_zero_left + variable_trf_zero_right:
                raise ValueError("Encoder's temporal window is too small for the "
                                 "requested zero padding.")

        self.variable_trf_zero_left = variable_trf_zero_left
        self.variable_trf_zero_right = variable_trf_zero_right

    @typechecked
    def get_recognition_points(self, dataset: BerpDataset,
                               params: ModelParameters,
                               ) -> TensorType[torch.long]:
        # TODO cache rec point computation?
        # profile and find out if it's worth it
        p_candidates_posterior = predictive_model(
            dataset.p_candidates, dataset.candidate_phonemes,
            params.confusion, params.lambda_
        )
        recognition_points = recognition_point_model(
            p_candidates_posterior, dataset.word_lengths, params.threshold
        )
        return recognition_points

    @typechecked
    def get_recognition_times(self, dataset: BerpDataset,
                              params: ModelParameters,
                              ) -> Tuple[TensorType[torch.long], TensorType[torch.float]]:
        recognition_points = self.get_recognition_points(dataset, params)
        recognition_times = recognition_points_to_times(
            recognition_points,
            dataset.phoneme_onsets_global,
            dataset.phoneme_offsets_global,
            dataset.word_lengths,
            scatter_point=self.scatter_point,
            prior_scatter_index=self.prior_scatter_index,
            prior_scatter_point=self.prior_scatter_point,
        )
        return recognition_points, recognition_times

    def _pre_transform_single(self, dataset: BerpDataset,
                              params: ModelParameters,
                              out: TRFDesignMatrix,
                              out_weight: Union[float, TensorType[float]] = 1.,
                              ) -> None:
        """
        Run word recognition logic for the given dataset and parameters,
        producing a regression design matrix. Operates in-place.
        """
        recognition_points, recognition_times = self.get_recognition_times(dataset, params)

        # TODO refactor: lots of overlapping logic with other pre_transform_single impls
        # Generate lag mask given zero-ing rules.
        lag_mask = torch.ones(out.shape[2], dtype=torch.bool)
        lag_mask[:self.variable_trf_zero_left] = False
        if self.variable_trf_zero_right != 0:
            lag_mask[-self.variable_trf_zero_right:] = False

        recognition_times_samp = time_to_sample(recognition_times, self.encoder.sfreq)
        _, X_variable = self._get_features(dataset)

        feature_start_idx = self.n_ts_features
        scatter_add(torch.narrow(out, 1, feature_start_idx, X_variable.shape[1]),
                    target_samples=recognition_times_samp,
                    target_values=X_variable,
                    lag_mask=lag_mask)
    
    def pre_transform(self, dataset: BerpDataset) -> Tuple[TRFDesignMatrix, MaskArray]:
        primed = self._get_cache_for_dataset(dataset)
        # acc = primed.design_matrix.clone()
        acc = clone_count(primed.design_matrix)
        
        # Ensure variable-onset features are zeroed out.
        feature_start_idx = self.n_ts_features
        acc[:, feature_start_idx:, :] = 0.

        for params, weight in zip(self.params, self.param_weights):
            self._pre_transform_single(dataset, params, out=acc, out_weight=weight)
        
        return acc, primed.validation_mask

    def pre_transform_expanded(self, dataset: BerpDataset) -> Iterator[Tuple[TRFDesignMatrix, MaskArray]]:
        primed = self._get_cache_for_dataset(dataset)
        feature_start_idx = self.n_ts_features

        for params in self.params:
            # NB _pre_transform_single operates in place, so we'll reset variable-onset
            # features each time.
            # TODO does this break things downstream?
            primed.design_matrix[:, feature_start_idx:, :] = 0.

            yield (self._pre_transform_single(dataset, params,
                                          #   out=primed.design_matrix.clone(),
                                              out=primed.design_matrix,
                                              out_weight=1.),
                   primed.validation_mask)


class GroupBerpFixedTRFForwardPipeline(GroupBerpTRFForwardPipeline):

    """
    Group pipeline for variable-onset TRF with just one parameter option.
    Provides easy scikit-learn-friendly parameter accessors.
    """
    # TODO figure out better structure, this is weird. maybe just have an independent model, rather
    # than one that simulates a list of params. it makes for awkward accessor/parameter update logic

    _param_keys = ["threshold", "confusion", "lambda_"]

    def __init__(self, encoder: Encoder,
                 ts_feature_names: List[str],
                 variable_feature_names: List[str],
                 encoder_key_re: Union[str, re.Pattern[str]],

                 threshold: torch.Tensor,
                 confusion: torch.Tensor,
                 lambda_: torch.Tensor,

                 scatter_point: float = 0,
                 prior_scatter_index: int = 0,
                 prior_scatter_point: float = 0.0,

                 variable_trf_zero_left: int = 0,
                 variable_trf_zero_right: int = 0,
                 **kwargs):
        self.threshold = threshold
        self.confusion = confusion
        self.lambda_ = lambda_

        super().__init__(encoder,
            ts_feature_names, variable_feature_names,
            encoder_key_re,
            self.params,
            scatter_point=scatter_point,
            prior_scatter_index=prior_scatter_index,
            prior_scatter_point=prior_scatter_point,
            variable_trf_zero_left=variable_trf_zero_left,
            variable_trf_zero_right=variable_trf_zero_right,
            **kwargs)

    def _model_params_getter(self) -> List[ModelParameters]:
        return [PartiallyObservedModelParameters(
            threshold=torch.as_tensor(self.threshold),
            confusion=torch.as_tensor(self.confusion),
            lambda_=torch.as_tensor(self.lambda_),
        )]
    def _model_params_setter(self, params: List[ModelParameters]):
        # HACK: No thx
        # We are here because the superclass tries to set `params` in its constructor.
        # but we already did that. So ignore it, with a check that the params will
        # match what we already have.
        if not params == self.params:
            raise RuntimeError("Hacky design assumption violated. Who is setting params?")
        return
    params = property(_model_params_getter, _model_params_setter)


class GroupBerpCannonTRFForwardPipeline(GroupBerpFixedTRFForwardPipeline):

    """
    Berp TRF model which learns independent parametric responses for
    words based on binned recognition time, rather than realigning a
    single characteristic response in a time series.
    """

    def __init__(self, encoder: Encoder,
                 ts_feature_names: List[str],
                 variable_feature_names: List[str],
                 encoder_key_re: Union[str, re.Pattern[str]],

                 threshold: torch.Tensor,
                 confusion: torch.Tensor,
                 lambda_: torch.Tensor,

                 n_quantiles: int = 3,

                 scatter_point: float = 0,
                 prior_scatter_index: int = 0,
                 prior_scatter_point: float = 0.0,

                 variable_trf_zero_left: int = 0,
                 variable_trf_zero_right: int = 0,
                 **kwargs):
        super().__init__(encoder, ts_feature_names, variable_feature_names,
                         encoder_key_re,
                         threshold, confusion, lambda_,
                         scatter_point=scatter_point,
                         prior_scatter_index=prior_scatter_index,
                         prior_scatter_point=prior_scatter_point,

                         variable_trf_zero_left=variable_trf_zero_left,
                         variable_trf_zero_right=variable_trf_zero_right,
                         **kwargs)

        self.n_quantiles = n_quantiles

    @property
    def encoder_predictor_names(self) -> Tuple[List[str], List[str]]:
        ts_predictor_names, variable_predictor_names = super().encoder_predictor_names
        variable_predictor_names = [
            f"{name}_{i}"
            for i in range(self.n_quantiles + 1)
            for name in variable_predictor_names
        ]
        return ts_predictor_names, variable_predictor_names

    def _check_shapes(self, dataset: Dataset):
        if not hasattr(self.encoder, "n_features_"):
            return

        datasets = [dataset] if isinstance(dataset, BerpDataset) else dataset.datasets
        for ds in datasets:
            assert self.encoder.n_features_ == dataset.n_ts_features + (self.n_quantiles + 1) * dataset.n_variable_features, \
                "Encoder should have N + 1 sets of variable-onset features, for N quantiles"

    def _fit_recognition_quantiles(self, dataset: Dataset) -> None:
        """
        Estimate recognition time quantiles for all words in a training set.
        Returns `n_quantiles + 1` list of edges for quantile bins.
        """
        if len(self.params) != 1:
            raise NotImplementedError()

        datasets = dataset.datasets if isinstance(dataset, NestedBerpDataset) else [dataset]
        
        # Compute local recognition times (relative to word onset)
        all_recognition_times = torch.cat([
            self.get_recognition_times(ds, self.params[0])[1] - ds.word_onsets
            for ds in datasets
        ])

        L.info(f"Computing recognition quantiles from dataset of {len(all_recognition_times)} words.")
        self.recognition_quantile_edges_ = torch.quantile(
            all_recognition_times, torch.linspace(0, 1, self.n_quantiles + 1).to(all_recognition_times.device))
        assert len(self.recognition_quantile_edges_) == self.n_quantiles + 1

        # These quantile assignments need to generalize to new datasets which might have more negative
        # minimum recognition times or more maximum recognition times. So account for this by setting
        # left and right quantile boundaries to -inf and inf, respectively.
        self.recognition_quantile_edges_[0] = -torch.inf
        self.recognition_quantile_edges_[-1] = torch.inf

        L.info("Recognition quantile edges: %s", self.recognition_quantile_edges_.cpu().numpy())

    @typechecked
    def _get_recognition_quantiles(self,
                                   dataset: BerpDataset,
                                   params: ModelParameters
                                   ) -> TensorType[B, torch.long]:
        """
        Compute assignments mapping each word to a recognition-time quantile.

        Returns a tensor mapping each word to a quantile index `[0, n_quantiles)`.
        """

        recognition_points, recognition_times = self.get_recognition_times(dataset, params)
        local_recognition_times = recognition_times - dataset.word_onsets

        if not hasattr(self, "recognition_quantile_edges_") or self.recognition_quantile_edges_ is None:
            raise RuntimeError("Recognition quantile bins have not been estimated. Stop.")            
        
        # NB we have N+1 buckets and at least one value will be assigned to the extreme left
        # bucket (the minimum value). Clamp output such that we have N buckets instead.
        # We could equivalently bucketize with `right=True` and then clamp on `[0, n_quantiles)`.
        recognition_bins = torch.clamp(
            torch.bucketize(local_recognition_times, self.recognition_quantile_edges_),
            1, self.n_quantiles)
        assert (recognition_bins > 0).all()
        recognition_bins -= 1

        L.debug("Recognition bin distribution: %s",
            recognition_bins.bincount(minlength=self.n_quantiles).cpu().numpy())

        return recognition_bins

    def _pre_transform_single(self, dataset: BerpDataset,
                              params: ModelParameters,
                              out: TRFDesignMatrix,
                              out_weight: Union[float, TensorType[torch.float]] = 1.,
                              ) -> None:
        """
        Run word recognition logic for the given dataset and parameters,
        producing a regression design matrix. Operates in-place.
        """
        # Get recognition quantile assignments for each word in the dataset.
        recognition_quantiles = self._get_recognition_quantiles(dataset, params)

        # Generate lag mask given zero-ing rules.
        lag_mask = torch.ones(out.shape[2], dtype=torch.bool)
        lag_mask[:self.variable_trf_zero_left] = False
        if self.variable_trf_zero_right != 0:
            lag_mask[-self.variable_trf_zero_right:] = False

        word_onset_samples = time_to_sample(dataset.word_onsets, dataset.sample_rate)
        _, X_variable = self._get_features(dataset)

        assert X_variable.shape[1] == len(self.variable_feature_names)
        n_variable_features = X_variable.shape[1]

        # All bins use the first surprisal feature.
        target_0 = torch.narrow(out, 1, self.n_ts_features, n_variable_features)
        scatter_add(target_0,
                    target_samples=word_onset_samples,
                    target_values=X_variable,
                    lag_mask=lag_mask)

        for rec_bin_i in range(self.n_quantiles):
            mask = recognition_quantiles == rec_bin_i

            feature_start_idx = self.n_ts_features + (rec_bin_i + 1) * n_variable_features
            samples, values = word_onset_samples[mask], X_variable[mask]

            # Pass just the slice of `out` that should be updated. NB `torch.narrow`
            # never returns a copy.
            target_i = torch.narrow(out, 1, feature_start_idx, n_variable_features)

            scatter_add(target_i,
                        target_samples=samples,
                        target_values=values * out_weight,
                        lag_mask=lag_mask)

    def pre_transform(self, dataset: BerpDataset) -> Tuple[TRFDesignMatrix, MaskArray]:
        self._check_shapes(dataset)
        return super().pre_transform(dataset)
    
    def pre_transform_expanded(self, dataset: BerpDataset) -> Iterator[Tuple[TRFDesignMatrix, np.ndarray]]:
        self._check_shapes(dataset)
        return super().pre_transform_expanded(dataset)

    def fit(self, dataset: NestedBerpDataset, y=None) -> "GroupBerpCannonTRFForwardPipeline":
        # First re-estimate recognition quantiles on the union of all datasets.
        self._fit_recognition_quantiles(dataset)
        super().fit(dataset, y=y)
        return self

    def partial_fit(self, *args, **kwargs):
        raise NotImplementedError()


class GroupVanillaTRFForwardPipeline(GroupTRFForwardPipeline):

    def _scatter(self, dataset: BerpDataset, design_matrix: TRFDesignMatrix):
        """
        Scatter-add variable-onset data onto word onset points in time series
        represented in `design_matrix`. Operates in-place.
        """
        if dataset.n_variable_features == 0:
            return

        recognition_onsets = dataset.word_onsets
        recognition_onsets_samp = time_to_sample(recognition_onsets, self.encoder.sfreq)
        _, X_variable = self._get_features(dataset)

        feature_start_idx = self.n_ts_features
        scatter_add(design_matrix[:, feature_start_idx:, :],
                    recognition_onsets_samp,
                    X_variable,
                    add=False)

    def pre_transform(self, dataset: BerpDataset) -> Tuple[TRFDesignMatrix, np.ndarray]:
        primed = self._get_cache_for_dataset(dataset)

        # Ensure variable-onset features are zeroed out.
        feature_start_idx = self.n_ts_features
        primed.design_matrix[:, feature_start_idx:, :] = 0.
        
        # Fine to not clone cached values -- they are constant.
        self._scatter(dataset, primed.design_matrix)

        return primed.design_matrix, primed.validation_mask


class GroupVanillaCannonTRFForwardPipeline(GroupBerpCannonTRFForwardPipeline):
    """
    Degenerate/vanilla form of the cannon pipeline, in which words are split
    based on surprisal (prior probability) rather than on inferred
    recognition time.
    """

    def _fit_recognition_quantiles(self, dataset: Dataset) -> None:
        """
        Estimate recognition time quantiles for all words in a training set.
        Returns `n_quantiles + 1` list of edges for quantile bins.
        """

        datasets = dataset.datasets if isinstance(dataset, NestedBerpDataset) else [dataset]

        all_surprisals = torch.cat([dataset.p_candidates[:, 0] for dataset in datasets])

        L.info(f"Computing surprisal quantiles from dataset of {len(all_surprisals)} words.")
        self.recognition_quantile_edges_ = torch.quantile(
            all_surprisals, torch.linspace(0, 1, self.n_quantiles + 1).to(all_surprisals.device))
        assert len(self.recognition_quantile_edges_) == self.n_quantiles + 1

        # These quantile assignments need to generalize to new datasets which might have more negative
        # minimum recognition times or more maximum recognition times. So account for this by setting
        # left and right quantile boundaries to -inf and inf, respectively.
        self.recognition_quantile_edges_[0] = -torch.inf
        self.recognition_quantile_edges_[-1] = torch.inf

        L.info("Recognition quantile edges: %s", self.recognition_quantile_edges_.cpu().numpy())

    @typechecked
    def _get_recognition_quantiles(self,
                                   dataset: BerpDataset,
                                   params: ModelParameters
                                   ) -> TensorType[B, torch.long]:
        """
        Compute assignments mapping each word to a recognition-time quantile.

        Returns a tensor mapping each word to a quantile index `[0, n_quantiles)`.
        """

        if not hasattr(self, "recognition_quantile_edges_") or self.recognition_quantile_edges_ is None:
            raise RuntimeError("Recognition quantile bins have not been estimated. Stop.")            
        
        # NB we have N+1 buckets and at least one value will be assigned to the extreme left
        # bucket (the minimum value). Clamp output such that we have N buckets instead.
        # We could equivalently bucketize with `right=True` and then clamp on `[0, n_quantiles)`.
        recognition_bins = torch.clamp(
            torch.bucketize(dataset.p_candidates[:, 0], self.recognition_quantile_edges_),
            1, self.n_quantiles)
        assert (recognition_bins > 0).all()
        recognition_bins -= 1

        L.debug("Recognition bin distribution: %s",
            recognition_bins.bincount(minlength=self.n_quantiles).cpu().numpy())

        return recognition_bins


class BerpTRFEMEstimator(BaseEstimator):
    """
    Jointly estimate parameters of a Berp model using expectation maximization.
    """

    @typechecked
    def __init__(self, pipeline: GroupBerpTRFForwardPipeline,
                 n_iter=1, warm_start=True,
                 early_stopping: Optional[int] = 1,
                 **kwargs):
        self.pipeline = pipeline
        self.param_resp_ = self.pipeline.param_weights

        self.n_iter = n_iter
        self.warm_start = warm_start
        self.early_stopping = early_stopping

        self._best_val_score_ = -np.inf
        self._no_improvement_count_ = 0
        
        if kwargs:
            L.warning(f"Unused kwargs: {kwargs}")

    def set_param_resp(self, resp: Responsibilities):
        self._param_resp_ = resp
        self.pipeline.param_weights = resp
    def get_param_resp(self):
        return self._param_resp_
    param_resp_: Responsibilities = property(get_param_resp, set_param_resp)

    @typechecked
    def _e_step(self, dataset: NestedBerpDataset) -> Responsibilities:
        """
        Compute responsibility values for each parameter in the grid for the
        given dataset.
        """
        # n_pipelines * n_param_options
        log_liks = self.pipeline.log_likelihood_expanded(dataset)
        # Compute joint probability of params, treating subject pipelines as independent
        resp = torch.stack(log_liks).sum(0)

        # Convert to probabilities
        resp -= resp.max()
        resp = resp.exp()
        resp = resp / resp.sum()
        return resp

    def _m_step(self, dataset: NestedBerpDataset):
        """
        Re-estimate TRF model conditioned on the current parameter weights.

        May raise `EarlyStopException`.
        """
        self.pipeline.partial_fit(dataset)

    def _reset_pipeline_early_stopping(self):
        if hasattr(self.pipeline, "reset_early_stopping"):
            self.pipeline.reset_early_stopping()

    def prime(self, dataset):
        return self.pipeline.prime(dataset)

    def _tb_update(self, X: NestedBerpDataset):
        """
        Update Tensorboard with current state.
        """
        tb = Tensorboard.instance()

        # TODO this is all coupled with the current inference setup.
        # It pulls out param properties from the pipeline, and constructs
        # its own param representations. Not exactly sustainable.

        # Compute expected threshold value.
        threshold = (self.param_resp_ * torch.stack([p.threshold for p in self.pipeline.params])).sum()
        tb.add_scalar("threshold", threshold.item())

        # Compute recognition points.
        # HACK Construct new parameter representation with the above threshold.
        params = replace(self.pipeline.params[0], threshold=threshold)
        recog_points = torch.cat([
            self.pipeline.get_recognition_points(dataset, params)
            for dataset in X.datasets
        ])
        tb.add_histogram("recognition_points", recog_points)

    def partial_fit(self, X: NestedBerpDataset, y=None,
                    X_val: Optional[NestedBerpDataset] = None,
                    use_tqdm=False,
                    ) -> "BerpTRFEMEstimator":
        if X_val is None and self.early_stopping is not None:
            L.warning("Early stopping requested, but no validation set provided.")

        tb = Tensorboard.instance()

        m_step_did_early_stop = False
        with trange(self.n_iter, desc="EM", disable=not use_tqdm) as pbar:
            postfix = {"train_score": self.score(X)}
            tb.add_scalar("em/train_score", postfix["train_score"])
            if X_val is not None:
                val_score = self._check_validation_score(X_val)
                postfix["val_score"] = val_score
                tb.add_scalar("em/val_score", val_score)
            pbar.set_postfix(**postfix)

            for _ in pbar:
                tb.global_step += 1

                self.param_resp_ = self._e_step(X)
                L.info("E-step finished")

                # HACK: print inferred threshold value
                print(self.param_resp_.numpy().round(3))
                self._tb_update(X)

                # Re-estimate encoder parameters
                try:
                    self._m_step(X)
                except EarlyStopException:
                    m_step_did_early_stop = True
                    L.info("M-step early stopped. Will run at least one more E-step")
                    self._reset_pipeline_early_stopping()
                else:
                    m_step_did_early_stop = False
                L.info("M-step finished")

                # Calculate scores
                postfix = {"train_score": self.score(X)}
                tb.add_scalar("em/train_score", postfix["train_score"])
                if X_val is not None:
                    try:
                        val_score = self._check_validation_score(X_val)
                    except EarlyStopException:
                        if m_step_did_early_stop:
                            L.info("EM early stopping")
                            break
                        else:
                            L.info("Val score halted, but M-step did not early stop. Will run at least one more M-step")

                    postfix["val_score"] = val_score
                    tb.add_scalar("em/val_score", val_score)
                pbar.set_postfix(**postfix)

        return self

    def _check_validation_score(self, X_val: NestedBerpDataset) -> float:
        """
        Evaluate validation score and update early stopping tracker.
        May raise EarlyStopException.
        """

        val_score = self.score(X_val)
        L.info("Val score: %f", val_score)
        if val_score > self._best_val_score_:
            self._best_val_score_ = val_score
            self._no_improvement_count_ = 0
        elif self.early_stopping is not None and self._no_improvement_count_ >= self.early_stopping:
            raise EarlyStopException()
        else:
            self._no_improvement_count_ += 1

        return val_score

    def fit(self, *args, **kwargs) -> "BerpTRFEMEstimator":
        return self.partial_fit(*args, **kwargs)

    def predict(self, dataset: NestedBerpDataset) -> TRFResponse:
        return self.pipeline.predict(dataset)

    def score(self, dataset: NestedBerpDataset, y=None):
        return self.pipeline.score(dataset)

    def log_likelihood(self, dataset: NestedBerpDataset, y=None) -> torch.Tensor:
        return self.pipeline.log_likelihood(dataset)


def load_confusion_parameters(
    confusion_path: str, dataset_phonemes: List[str],
    smooth=False) -> torch.Tensor:
    confusion = np.load(confusion_path)
    
    # Set of phonemes should be superset of dataset phonemes
    if not dataset_phonemes == confusion["phonemes"].tolist():
        diff = set(dataset_phonemes) - set(confusion["phonemes"])
        if diff:
            raise ValueError(f"Some phonemes provided to the pipeline are not represented "
                             f"in the confusion parameters: {', '.join(diff)}")
        else:
            raise ValueError("Dataset phonemes and confusion matrix phonemes are out of order.")

    confusion_matrix = confusion["confusion"]
    
    # Check shapes.
    assert confusion_matrix.shape == (len(dataset_phonemes), len(dataset_phonemes))

    # Smooth.
    if smooth:
        confusion_matrix += 1.

    # Normalize. Each column should be a probability distribution.
    confusion_matrix /= confusion_matrix.sum(axis=0, keepdims=True) + 1e-5

    return torch.tensor(confusion_matrix)


def prepare_or_create_confusion(confusion_path: Optional[str],
                                phonemes: List[str]) -> torch.Tensor:
    if confusion_path == "flat":
        confusion = torch.full((len(phonemes), len(phonemes)), 1. / len(phonemes))
    elif confusion_path is not None:
        confusion = load_confusion_parameters(
            to_absolute_path(confusion_path), phonemes)
    else:
        confusion = torch.eye(len(phonemes)) + 0.01
        confusion /= confusion.sum(dim=0, keepdim=True)

    return confusion


def update_with_pretrained(pipeline: GroupBerpTRFForwardPipeline,
                           pretrained: GroupTRFForwardPipeline):
    if not isinstance(pretrained, GroupTRFForwardPipeline):
        raise ValueError(f"Unknown pretrained pipeline type {type(pretrained)}")
    if isinstance(pretrained, GroupBerpTRFForwardPipeline):
        L.warning("Initializing Berp pipeline with another pretrained "
                  "Berp pipeline. Not sure what to do here. TODO.")

    # Take encoders.
    new_encoder_keys = sorted(pretrained.encoders_.keys())
    L.info("Adding pretrained encoders with keys: %r", new_encoder_keys)
    if set(new_encoder_keys) & set(pipeline.encoders_.keys()):
        overlap = sorted(set(new_encoder_keys) & set(pipeline.encoders_.keys()))
        raise ValueError(
            "Pipeline already has encoders with keys matching those in this "
           f"pretrained pipeline. What to do?\n{', '.join(overlap)}")

    # Keep the following encoder parameters from the current pipeline.
    # TODO should probably be the reverse -- configurably override just
    # a set of the current pipeline parameters from the pretrained pipeline.
    keep_params = ["optim__n_batches", "optim__early_stopping"]
    this_params = pipeline.encoder.get_params()

    for name, enc in pretrained.encoders_.items():
        enc = deepcopy(enc)
        enc.set_params(**{k: this_params[k] for k in keep_params})
        pipeline._set_encoder(name, enc)

    return pipeline


def update_with_pretrained_paths(pipeline: GroupBerpTRFForwardPipeline,
                                 paths: List[str]):
    for path in paths:
        with open(to_absolute_path(path), "rb") as f:
            pipeline = update_with_pretrained(pipeline, pickle.load(f))

    return pipeline


def make_pipeline(
    trf: TemporalReceptiveField,
    params: List[ModelParameters],
    pretrained_pipeline_paths: Optional[List[str]] = None,
    *args, **kwargs) -> GroupBerpTRFForwardPipeline:

    pipeline = GroupBerpTRFForwardPipeline(trf, params=params, **kwargs)

    if pretrained_pipeline_paths is not None:
        pipeline = update_with_pretrained_paths(pipeline, pretrained_pipeline_paths)

    return pipeline


def BerpTRFFixed(trf: TemporalReceptiveField,
                 features: FeatureConfig,
                 threshold: torch.Tensor,
                 phonemes: List[str],
                 confusion_path: Optional[str] = None,
                 pretrained_pipeline_paths: Optional[List[str]] = None,
                 **kwargs) -> GroupBerpFixedTRFForwardPipeline:
    pipeline = GroupBerpFixedTRFForwardPipeline(
        trf,
        ts_feature_names=list(features.ts_feature_names),
        variable_feature_names=list(features.variable_feature_names),
        threshold=torch.as_tensor(threshold),
        confusion=prepare_or_create_confusion(confusion_path, phonemes),
        lambda_=torch.tensor(1.),
        **kwargs,
    )

    if pretrained_pipeline_paths is not None:
        pipeline = update_with_pretrained_paths(pipeline, pretrained_pipeline_paths)

    return pipeline


def BerpTRFCannon(trf: TemporalReceptiveField,
                  features: FeatureConfig,
                  threshold: torch.Tensor,
                  lambda_: torch.Tensor,
                  n_quantiles: int,
                  phonemes: List[str],
                  confusion_path: Optional[str] = None,
                  pretrained_pipeline_paths: Optional[List[str]] = None,
                  **kwargs) -> GroupBerpCannonTRFForwardPipeline:
    pipeline = GroupBerpCannonTRFForwardPipeline(
        trf,
        ts_feature_names=list(features.ts_feature_names),
        variable_feature_names=list(features.variable_feature_names),
        threshold=torch.as_tensor(threshold),
        confusion=prepare_or_create_confusion(confusion_path, phonemes),
        lambda_=lambda_,
        n_quantiles=n_quantiles,
        **kwargs,
    )

    if pretrained_pipeline_paths is not None:
        pipeline = update_with_pretrained_paths(pipeline, pretrained_pipeline_paths)

    return pipeline


def VanillaCannonTRF(trf: TemporalReceptiveField,
                     features: FeatureConfig,
                     threshold: torch.Tensor,
                     lambda_: torch.Tensor,
                     n_quantiles: int,
                     phonemes: List[str],
                     confusion_path: Optional[str] = None,
                     pretrained_pipeline_paths: Optional[List[str]] = None,
                     **kwargs) -> GroupVanillaCannonTRFForwardPipeline:
    # NB the vanilla cannon subclasses a berp class, so we have a bunch of dummy
    # parameters here. this is fine
    pipeline = GroupVanillaCannonTRFForwardPipeline(
        trf,
        ts_feature_names=list(features.ts_feature_names),
        variable_feature_names=list(features.variable_feature_names),
        threshold=torch.as_tensor(threshold),
        confusion=prepare_or_create_confusion(confusion_path, phonemes),
        lambda_=lambda_,
        n_quantiles=n_quantiles,
        **kwargs,
    )

    if pretrained_pipeline_paths is not None:
        pipeline = update_with_pretrained_paths(pipeline, pretrained_pipeline_paths)

    return pipeline


def BerpTRFEM(trf: TemporalReceptiveField,
              latent_params: Dict[str, Dict[str, BaseDistribution]],
              n_outputs: int, phonemes: List[str], 
              confusion_path: Optional[str] = None,
              pretrained_pipeline_paths: Optional[List[str]] = None,
              **kwargs):
    trf.set_params(n_outputs=n_outputs)

    # TODO lol complicated
    params = []
    base_params = PartiallyObservedModelParameters(
        lambda_=torch.tensor(1.),
        confusion=prepare_or_create_confusion(confusion_path, phonemes),
        threshold=torch.tensor(0.5),
    )
    for _ in range(20):  # DEV
        rands = torch.rand(len(latent_params))
        param_updates = {}
        for param_name, param_dist in latent_params.items():
            # TODO this structure is dumb
            param_dist = next(iter(param_dist.values()))

            if isinstance(param_dist, UniformDistribution):
                param_updates[param_name] = (rands * (param_dist.high - param_dist.low) + param_dist.low).squeeze()
            else:
                raise NotImplementedError(f"Unsupported distribution {param_dist} for {param_name}")

        params.append(replace(base_params, **param_updates))

    pipeline = make_pipeline(trf, params, pretrained_pipeline_paths, **kwargs)

    return BerpTRFEMEstimator(pipeline, **kwargs)


def BasicTRF(trf, features: FeatureConfig, **kwargs):
    pipeline = GroupVanillaTRFForwardPipeline(
        trf,
        ts_feature_names=list(features.ts_feature_names),
        variable_feature_names=list(features.variable_feature_names),
        **kwargs)
    return pipeline