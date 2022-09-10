from dataclasses import dataclass, replace
from functools import singledispatchmethod
import logging
import re
from typing import Optional, List, Dict, Union, Tuple, TypeVar, Generic, Iterator

import numpy as np
from optuna.distributions import BaseDistribution, UniformDistribution
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ShuffleSplit
import torch
from torchtyping import TensorType  # type: ignore
from tqdm.auto import tqdm
from typeguard import typechecked

from berp.cv import EarlyStopException
from berp.datasets import BerpDataset, NestedBerpDataset
from berp.models.reindexing_regression import \
    predictive_model, recognition_point_model, PartiallyObservedModelParameters
from berp.models.trf import TemporalReceptiveField, TRFPredictors, \
    TRFDesignMatrix, TRFResponse, TRFDelayer
from berp.typing import is_probability, DIMS
from berp.util import time_to_sample

L = logging.getLogger(__name__)

# Type variables
B, N_C, N_P, N_F, N_F_T, V_P, T, S = \
    DIMS.B, DIMS.N_C, DIMS.N_P, DIMS.N_F, DIMS.N_F_T, DIMS.V_P, DIMS.T, DIMS.S
P = "num_params"
Responsibilities = TensorType[P, is_probability]


def BerpTRFEM(trf, latent_params: Dict[str, Dict[str, BaseDistribution]],
              n_outputs: int, n_phonemes: int, **kwargs):
    trf.set_params(n_outputs=n_outputs)

    # TODO param_grid
    from pprint import pprint
    pprint(kwargs)

    # TODO lol complicated
    params = []
    # TODO should be a parameter of the model
    confusion = torch.eye(n_phonemes) + 0.01
    confusion /= confusion.sum(dim=0, keepdim=True)

    base_params = PartiallyObservedModelParameters(
        lambda_=torch.tensor(1.),
        confusion=confusion,
        threshold=torch.tensor(0.5),
    )
    for _ in range(50):  # DEV
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

    pipeline = GroupBerpTRFForwardPipeline(trf, params=params, **kwargs)
    return BerpTRFEMEstimator(pipeline, **kwargs)


def BasicTRF(trf, n_outputs: int, **kwargs):
    pipeline = GroupVanillaTRFForwardPipeline(trf, params, **kwargs)
    return pipeline


# HACK specific to DKZ/gillis. generalize this feature
subject_re = re.compile(r"^DKZ_\d/([^/]+)")


def scatter_add(design_matrix: TRFDesignMatrix,
                target_samples: TensorType[B, torch.long],
                target_values: TensorType[B, "n_target_features", float],
                add=True):
    """
    Scatter-add or update values to the given samples of the TRF design matrix,
    maintaining lag structure (i.e. duplicating target values at different lags).

    If `add` is `True`, scatter-add to existing values; otherwise replace with
    given values.

    Operates in-place.
    """

    # TODO unittest

    assert target_samples.shape[0] == target_values.shape[0]
    assert target_values.shape[1] == design_matrix.shape[1]
    assert target_values.dtype == design_matrix.dtype

    # NB no copy.
    out = design_matrix

    for delay in range(out.shape[2]):
        # Mask out items which, with this delay, would exceed the right edge
        # of the time series.
        mask = target_samples + delay < out.shape[0]

        if add:
            out[target_samples[mask] + delay, :, delay] += target_values[mask]
        else:
            out[target_samples[mask] + delay, :, delay] = target_values[mask]


def _make_validation_mask(dataset: BerpDataset, validation_fraction: float,
                          random_state=None) -> np.ndarray:
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
    validation_mask: np.ndarray

    def check_compatible(self, dataset: BerpDataset):
        assert dataset.name.startswith(self.cache_key)
        
        n_samples, n_features = self.design_matrix.shape[:2]
        assert n_features == dataset.n_total_features
        
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


def make_dummy_design_matrix(dataset: BerpDataset, delayer: TRFDelayer) -> TRFDesignMatrix:
    """
    Prepare a design matrix with time series values inserted, leaving variable-onset values
    zero. Should then be combined with `_scatter_variable`.
    """
    dummy_variable_predictors: TRFPredictors = \
        torch.zeros(dataset.n_samples, dataset.X_variable.shape[1], dtype=dataset.X_ts.dtype)
    dummy_predictors = torch.concat([dataset.X_ts, dummy_variable_predictors], dim=1)
    design_matrix, _ = delayer.transform(dummy_predictors)
    return design_matrix


# Global cache container for all pipeline forwards.
# This allows us to reuse the cache between e.g. different instantiations of full
# pipelines with different hparams, but with the same data.
GLOBAL_CACHE: Dict[str, ForwardPipelineCache] = {}


clones = [0]
def clone_count(x):
    clones[0] += 1
    # print("=====+++CLONE", clones[0])
    return x.clone()


class ScatterParamsMixin:

    scatter_key: Optional[str] = None
    scatter_targets: Optional[List[BaseEstimator]] = None

    def _prepare_params_scatter(self, scatter_key: str, scatter_targets: List[BaseEstimator]):
        self.scatter_key = scatter_key
        self.scatter_targets = scatter_targets

    def set_params(self, **params):
        if self.scatter_key is None:
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


Encoder = TypeVar("Encoder")
class GroupTRFForwardPipeline(ScatterParamsMixin, BaseEstimator, Generic[Encoder]):

    """
    Abstract class. Jointly estimates many TRF encoders, optionally storing
    group-level parameters as well (e.g. latent onset model parameters).

    NOT THREAD SAFE. Updates in-place on `ForwardPipelineCache` instances.
    """

    def __init__(self, encoder: Encoder, **kwargs):
        self.encoder = encoder

        self.delayer = TRFDelayer(encoder.tmin, encoder.tmax, encoder.sfreq)
        self.encoders_: Dict[str, Encoder] = {}

        if kwargs:
            L.warning(f"Unused kwargs: {kwargs}")

    #region caching and weight sharing logic

    def _build_cache_for_dataset(self, dataset: BerpDataset) -> ForwardPipelineCache:
        return ForwardPipelineCache(
            cache_key=dataset.name,
            design_matrix=make_dummy_design_matrix(dataset, self.delayer),
            validation_mask=_make_validation_mask(dataset, 0.1),  # TODO magic number
        )

    def _get_cache_for_dataset(self, dataset: BerpDataset) -> ForwardPipelineCache:
        key = dataset.base_name
        try:
            cache = GLOBAL_CACHE[key]
        except KeyError as e:
            raise KeyError(f"Pipeline was never primed for dataset key {key}.") from e
        else:
            return cache.get_cache_for(dataset)

    def prime(self, dataset: Union[BerpDataset, NestedBerpDataset]):
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
            GLOBAL_CACHE[d.name] = self._build_cache_for_dataset(d)

            # Ensure we have an encoder ready for the relevant dataset group.
            self._get_or_create_encoder(d)

        self._prepare_params_scatter("encoder", list(self.encoders_.values()))

    def _get_encoder_key(self, dataset: BerpDataset) -> str:
        """
        Compute a key by which this dataset should be mapped with others to
        a single encoder. (most intuitively this should be e.g. a subject ID.)
        """
        # TODO specific to Gillis
        return subject_re.match(dataset.name).group(1)

    def _get_encoder(self, dataset: BerpDataset) -> Encoder:
        """
        Get the relevant encoder for making predictions on this dataset.
        """
        key = self._get_encoder_key(dataset)
        try:
            return self.encoders_[key]
        except KeyError as e:
            raise KeyError(f"encoder not found for key {key}. Did you prime this model first?") from e

    def _get_or_create_encoder(self, dataset: BerpDataset) -> Encoder:
        try:
            return self._get_encoder(dataset)
        except KeyError:
            key = self._get_encoder_key(dataset)
            L.info(f"Creating encoder for key {key}...")

            self.encoders_[key] = clone(self.encoder)
            return self.encoders_[key]

    def _get_encoders(self, dataset: NestedBerpDataset) -> List[Tuple[BerpDataset, Encoder]]:
        return [(d, self._get_encoder(d)) for d in dataset.datasets]

    def _get_or_create_encoders(self, dataset: NestedBerpDataset) -> List[Tuple[BerpDataset, Encoder]]:
        return [(d, self._get_or_create_encoder(d)) for d in dataset.datasets]

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

    def _fit(self, encoder: Encoder, dataset: BerpDataset):
        design_matrix, _ = self.pre_transform(dataset)
        encoder.fit(design_matrix, dataset.Y)

    def fit(self, dataset: NestedBerpDataset, y=None) -> "GroupBerpTRFForwardPipeline":
        for d, enc in self._get_or_create_encoders(dataset):
            self._fit(enc, d)
        return self

    def _partial_fit(self, encoder: Encoder, dataset: BerpDataset):
        design_matrix, validation_mask = self.pre_transform(dataset)
        encoder.partial_fit(design_matrix, dataset.Y, validation_mask=validation_mask)

    def partial_fit(self, dataset: NestedBerpDataset, y=None) -> "GroupBerpTRFForwardPipeline":
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
    def score(self, dataset) -> float:
        raise NotImplementedError

    @score.register
    @typechecked
    def _(self, dataset: BerpDataset) -> float:
        enc = self._get_or_create_encoder(dataset)
        design_matrix, _ = self.pre_transform(dataset)
        return enc.score(design_matrix, dataset.Y)

    @score.register
    @typechecked
    def _(self, dataset: NestedBerpDataset) -> float:
        scores = [self.score(d) for d in dataset.datasets]
        return np.mean(scores)

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
                 params: List[PartiallyObservedModelParameters],
                 param_weights: Optional[Responsibilities] = None,
                 **kwargs):
        super().__init__(encoder, **kwargs)

        self.params = params
        self.param_weights = param_weights if param_weights is not None else \
            torch.ones(len(self.params), dtype=torch.float) / len(self.params)

    def _scatter_variable(self,
                          dataset: BerpDataset,
                          recognition_points: TensorType[B, torch.long],
                          out: TRFDesignMatrix,
                          out_weight: float = 1.,
                          ) -> TRFDesignMatrix:
        """
        Scatter variable predictors into design matrix, which is a
        lagged time series.

        Args:
            dataset:
            recognition_points:
            out: If not `None`, scatter-add to this tensor rather than
                returning a modified copy of the dummy design matrix.
            out_weight: apply this weight to the scatter-add.
        """
        
        # TODO unittest this !!

        assert len(recognition_points) == dataset.X_variable.shape[0]
        feature_start_idx = dataset.n_ts_features

        # Compute recognition onset times and convert to sample representation.
        recognition_onsets = torch.gather(
            dataset.phoneme_onsets_global, 1,
            recognition_points.unsqueeze(1)).squeeze(1)
        recognition_onsets_samp = time_to_sample(recognition_onsets, self.encoder.sfreq)

        # Scatter-add, lagging over delay axis.
        to_add = out_weight * dataset.X_variable
        scatter_add(out[:, feature_start_idx:, :], recognition_onsets_samp, to_add)

        return out

    def _pre_transform_single(self, dataset: BerpDataset,
                              params: PartiallyObservedModelParameters,
                              out: TRFDesignMatrix,
                              out_weight: float = 1.,
                              ) -> TRFDesignMatrix:
        p_word_posterior = predictive_model(
            dataset.p_word, dataset.candidate_phonemes,
            params.confusion, params.lambda_
        )
        recognition_points = recognition_point_model(
            p_word_posterior, dataset.word_lengths, params.threshold
        )

        design_matrix: TRFDesignMatrix = self._scatter_variable(
            dataset,
            recognition_points,
            out=out, out_weight=out_weight)
        return design_matrix
    
    def pre_transform(self, dataset: BerpDataset) -> Tuple[TRFDesignMatrix, np.ndarray]:
        primed = self._get_cache_for_dataset(dataset)
        # acc = primed.design_matrix.clone()
        acc = clone_count(primed.design_matrix)
        
        # Ensure variable-onset features are zeroed out.
        feature_start_idx = dataset.n_ts_features
        acc[:, feature_start_idx:, :] = 0.

        for params, weight in zip(self.params, self.param_weights):
            acc = self._pre_transform_single(dataset, params, out=acc, out_weight=weight)
        return acc, primed.validation_mask

    def pre_transform_expanded(self, dataset: BerpDataset) -> Iterator[Tuple[TRFDesignMatrix, np.ndarray]]:
        primed = self._get_cache_for_dataset(dataset)
        feature_start_idx = dataset.n_ts_features

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

        feature_start_idx = dataset.n_ts_features
        scatter_add(design_matrix[:, feature_start_idx:, :],
                    recognition_onsets_samp,
                    dataset.X_variable,
                    add=False)

    def pre_transform(self, dataset: BerpDataset) -> Tuple[TRFDesignMatrix, np.ndarray]:
        primed = self._get_cache_for_dataset(dataset)
        
        # Fine to not clone cached values -- they are constant.
        self._scatter(dataset, primed.design_matrix)

        return primed.design_matrix, primed.validation_mask


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

    def prime(self, dataset):
        return self.pipeline.prime(dataset)

    def partial_fit(self, X: NestedBerpDataset, y=None,
                    X_val: Optional[NestedBerpDataset] = None
                    ) -> "BerpTRFEMEstimator":
        best_score = -np.inf
        no_improvement_count = 0
        for _ in range(self.n_iter):
            self.param_resp_ = self._e_step(X)
            L.info("E-step finished")

            # Re-estimate encoder parameters
            self._m_step(X)
            L.info("M-step finished")

            if X_val is not None:
                val_score = self.score(X_val)
                L.info("Val score: %f", val_score)
                if val_score > best_score:
                    best_score = val_score
                    no_improvement_count = 0
                elif self.early_stopping is not None and no_improvement_count > self.early_stopping:
                    L.warning("Early stopping")
                    break
                else:
                    no_improvement_count += 1

        return self

    def fit(self, X: NestedBerpDataset, y=None) -> "BerpTRFEMEstimator":
        return self.partial_fit(X)

    def predict(self, dataset: NestedBerpDataset) -> TRFResponse:
        return self.pipeline.predict(dataset)

    def score(self, dataset: NestedBerpDataset, y=None):
        return self.pipeline.score(dataset)

    def log_likelihood(self, dataset: NestedBerpDataset, y=None) -> torch.Tensor:
        return self.pipeline.log_likelihood(dataset)