from dataclasses import dataclass, replace
from functools import singledispatchmethod
import logging
import re
from typing import Optional, List, Dict, Union, Iterator, Tuple

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
    confusion = torch.eye(n_phonemes) + 0.2
    confusion /= confusion.sum(dim=0, keepdim=True)

    base_params = PartiallyObservedModelParameters(
        lambda_=torch.tensor(1.),
        confusion=confusion,
        threshold=torch.tensor(0.5),
    )
    for _ in range(100):
        rands = torch.rand(len(latent_params))
        param_updates = {}
        for param_name, param_dist in latent_params.items():
            # TODO this structure is dumb
            param_dist = next(iter(param_dist.values()))

            if isinstance(param_dist, UniformDistribution):
                param_updates[param_name] = rands * (param_dist.high - param_dist.low) + param_dist.low
            else:
                raise NotImplementedError(f"Unsupported distribution {param_dist} for {param_name}")

        params.append(replace(base_params, **param_updates))

    pipeline = GroupBerpTRFForwardPipeline(trf, params, **kwargs)
    return BerpTRFEMEstimator(pipeline, **kwargs)


# HACK specific to DKZ/gillis. generalize this feature
subject_re = re.compile(r"^DKZ_\d/([^/]+)")


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


class BerpTRFForwardPipeline(BaseEstimator):

    """
    This pipeline combines a temporal receptive field with the 
    Berp latent-onset model.
    
    Additionally, it amortizes some of the more expensive parts of the
    pipeline: in particular, the generation of a design matrix for the
    time series regression.
    """

    # TODO could properly use a pipeline for some of this probably
    # TODO backport to vanilla TRF.

    def __init__(self, name: str,
                 encoder: TemporalReceptiveField,
                 params: List[PartiallyObservedModelParameters],
                 param_weights: Optional[Responsibilities] = None,
                 **kwargs):
        self.name = name
        self.encoder = encoder
        self.params = params
        self.param_weights = param_weights if param_weights is not None else \
            torch.ones(len(self.params), dtype=torch.float) / len(self.params)

        self.delayer = TRFDelayer(encoder.tmin, encoder.tmax, encoder.sfreq)

        if kwargs:
            L.warning(f"Unused kwargs: {kwargs}")

    def _prime(self, dataset: BerpDataset) -> ForwardPipelineCache: 
        key = dataset.base_name
        try:
            cache = GLOBAL_CACHE[key]
        except KeyError as e:
            raise KeyError(f"Pipeline was never primed for key {key}.") from e
        else:
            return cache.get_cache_for(dataset)

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
        for delay in range(out.shape[2]):
            # Mask out items which, with this delay, would exceed the right edge
            # of the time series.
            mask = recognition_onsets_samp + delay < out.shape[0]

            out[recognition_onsets_samp[mask] + delay,
                feature_start_idx:,
                delay] += to_add[mask]

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
    
    def _pre_transform(self, dataset: BerpDataset) -> Tuple[TRFDesignMatrix, np.ndarray]:
        """
        Run a forward pass, averaging out model parameters.
        """
        primed = self._prime(dataset)
        acc = primed.design_matrix.clone()
        
        # Ensure variable-onset features are zeroed out.
        feature_start_idx = dataset.n_ts_features
        acc[:, feature_start_idx:, :] = 0.

        for params, weight in zip(self.params, self.param_weights):
            acc = self._pre_transform_single(dataset, params, out=acc, out_weight=weight)
        return acc, primed.validation_mask

    def _pre_transform_expanded(self, dataset: BerpDataset) -> Tuple[List[TRFDesignMatrix], np.ndarray]:
        """
        Run a forward pass, returning a list of design matrices for each parameter option.
        """
        primed = self._prime(dataset)

        ret = []
        for params in self.params:
            ret.append(self._pre_transform_single(dataset, params,
                                                  out=primed.design_matrix.clone(),
                                                  out_weight=1.))
        return ret, primed.validation_mask

    def fit(self, dataset: BerpDataset) -> "BerpTRFForwardPipeline":
        design_matrix, _ = self._pre_transform(dataset)
        self.encoder.fit(design_matrix, dataset.Y)
        return self

    def partial_fit(self, dataset: BerpDataset) -> "BerpTRFForwardPipeline":
        design_matrix, validation_mask = self._pre_transform(dataset)
        self.encoder.partial_fit(design_matrix, dataset.Y, validation_mask=validation_mask)
        return self

    def predict(self, dataset: BerpDataset) -> TRFResponse:
        design_matrix, _ = self._pre_transform(dataset)
        return self.encoder.predict(design_matrix)

    def score(self, dataset: BerpDataset) -> float:
        design_matrix, _ = self._pre_transform(dataset)
        return self.encoder.score(design_matrix, dataset.Y)

    def log_likelihood(self, dataset: BerpDataset) -> TensorType[B, torch.float]:
        design_matrix, _ = self._pre_transform(dataset)
        return self.encoder.log_likelihood(design_matrix, dataset.Y).sum()

    def log_likelihood_expanded(self, dataset: BerpDataset) -> TensorType["param_grid", B, torch.float]:
        """
        Compute dataset log-likelihood for each parameter option
        independently.
        """
        design_matrices, _ = self._pre_transform_expanded(dataset)
        return torch.stack([self.encoder.log_likelihood(dm, dataset.Y).sum()
                            for dm in design_matrices])


class ScatterParamsMixin:

    scatter_key: Optional[str] = None
    scatter_targets: Optional[List[BaseEstimator]] = None

    def _prepare_params_scatter(self, scatter_key: str, scatter_targets: List[BaseEstimator]):
        self.scatter_key = scatter_key
        self.scatter_targets = scatter_targets

    def set_params(self, **params):
        import ipdb; ipdb.set_trace()
        if self.scatter_key is None:
            return super().set_params(**params)
        if not params:
            return self

        to_pop = []
        for orig_key, value in params.items():
            key, delim, sub_key = orig_key.partition("__")
            if key == self.scatter_key:
                to_pop.append(key)

                for target in self.scatter_targets:
                    target.set_params(**{sub_key: value})

        for orig_key in to_pop:
            params.pop(orig_key)

        return super().set_params(**params)


class GroupBerpTRFForwardPipeline(ScatterParamsMixin, BaseEstimator):
    
    """
    Jointly estimates many Berp encoders, with shared parameters for
    latent-onset model.
    """

    def __init__(self, encoder: TemporalReceptiveField,
                 params: List[PartiallyObservedModelParameters],
                 param_weights: Optional[Responsibilities] = None,
                 **kwargs):
        self.encoder = encoder
        self.delayer = TRFDelayer(encoder.tmin, encoder.tmax, encoder.sfreq)

        self.pipelines_: Dict[str, BerpTRFForwardPipeline] = {}

        self.params = params
        self.param_weights = param_weights if param_weights is not None else \
            torch.ones(len(self.params), dtype=torch.float) / len(self.params)

        if kwargs:
            L.warning(f"Unused kwargs: {kwargs}")

    def set_param_weights(self, weights: Responsibilities):
        self._param_weights = weights
        for pipeline in self.pipelines_.values():
            pipeline.param_weights = weights
    def get_param_weights(self) -> Responsibilities:
        return self._param_weights
    param_weights = property(get_param_weights, set_param_weights)

    def _build_cache_for_dataset(self, dataset: BerpDataset) -> ForwardPipelineCache:
        return ForwardPipelineCache(
            cache_key=dataset.name,
            design_matrix=make_dummy_design_matrix(dataset, self.delayer),
            validation_mask=_make_validation_mask(dataset, 0.1),  # TODO magic number
        )

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

            # Ensure we have a pipeline ready for the relevant dataset group.
            self._get_or_create_pipeline(d)

        self._prepare_params_scatter("encoder", self.pipelines_)

    def _get_pipeline_key(self, dataset: BerpDataset) -> str:
        """
        Compute a key by which this dataset should be mapped with others to
        a single forward pipeline. (most intuitively this should be e.g.
        a subject ID.)
        """
        # TODO specific to Gillis
        return subject_re.match(dataset.name).group(1)

    def _get_pipeline(self, dataset: BerpDataset) -> BerpTRFForwardPipeline:
        """
        Get the relevant pipeline for making predictions on this dataset.
        """
        key = self._get_pipeline_key(dataset)
        try:
            return self.pipelines_[key]
        except KeyError as e:
            raise KeyError(f"Pipeline not found for key {key}. Did you prime this model first?") from e

    def _get_or_create_pipeline(self, dataset: BerpDataset) -> BerpTRFForwardPipeline:
        try:
            return self._get_pipeline(dataset)
        except KeyError:
            key = self._get_pipeline_key(dataset)
            L.info(f"Creating pipeline for key {key}...")

            self.pipelines_[key] = BerpTRFForwardPipeline(
                key,
                clone(self.encoder),
                self.params,
                self.param_weights)
            return self.pipelines_[key]

    def _get_pipelines(self, dataset: NestedBerpDataset) -> List[Tuple[BerpDataset, BerpTRFForwardPipeline]]:
        return [(d, self._get_pipeline(d)) for d in dataset.datasets]

    def _get_or_create_pipelines(self, dataset: NestedBerpDataset) -> List[Tuple[BerpDataset, BerpTRFForwardPipeline]]:
        return [(d, self._get_or_create_pipeline(d)) for d in dataset.datasets]

    def fit(self, dataset: NestedBerpDataset, y=None) -> "GroupBerpTRFForwardPipeline":
        for d, pipe in self._get_or_create_pipelines(dataset):
            pipe.fit(d)
        return self

    def partial_fit(self, dataset: NestedBerpDataset, y=None) -> "GroupBerpTRFForwardPipeline":
        for d, pipe in self._get_or_create_pipelines(dataset):
            pipe.partial_fit(d)
        return self

    @typechecked
    def predict(self, dataset: BerpDataset) -> TRFResponse:
        return self._get_or_create_pipeline(dataset).predict(dataset)

    @singledispatchmethod
    def score(self, dataset) -> float:
        raise NotImplementedError

    @score.register
    @typechecked
    def _(self, dataset: BerpDataset) -> float:
        return self._get_or_create_pipeline(dataset).score(dataset)

    @score.register
    @typechecked
    def _(self, dataset: NestedBerpDataset) -> float:
        scores = [self.score(d) for d in dataset.datasets]
        return np.mean(scores)

    @typechecked
    def log_likelihood(self, dataset: Union[BerpDataset, NestedBerpDataset]
                       ) -> Union[TensorType[B, torch.float],
                                  List[TensorType[B, torch.float]]]:
        if isinstance(dataset, BerpDataset):
            return self._get_or_create_pipeline(dataset).log_likelihood(dataset)
        else:
            return [pipe.log_likelihood(d)
                    for d, pipe in self._get_or_create_pipelines(dataset)]

    @singledispatchmethod
    def log_likelihood_expanded(self, dataset):
        raise NotImplementedError

    @log_likelihood_expanded.register
    @typechecked
    def _(self, dataset: BerpDataset) -> TensorType["param_grid", B, torch.float]:
        return self._get_or_create_pipeline(dataset).log_likelihood_expanded(dataset)

    @log_likelihood_expanded.register
    @typechecked
    def _(self, dataset: NestedBerpDataset
          ) -> Union[List[TensorType["param_grid", B, torch.float]],
                     List[torch.Tensor]]:
        return [pipe.log_likelihood_expanded(d)
                for d, pipe in self._get_or_create_pipelines(dataset)]


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
        """
        try:
            self.pipeline.partial_fit(dataset)
        except EarlyStopException: pass

    def partial_fit(self, X: NestedBerpDataset, y=None) -> "BerpTRFEMEstimator":
        best_score = -np.inf
        no_improvement_count = 0
        for _ in range(self.n_iter):
            self.param_resp_ = self._e_step(X)
            L.info("E-step finished")

            # Re-estimate encoder parameters
            self._m_step(X)
            L.info("M-step finished")

            # TODO score on validation set
            val_score = self.score(X)
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