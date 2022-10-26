from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import singledispatchmethod
import logging
import pickle
import re
from typing import Optional, List, Dict, Union, Tuple, TypeVar, Generic, Iterator

from hydra.utils import to_absolute_path
import numpy as np
from optuna.distributions import BaseDistribution, UniformDistribution
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import ShuffleSplit
import torch
from torchtyping import TensorType  # type: ignore
from tqdm.auto import tqdm, trange
from typeguard import typechecked

from berp.cv import EarlyStopException
from berp.datasets import BerpDataset, NestedBerpDataset
from berp.models.reindexing_regression import \
    predictive_model, recognition_point_model, PartiallyObservedModelParameters
from berp.models.trf import TemporalReceptiveField, TRFPredictors, \
    TRFDesignMatrix, TRFResponse, TRFDelayer
from berp.tensorboard import Tensorboard
from berp.typing import is_probability, DIMS
from berp.util import time_to_sample

L = logging.getLogger(__name__)

# Type variables
B, N_C, N_P, N_F, N_F_T, V_P, T, S = \
    DIMS.B, DIMS.N_C, DIMS.N_P, DIMS.N_F, DIMS.N_F_T, DIMS.V_P, DIMS.T, DIMS.S
P = "num_params"
Responsibilities = TensorType[P, is_probability]


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

    def _set_encoder(self, key: str, encoder: Encoder):
        self.encoders_[key] = encoder
        self._prepare_params_scatter("encoder", list(self.encoders_.values()))

    def _get_or_create_encoder(self, dataset: BerpDataset) -> Encoder:
        try:
            return self._get_encoder(dataset)
        except KeyError:
            key = self._get_encoder_key(dataset)
            L.debug(f"Creating encoder for key {key}")
            enc = clone(self.encoder)
            enc.set_name(key)
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

    def _fit(self, encoder: Encoder, datasets: List[BerpDataset]):
        design_matrices, Ys = [], []
        for d in datasets:
            design_matrix, _ = self.pre_transform(d)
            design_matrices.append(design_matrix)
            Ys.append(d.Y)

        design_matrix = torch.cat(design_matrices, dim=0)
        Y = torch.cat(Ys, dim=0)
        encoder.fit(design_matrix, Y)

    def fit(self, dataset: NestedBerpDataset, y=None) -> "GroupBerpTRFForwardPipeline":
        # Group datasets by encoder and fit jointly.
        grouped_datasets = defaultdict(list)
        for dataset in dataset.datasets:
            key = self._get_encoder_key(dataset)
            grouped_datasets[key].append(dataset)

        for key, datasets in tqdm(grouped_datasets.items(), desc="Exact fit"):
            enc = self._get_or_create_encoder(datasets[0])
            self._fit(enc, datasets)

        return self

    def _partial_fit(self, encoder: Encoder, dataset: BerpDataset):
        design_matrix, validation_mask = self.pre_transform(dataset)
        encoder.partial_fit(design_matrix, dataset.Y, validation_mask=validation_mask,
                            dataset_tag=dataset.name)

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
                 recognition_scatter_point: float = 0.0,
                 prior_scatter_point: Tuple[int, float] = (0, 0.0),
                 **kwargs):
        """
        Args:
            encoder:
            params:
            param_weights:
            recognition_scatter_point: Describes expected recognition time logic when
                words are recognized given some perceptual input.
                See `reindexing_regression.recognition_points_to_times`.
            prior_scatter_point: Describes expected recognition time logic when words are
                recognized prior to perceptual input.
                See `reindexing_regression.recognition_points_to_times`
        """
        super().__init__(encoder, **kwargs)

        self.params = params
        self.param_weights = param_weights if param_weights is not None else \
            torch.ones(len(self.params), dtype=torch.float) / len(self.params)
        
        self.recognition_scatter_point = recognition_scatter_point
        self.prior_scatter_point = prior_scatter_point

    def _scatter_variable(self,
                          dataset: BerpDataset,
                          recognition_times: TensorType[B, float],
                          out: TRFDesignMatrix,
                          out_weight: float = 1.,
                          ) -> TRFDesignMatrix:
        """
        Scatter variable-onset predictors into design matrix, which is a
        lagged time series.

        Args:
            dataset:
            recognition_times:
            out: scatter-add to this tensor
            out_weight: apply this weight to the scatter-add.
        """
        
        # TODO unittest this !!

        assert len(recognition_times) == dataset.X_variable.shape[0]
        feature_start_idx = dataset.n_ts_features

        # Compute recognition onset times and convert to sample representation.
        recognition_times_samp = time_to_sample(recognition_times, self.encoder.sfreq)

        # Scatter-add, lagging over delay axis.
        to_add = out_weight * dataset.X_variable
        scatter_add(out[:, feature_start_idx:, :], recognition_times_samp, to_add)

        return out

    def get_recognition_points(self, dataset: BerpDataset,
                               params: PartiallyObservedModelParameters,
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

    def get_recognition_times(self, dataset: BerpDataset,
                              params: PartiallyObservedModelParameters,
                              ) -> TensorType[torch.float]:
        recognition_points = self.get_recognition_points(dataset, params)
        return self._scatter_recognition_points(dataset, recognition_points)

    def _pre_transform_single(self, dataset: BerpDataset,
                              params: PartiallyObservedModelParameters,
                              out: TRFDesignMatrix,
                              out_weight: float = 1.,
                              ) -> TRFDesignMatrix:
        """
        Run word recognition logic for the given dataset and parameters,
        producing a regression design matrix.
        """
        recognition_points = self.get_recognition_points(dataset, params)
        recognition_times = recognition_points_to_times(
            recognition_points,
            dataset.phoneme_onsets_global,
            dataset.phoneme_offsets_global,
            dataset.word_lengths,
            scatter_point=self.scatter_point,
            prior_scatter_point=self.prior_scatter_point,
        )

        design_matrix: TRFDesignMatrix = self._scatter_variable(
            dataset, recognition_times,
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


class GroupBerpFixedTRFForwardPipeline(GroupBerpTRFForwardPipeline):

    """
    Group pipeline for variable-onset TRF with just one parameter option.
    Provides easy scikit-learn-friendly parameter accessors.
    """

    _param_keys = ["threshold", "confusion", "lambda_"]

    def __init__(self, encoder: Encoder,
                 threshold: torch.Tensor,
                 confusion: torch.Tensor,
                 lambda_: torch.Tensor,
                 recognition_scatter_point: float = 0,
                 **kwargs):
        params = PartiallyObservedModelParameters(
            threshold=threshold,
            confusion=confusion,
            lambda_=lambda_,
        )
        super().__init__(encoder, [params],
            recognition_scatter_point=recognition_scatter_point,
            **kwargs)

        self.threshold = threshold
        self.confusion = confusion
        self.lambda_ = lambda_

    def get_params(self, deep=True):
        ret = super().get_params(deep)
        for param_key in self._param_keys:
            ret[param_key] = getattr(self.params[0], param_key)
        return ret

    def set_params(self, **params):
        # Add passthrough for updating threshold if we are just using one parameter
        # value.
        for param_key in self._param_keys:
            if param_key in params:
                setattr(self.params[0], param_key, torch.as_tensor(params.pop(param_key)))

        super().set_params(**params)


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
    confusion_path: str, dataset_phonemes: List[str]) -> torch.Tensor:
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
    confusion_matrix += 1.

    # Normalize. Each column should be a probability distribution.
    confusion_matrix /= confusion_matrix.sum(axis=0, keepdims=True) + 1e-5

    return torch.tensor(confusion_matrix)


def prepare_or_create_confusion(confusion_path: Optional[str],
                                phonemes: List[str]) -> torch.Tensor:
    if confusion_path is not None:
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
    params: List[PartiallyObservedModelParameters],
    pretrained_pipeline_paths: Optional[List[str]] = None,
    *args, **kwargs) -> GroupBerpTRFForwardPipeline:

    pipeline = GroupBerpTRFForwardPipeline(trf, params=params, **kwargs)

    if pretrained_pipeline_paths is not None:
        pipeline = update_with_pretrained_paths(pipeline, pretrained_pipeline_paths)

    return pipeline


def BerpTRFFixed(trf: TemporalReceptiveField,
                 threshold: torch.Tensor,
                 n_outputs: int,
                 phonemes: List[str],
                 confusion_path: Optional[str] = None,
                 pretrained_pipeline_paths: Optional[List[str]] = None,
                 **kwargs) -> GroupBerpFixedTRFForwardPipeline:
    trf.set_params(n_outputs=n_outputs)

    pipeline = GroupBerpFixedTRFForwardPipeline(
        trf,
        threshold=threshold,
        confusion=prepare_or_create_confusion(confusion_path, phonemes),
        lambda_=torch.tensor(1.),
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


def BasicTRF(trf, n_outputs: int, **kwargs):
    trf.set_params(n_outputs=n_outputs)
    pipeline = GroupVanillaTRFForwardPipeline(trf, **kwargs)
    return pipeline