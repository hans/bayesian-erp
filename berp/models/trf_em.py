from dataclasses import replace
from functools import singledispatchmethod
import logging
import re
from typing import Optional, List, Dict, Union, Iterator, Tuple

import numpy as np
from optuna.distributions import BaseDistribution, UniformDistribution
from sklearn.base import BaseEstimator, clone
import torch
from torchtyping import TensorType  # type: ignore
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
    base_params = PartiallyObservedModelParameters(
        lambda_=torch.tensor(1.),
        confusion=torch.eye(n_phonemes),  # TODO need this as param
        threshold=torch.tensor(0.5),
    )
    for _ in range(10):
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

class BerpTRFForwardPipeline(BaseEstimator):

    # TODO could properly use a pipeline for some of this probably
    # TODO backport to vanilla TRF.

    def __init__(self, encoder: TemporalReceptiveField,
                 params: List[PartiallyObservedModelParameters],
                 param_weights: Optional[Responsibilities] = None,
                 **kwargs):
        self.encoder = encoder
        self.params = params
        self.param_weights = param_weights if param_weights is not None else \
            torch.ones(len(self.params), dtype=torch.float) / len(self.params)

        self.delayer = TRFDelayer(encoder.tmin, encoder.tmax, encoder.sfreq)

        if kwargs:
            L.warning(f"Unused kwargs: {kwargs}")

    def _prime(self, dataset: BerpDataset):
        if hasattr(self, "dataset_name_"):
            self._check_primed(dataset)
            return

        self.dataset_name_ = dataset.name
        self.dataset_ts_features_ = dataset.n_ts_features
        self.dataset_shape_ = (dataset.n_samples, dataset.n_total_features)

        # Prepare scatter and delay transform.
        self._dummy_design_matrix = self._make_design_matrix(dataset)

    def _check_primed(self, dataset: BerpDataset):
        """
        When the pipeline has already been primed, verify that input dataset is compatible.
        """
        assert self.dataset_name_ == dataset.name
        assert self.dataset_shape_ == (dataset.n_samples, dataset.n_total_features)

    def _make_design_matrix(self, dataset: BerpDataset):
        """
        Prepare a design matrix with time series values inserted, leaving variable-onset values
        zero. Should then be combined with `_scatter_variable`.
        """

        dummy_variable_predictors: TRFPredictors = \
            torch.zeros(dataset.n_samples, dataset.X_variable.shape[1], dtype=dataset.X_ts.dtype)
        dummy_predictors = torch.concat([dataset.X_ts, dummy_variable_predictors], dim=1)
        design_matrix, _ = self.delayer.transform(dummy_predictors)
        return design_matrix

    def _scatter_variable(self,
                          dataset: BerpDataset,
                          recognition_points: TensorType[B, torch.long],
                          out: TRFDesignMatrix,
                          out_weight: float = 1.,
                          ) -> TRFDesignMatrix:
        """
        Scatter variable predictors into design matrix.

        Args:
            dataset:
            recognition_points:
            out: If not `None`, scatter-add to this tensor rather than
                returning a modified copy of the dummy design matrix.
            out_weight: apply this weight to the scatter-add.
        """

        assert len(recognition_points) == dataset.X_variable.shape[0]

        feature_start_idx = dataset.n_ts_features
        

        out[:, feature_start_idx:, :] = 0.

        # Compute recognition onset times and convert to sample representation.
        recognition_onsets = torch.gather(dataset.phoneme_onsets, 1, recognition_points.unsqueeze(1)).squeeze(1)
        recognition_onsets_samp = time_to_sample(recognition_onsets, self.encoder.sfreq)

        # Scatter-add, broadcasting over delay axis.
        to_add = (out_weight * dataset.X_variable).unsqueeze(2)
        out[recognition_onsets_samp, feature_start_idx:, :] += to_add

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
    
    def _pre_transform(self, dataset: BerpDataset,
                       out: Optional[TRFDesignMatrix] = None) -> TRFDesignMatrix:
        """
        Run a forward pass, averaging out model parameters.
        """

        acc = out or self._dummy_design_matrix.clone()
        for params, weight in zip(self.params, self.param_weights):
            acc = self._pre_transform_single(dataset, params, out=acc, out_weight=weight)
        return acc

    def _pre_transform_expanded(self, dataset: BerpDataset,
                                out: Optional[TRFDesignMatrix] = None) -> List[TRFDesignMatrix]:
        """
        Run a forward pass, returning a list of design matrices for each parameter option.
        """
        ret = []
        for params in self.params:
            acc = out.clone() if out is not None else self._dummy_design_matrix.clone()
            acc = self._pre_transform_single(dataset, params, out=acc)
            ret.append(acc)
        return ret

    def fit(self, dataset: BerpDataset) -> "BerpTRFForwardPipeline":
        self._prime(dataset)
        design_matrix = self._pre_transform(dataset)
        self.encoder.fit(design_matrix, dataset.Y)
        return self

    def partial_fit(self, dataset: BerpDataset) -> "BerpTRFForwardPipeline":
        self._prime(dataset)
        design_matrix = self._pre_transform(dataset)
        self.encoder.partial_fit(design_matrix, dataset.Y)
        return self

    def predict(self, dataset: BerpDataset) -> TRFResponse:
        # TODO assumes will not be training set -- could be expensive assumption
        design_matrix = self._pre_transform(dataset, out=self._make_design_matrix(dataset))
        return self.encoder.predict(design_matrix)

    def score(self, dataset: BerpDataset) -> float:
        # TODO assumes will not be training set -- could be expensive assumption
        design_matrix = self._pre_transform(dataset, out=self._make_design_matrix(dataset))
        return self.encoder.score(design_matrix, dataset.Y)

    def log_likelihood(self, dataset: BerpDataset) -> TensorType[B, torch.float]:
        # TODO assumes will not be training set -- could be expensive assumption
        design_matrix = self._pre_transform(dataset, out=self._make_design_matrix(dataset))
        return self.encoder.log_likelihood(design_matrix, dataset.Y).sum()

    def log_likelihood_expanded(self, dataset: BerpDataset) -> TensorType["param_grid", B, torch.float]:
        """
        Compute dataset log-likelihood for each parameter option
        independently.
        """
        design_matrices = self._pre_transform_expanded(dataset, out=self._make_design_matrix(dataset))
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
        self.params = params
        self.param_weights = param_weights if param_weights is not None else \
            torch.ones(len(self.params), dtype=torch.float) / len(self.params)

        self.pipelines_: Dict[str, BerpTRFForwardPipeline] = {}

        if kwargs:
            L.warning(f"Unused kwargs: {kwargs}")

    def _prime(self, dataset: NestedBerpDataset):
        print("--------PRIMED")
        if hasattr(self, "dataset_names_"):
            self._check_primed(dataset)

        self.dataset_names_ = dataset.names

        self.pipelines_ = {}
        # TODO assert exhaustive (one subdataset per subject)
        for d in dataset.datasets:
            subject_name = subject_re.match(d.name).group(1)
            pipeline = BerpTRFForwardPipeline(
                clone(self.encoder),
                self.params,
                self.param_weights)
            pipeline._prime(d)
            self.pipelines_[subject_name] = pipeline

        self._prepare_params_scatter("encoder", self.pipelines_)

    def _get_pipelines(self, dataset: NestedBerpDataset) -> Iterator[Tuple[BerpDataset, BerpTRFForwardPipeline]]:
        for d in dataset.datasets:
            subject_name = subject_re.match(d.name).group(1)
            yield d, self.pipelines_[subject_name]

    def _check_primed(self, dataset: NestedBerpDataset):
        subjects = [subject_re.match(d.name).group(1) for d in dataset.datasets]
        assert set(subjects) == set(self.pipelines_.keys())
        for ds, pipe in self._get_pipelines(dataset):
            pipe._check_primed(ds)

    def fit(self, dataset: NestedBerpDataset, y=None) -> "GroupBerpTRFForwardPipeline":
        self._prime(dataset)
        for d, pipe in self._get_pipelines(dataset):
            pipe.fit(d)
        return self

    def partial_fit(self, dataset: NestedBerpDataset, y=None) -> "GroupBerpTRFForwardPipeline":
        self._prime(dataset)
        for d, pipe in self._get_pipelines(dataset):
            print(getattr(pipe, "dataset_name_"), d.name)
            pipe.partial_fit(d)
        return self

    def _get_pipeline(self, dataset: BerpDataset) -> BerpTRFForwardPipeline:
        """
        Get the relevant pipeline for making predictions on this dataset.
        """
        subject_name = subject_re.match(dataset.name).group(1)
        return self.pipelines_[subject_name]

    @typechecked
    def predict(self, dataset: BerpDataset) -> TRFResponse:
        return self._get_pipeline(dataset).predict(dataset)

    @singledispatchmethod
    def score(self, dataset) -> float:
        raise NotImplementedError

    @score.register
    @typechecked
    def _(self, dataset: BerpDataset) -> float:
        return self._get_pipeline(dataset).score(dataset)

    @score.register
    @typechecked
    def _(self, dataset: NestedBerpDataset) -> float:
        self._prime(dataset)
        import ipdb; ipdb.set_trace()
        scores = [self.score(d) for d in dataset.datasets]
        return np.mean(scores)

    @typechecked
    def log_likelihood(self, dataset: Union[BerpDataset, NestedBerpDataset]
                       ) -> Union[TensorType[B, torch.float],
                                  List[TensorType[B, torch.float]]]:
        if isinstance(dataset, BerpDataset):
            return self._get_pipeline(dataset).log_likelihood(dataset)
        else:
            self._prime(dataset)
            return [pipe.log_likelihood(d)
                    for pipe, d in zip(self.pipelines_, dataset.datasets)]

    @singledispatchmethod
    def log_likelihood_expanded(self, dataset):
        raise NotImplementedError

    @log_likelihood_expanded.register
    @typechecked
    def _(self, dataset: BerpDataset) -> TensorType["param_grid", B, torch.float]:
        return self._get_pipeline(dataset).log_likelihood_expanded(dataset)

    @log_likelihood_expanded.register
    @typechecked
    def _(self, dataset: NestedBerpDataset
          ) -> Union[List[TensorType["param_grid", B, torch.float]],
                     List[torch.Tensor]]:
        self._prime(dataset)
        return [pipe.log_likelihood_expanded(d)
                for d, pipe in self._get_pipelines(dataset)]


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

        self.n_iter = n_iter
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        
        if kwargs:
            L.warning(f"Unused kwargs: {kwargs}")

    @typechecked
    def _e_step(self, dataset: NestedBerpDataset) -> Responsibilities:
        """
        Compute responsibility values for each parameter in the grid for the
        given dataset.
        """
        resp = torch.stack(self.pipeline.log_likelihood_expanded(dataset)).sum(0)

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

            # Re-estimate encoder parameters
            self.pipeline.param_weights = self.param_resp_
            self._m_step(X)

            # TODO score on validation set
            val_score = self.score(X)
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