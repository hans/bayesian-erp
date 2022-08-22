from dataclasses import replace
import logging
from typing import Optional, List, Dict, Union

import numpy as np
from optuna.distributions import BaseDistribution, UniformDistribution
from sklearn.base import BaseEstimator
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

    pipeline = BerpTRFForwardPipeline(trf, params, **kwargs)
    return BerpTRFEMEstimator(pipeline, **kwargs)


class BerpTRFForwardPipeline(BaseEstimator):

    # TODO could properly use a pipeline for some of this probably
    # TODO backport to vanilla TRF.

    def __init__(self, encoder: TemporalReceptiveField,
                 params: List[PartiallyObservedModelParameters],
                 param_weights: Optional[Responsibilities] = None,
                 **kwargs):
        self.encoder = encoder
        self.params = params
        self.param_weights = param_weights or \
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
        dummy_variable_predictors: TRFPredictors = \
            torch.zeros(dataset.n_samples, dataset.X_variable.shape[1], dtype=dataset.X_ts.dtype)
        dummy_predictors = torch.concat([dataset.X_ts, dummy_variable_predictors], dim=1)
        self._dummy_design_matrix, _ = self.delayer.transform(dummy_predictors)

    def _check_primed(self, dataset: BerpDataset):
        """
        When the pipeline has already been primed, verify that input dataset is compatible.
        """
        assert self.dataset_name_ == dataset.name
        assert self.dataset_shape_ == (dataset.n_samples, dataset.n_total_features)

    def _scatter_variable(self,
                          dataset: BerpDataset,
                          recognition_points: TensorType[B, torch.long],
                          out: Optional[TRFDesignMatrix] = None,
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
        
        if out is None:
            out = self._dummy_design_matrix.clone()
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
                              out: Optional[TRFDesignMatrix] = None,
                              out_weight: float = 1.,
                              ) -> TRFDesignMatrix:
        self._prime(dataset)

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
    
    def _pre_transform(self, dataset: BerpDataset) -> TRFDesignMatrix:
        """
        Run a forward pass, averaging out model parameters.
        """

        self._prime(dataset)

        acc = None
        for params, weight in zip(self.params, self.param_weights):
            acc = self._pre_transform_single(dataset, params, out=acc, out_weight=weight)
        return acc

    def _pre_transform_expanded(self, dataset: BerpDataset) -> List[TRFDesignMatrix]:
        """
        Run a forward pass, returning a list of design matrices for each parameter option.
        """

        self._prime(dataset)
        return [self._pre_transform_single(dataset, params) for params in self.params]

    def fit(self, dataset: BerpDataset) -> "BerpTRFForwardPipeline":
        design_matrix = self._pre_transform(dataset)
        self.encoder.fit(design_matrix, dataset.Y)
        return self

    def partial_fit(self, dataset: BerpDataset) -> "BerpTRFForwardPipeline":
        design_matrix = self._pre_transform(dataset)
        self.encoder.partial_fit(design_matrix, dataset.Y)
        return self

    def predict(self, dataset: BerpDataset) -> TRFResponse:
        design_matrix = self._pre_transform(dataset)
        return self.encoder.predict(design_matrix)

    def score(self, dataset: BerpDataset) -> float:
        design_matrix = self._pre_transform(dataset)
        return self.encoder.score(design_matrix, dataset.Y)

    def log_likelihood(self, dataset: BerpDataset) -> TensorType[B, torch.float]:
        design_matrix = self._pre_transform(dataset)
        return self.encoder.log_likelihood(design_matrix, dataset.Y).sum()

    def log_likelihood_expanded(self, dataset: BerpDataset) -> TensorType["param_grid", B, torch.float]:
        """
        Compute dataset log-likelihood for each parameter option
        independently.
        """
        design_matrices = self._pre_transform_expanded(dataset)
        return torch.stack([self.encoder.log_likelihood(dm, dataset.Y).sum()
                            for dm in design_matrices])


class BerpTRFEMEstimator(BaseEstimator):
    """
    Jointly estimate parameters of a Berp model using expectation maximization.
    """

    @typechecked
    def __init__(self, pipeline: BerpTRFForwardPipeline,
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
    def _e_step(self, dataset: BerpDataset) -> Responsibilities:
        """
        Compute responsibility values for each parameter in the grid for the
        given dataset.
        """
        resp = self.pipeline.log_likelihood_expanded(dataset)

        # Convert to probabilities
        resp -= resp.max()
        resp = resp.exp()
        resp = resp / resp.sum()
        return resp

    def _m_step(self, dataset: BerpDataset):
        """
        Re-estimate TRF model conditioned on the current parameter weights.
        """
        try:
            self.pipeline.partial_fit(dataset)
        except EarlyStopException: pass

    def partial_fit(self, X: Union[NestedBerpDataset, BerpDataset]) -> "BerpTRFEMEstimator":
        if isinstance(X, NestedBerpDataset):
            for x in X.datasets:
                self.partial_fit(x)
            return self

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

    def predict(self, dataset: BerpDataset) -> TRFResponse:
        return self.pipeline.predict(dataset)

    def score(self, dataset: BerpDataset):
        return self.pipeline.score(dataset)

    def log_likelihood(self, dataset: BerpDataset) -> torch.Tensor:
        return self.pipeline.log_likelihood(dataset)