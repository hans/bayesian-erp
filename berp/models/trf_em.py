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
    scatter_model, PartiallyObservedModelParameters
from berp.models.trf import TemporalReceptiveField, TRFPredictors, \
    TRFDesignMatrix, TRFResponse, TRFDelayer
from berp.solvers import Solver
from berp.typing import is_probability

L = logging.getLogger(__name__)

# Type variables
P = "num_params"
Responsibilities = TensorType[P, is_probability]


def BerpTRFEM(trf, latent_params: Dict[str, Dict[str, BaseDistribution]],
              n_outputs: int, **kwargs):
    trf.set_params(n_outputs=n_outputs)  

    # TODO param_grid
    from pprint import pprint
    pprint(kwargs)

    # TODO lol complicated
    assert list(latent_params.keys()) == ["threshold"]
    threshold_dist = next(iter(latent_params["threshold"].values()))
    assert isinstance(threshold_dist, UniformDistribution)
    param_grid = torch.rand(10) * (threshold_dist.high - threshold_dist.low) + threshold_dist.low

    return BerpTRFEMEstimator(
        encoder=trf,
        param_grid=param_grid,
        **kwargs,)


class BerpTRFEMEstimator(BaseEstimator):
    """
    Jointly estimate parameters of a Berp model using expectation maximization.
    """

    @typechecked
    def __init__(self, encoder: TemporalReceptiveField,
                 param_grid: torch.Tensor,
                 n_iter=1, warm_start=True,
                 early_stopping: Optional[int] = 1,
                 **kwargs):
        self.encoder = encoder
        self.delayer = TRFDelayer(encoder.tmin, encoder.tmax, encoder.sfreq)

        self.param_grid = param_grid
        self.n_iter = n_iter
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        
        if kwargs:
            L.warning(f"Unused kwargs: {kwargs}")

    def _initialize(self, dataset: BerpDataset):
        # Parameter responsibilities
        self.param_resp_ = torch.rand_like(self.param_grid)
        self.param_resp_ /= self.param_resp_.sum()

        # TODO this should be a param of the model
        confusion = torch.eye(len(dataset.phonemes))

        self.param_template = PartiallyObservedModelParameters(
            lambda_=torch.zeros(1),
            confusion=confusion,
            threshold=torch.tensor(0.5),
        )

    def _e_step(self, dataset: BerpDataset) -> Responsibilities:
        """
        Compute responsibility values for each parameter in the grid for the
        given dataset.
        """
        resp = torch.zeros(len(self.param_grid), dtype=torch.float)
        for i, param in enumerate(self.param_grid):
            params = replace(self.param_template, threshold=param)
            _, _, design_matrix = scatter_model(params, dataset)
            # TODO cache this sucker
            delayed, _ = self.delayer.transform(design_matrix)

            test_ll = self.encoder.log_likelihood(delayed, dataset.Y).sum()
            resp[i] = test_ll

        # Convert to probabilities
        resp -= resp.max()
        resp = resp.exp()
        resp = resp / resp.sum()
        return resp

    def _weighted_design_matrix(self, dataset: BerpDataset) -> TRFDesignMatrix:
        """
        Compute expected predictor matrix under current parameter distribution.
        """
        X_mixed = torch.empty((dataset.n_samples, dataset.n_total_features),
                              dtype=torch.float)
        for param, resp in zip(self.param_grid, self.param_resp_):
            params = replace(self.param_template, threshold=param)

            # TODO scatter-add in place? if we're looking to save time/space
            _, _, design_matrix = scatter_model(params, dataset)
            X_mixed += resp * design_matrix

        delayed, _ = self.delayer.transform(X_mixed)
        return delayed

    def _m_step(self, dataset: BerpDataset):
        """
        Re-estimate TRF model conditioned on the current parameter weights.
        """
        X_mixed = self._weighted_design_matrix(dataset)

        try:
            self.encoder.partial_fit(X_mixed, dataset.Y)
        except EarlyStopException: pass

    def partial_fit(self, X: Union[NestedBerpDataset, BerpDataset]) -> "BerpTRFEMEstimator":
        if isinstance(X, NestedBerpDataset):
            for x in X.datasets:
                self.partial_fit(x)
            return self

        if not self.warm_start or not hasattr(self, "param_resp_"):
            self._initialize(X)

        best_score = -np.inf
        no_improvement_count = 0
        for _ in range(self.n_iter):
            self.param_resp_ = self._e_step(X)
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
        X_mixed = self._weighted_design_matrix(dataset)
        return self.encoder.predict(X_mixed)

    def score(self, dataset: BerpDataset):
        X_mixed = self._weighted_design_matrix(dataset)
        return self.encoder.score(X_mixed, dataset.Y)

    def log_likelihood(self, dataset: BerpDataset) -> torch.Tensor:
        X_mixed = self._weighted_design_matrix(dataset)
        return self.encoder.log_likelihood(X_mixed, dataset.Y)