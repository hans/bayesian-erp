from dataclasses import replace
import logging
from typing import *

import numpy as np
from sklearn.base import BaseEstimator
import torch
from torchtyping import TensorType

from berp.datasets import BerpDataset, NestedBerpDataset
from berp.models.reindexing_regression import scatter_model, PartiallyObservedModelParameters
from berp.models.trf import TemporalReceptiveField, TRFPredictors, TRFDesignMatrix, TRFResponse, TRFDelayer
from berp.solvers import Solver
from berp.typing import is_probability

L = logging.getLogger(__name__)

# Type variables
P = "num_params"
Responsibilities = TensorType[P, is_probability]


class BerpTRFEMEstimator(BaseEstimator):
    """
    Jointly estimate parameters of a Berp model using expectation maximization.
    """

    def __init__(self, encoder: TemporalReceptiveField,
                 optim: Solver, param_grid: torch.Tensor,
                 n_iter=1, warm_start=True,
                 early_stopping: Optional[int] = 1,
                 **kwargs):
        self.encoder = encoder
        self.optim = optim
        self.delayer = TRFDelayer(encoder.tmin, encoder.tmax, encoder.sfreq)

        self.param_grid = param_grid
        self.n_iter = n_iter
        self.warm_start = warm_start
        
        if kwargs:
            L.warning(f"Unused kwargs: {kwargs}")

    def _initialize(self):
        # Parameter responsibilities
        self.param_resp_ = torch.rand_like(self.param_grid)
        self.param_resp_ /= self.param_resp_.sum()

        self.param_template = PartiallyObservedModelParameters(
            lambda_=torch.zeros(1),
            confusion=torch.eye(5),  # TODO
            threshold=torch.tensor(0.5),
        )

    def _e_step(self, X: BerpDataset) -> Responsibilities:
        """
        Compute responsibility values for each parameter in the grid for the given dataset.
        """
        resp = torch.zeros(len(self.param_grid), dtype=torch.float)
        for i, param in enumerate(self.param_grid):
            params = replace(self.param_template, threshold=param)
            _, _, design_matrix = scatter_model(params, X)
            test_ll = self.encoder.log_likelihood(design_matrix)
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
        X_mixed = torch.empty((dataset.n_samples, dataset.n_total_features), dtype=torch.float)
        for param, resp in zip(self.param_grid, self.param_resp_):
            params = replace(self.param_template, threshold=param)

            # TODO scatter-add in place? if we're looking to save time/space
            _, _, design_matrix = scatter_model(params, dataset)
            X_mixed += resp * design_matrix

        delayed = self.delayer.transform(X_mixed)
        return delayed

    def _m_step(self, dataset: BerpDataset):
        """
        Re-estimate TRF model conditioned on the current parameter weights.
        """
        X_mixed = self._weighted_design_matrix(dataset, self.param_resp_)
        self.encoder.partial_fit(X_mixed, dataset.Y)

    def partial_fit(self, X: Union[NestedBerpDataset, BerpDataset]) -> "BerpTRFEMEstimator":
        if not self.warm_start or not hasattr(self, "param_resp_"):
            self._initialize()

        if isinstance(X, NestedBerpDataset):
            for x in X.datasets:
                self.partial_fit(x)
            return self

        best_score = -np.inf
        no_improvement_count = 0
        for _ in range(self.n_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)

            # TODO score on validation set
            val_score = self.score(X)
            if val_score > best_score:
                best_score = val_score
                no_improvement_count = 0
            elif no_improvement_count > self.early_stopping:
                L.warning("Early stopping")
                break
            else:
                no_improvement_count += 1

        return self

    def predict(self, X: BerpDataset) -> TRFResponse:
        X_mixed = self._weighted_design_matrix(X)
        return self.encoder.predict(X_mixed)

    def log_likelihood(self, X: BerpDataset) -> torch.Tensor:
        X_mixed = self._weighted_design_matrix(X)
        return self.encoder.log_likelihood(X_mixed)