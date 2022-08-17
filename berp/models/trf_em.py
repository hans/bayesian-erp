import logging
from typing import *

import numpy as np
from sklearn.base import BaseEstimator
import torch
from torchtyping import TensorType

from berp.datasets import BerpDataset, NestedBerpDataset
from berp.models.trf import TemporalReceptiveField
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

        self.param_grid = param_grid
        self.n_iter = n_iter
        self.warm_start = warm_start
        
        if kwargs:
            L.warning(f"Unused kwargs: {kwargs}")

    def _initialize(self):
        self.param_weights_ = torch.rand_like(self.param_grid)
        self.param_weights_ /= self.param_weights_.sum()

    def _e_step(self, X: BerpDataset) -> Responsibilities:
        """
        Compute responsibility values for each parameter in the grid for the given dataset.
        """
        raise NotImplementedError()

    def _m_step(self, X: BerpDataset, resp: Responsibilities) -> "BerpTRFEMEstimator":
        """
        Re-estimate TRF model conditioned on the current parameter weights.
        """
        raise NotImplementedError()

    def partial_fit(self, X: Union[NestedBerpDataset, BerpDataset]) -> "BerpTRFEMEstimator":
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