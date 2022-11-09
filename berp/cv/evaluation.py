
from typing import Callable, TypeVar, Generic
from typeguard import typechecked

import numpy as np

from berp.datasets import BerpDataset, NestedBerpDataset
from berp.models.trf_em import GroupTRFForwardPipeline


Score = TypeVar("Score")
class BaselinedScorer(Generic[Score]):

    """
    Multidimensional time series scorer which computes per-dimension
    improvement relative to a baseline model.
    """

    def __init__(self, baseline_model: GroupTRFForwardPipeline,
                 aggregation_fn: Callable[[np.ndarray], Score] = np.mean):
        self.baseline_model = baseline_model
        self.aggregation_fn = aggregation_fn

    def __call__(self, estimator: GroupTRFForwardPipeline,
                 X, y=None) -> Score:
        baseline_score = self.baseline_model.score_multidimensional(X, y)
        score = estimator.score_multidimensional(X, y)

        # TODO make sure we get multidimensional results
        if isinstance(X, BerpDataset):
            assert baseline_score.ndim == 1
        elif isinstance(X, NestedBerpDataset):
            assert baseline_score.ndim == 2
        else:
            raise ValueError()
        assert baseline_score.shape == score.shape

        ret = score - baseline_score
        return self.aggregation_fn(ret)