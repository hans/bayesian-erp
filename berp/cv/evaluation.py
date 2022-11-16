
from typing import Callable, TypeVar, Generic, Optional, Union
from typeguard import typechecked

import numpy as np

from berp.datasets import BerpDataset, NestedBerpDataset
from berp.models.trf_em import GroupTRFForwardPipeline
from berp.tensorboard import tb_add_scalar, tb_global_step


def log_scores(dataset: NestedBerpDataset, scores: np.ndarray, tag="score"):
    for ds, ds_scores in zip(dataset.datasets, scores):
        assert ds.sensor_names is not None
        for sensor_idx, sensor_score in enumerate(ds_scores):
            tb_add_scalar(f"{tag}/{ds.name}/{ds.sensor_names[sensor_idx]}", sensor_score)

Score = TypeVar("Score")
class BaselinedScorer(Generic[Score]):

    """
    Multidimensional time series scorer which computes per-dimension
    improvement, optionally relative to a baseline model.
    """

    def __init__(self,
                 baseline_model: Optional[GroupTRFForwardPipeline] = None,
                 aggregation_fn: Callable[[np.ndarray], Score] = np.mean):
        self.baseline_model = baseline_model
        self.aggregation_fn = aggregation_fn

    def __call__(self, estimator: GroupTRFForwardPipeline,
                 X: Union[BerpDataset, NestedBerpDataset],
                 y=None) -> Score:
        if isinstance(X, BerpDataset):
            X = NestedBerpDataset([X])
        
        score = estimator.score_multidimensional(X, y)
        log_scores(X, score, tag="score/est")

        if self.baseline_model is not None:
            baseline_score = self.baseline_model.score_multidimensional(X, y)

            if isinstance(X, BerpDataset):
                assert baseline_score.ndim == 1
            elif isinstance(X, NestedBerpDataset):
                assert baseline_score.ndim == 2
            else:
                raise ValueError()
            assert baseline_score.shape == score.shape

            log_scores(X, baseline_score, tag="score/baseline")

            score = score - baseline_score
            log_scores(X, score, tag="score/delta")

        tb_global_step()
        return self.aggregation_fn(score)