from unittest.mock import MagicMock

import numpy as np

from berp.cv.evaluation import BaselinedScorer
from berp.datasets import NestedBerpDataset
from berp.models.trf_em import GroupTRFForwardPipeline


class TestBaselinedScorer:

    def test_baseline_numeric(self):
        n_sensors = 10
        baseline_scores = np.random.random((3, n_sensors))
        scores = np.random.random((3, n_sensors))

        baseline_model = MagicMock(
            spec=GroupTRFForwardPipeline,
            score_multidimensional=lambda X, y: baseline_scores,
        )

        estimator = MagicMock(
            spec=GroupTRFForwardPipeline,
            score_multidimensional=lambda X, y: scores,
        )

        X = MagicMock(spec=NestedBerpDataset, n_sensors=n_sensors, datasets=[])

        scorer = BaselinedScorer(baseline_model)
        ret = scorer(estimator, X)
        assert isinstance(ret, np.floating)
        assert ret == np.mean(scores - baseline_scores)

        scorer = BaselinedScorer(baseline_model, aggregation_fn=lambda x: x)
        ret = scorer(estimator, X)
        assert isinstance(ret, np.ndarray)
        np.testing.assert_allclose(ret, scores - baseline_scores)