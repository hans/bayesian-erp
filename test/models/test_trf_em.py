from dataclasses import replace
from typing import List, Tuple

import pytest
import torch

from berp.datasets import BerpDataset
from berp.datasets.base import NestedBerpDataset
from berp.generators import thresholded_recognition_simple as generator
from berp.generators.stimulus import RandomStimulusGenerator
from berp.models.reindexing_regression import PartiallyObservedModelParameters, ModelParameters
from berp.models.trf import TemporalReceptiveField
from berp.models.trf_em import BerpTRFEMEstimator, GroupBerpTRFForwardPipeline
from berp.solvers import Solver, AdamSolver


@pytest.fixture
def synth_params() -> ModelParameters:
    return ModelParameters(
        lambda_=torch.tensor(1.0),
        confusion=generator.phoneme_confusion,
        threshold=torch.distributions.Beta(1.2, 1.2).sample(),

        # NB only used for generation, not in model
        a=torch.tensor(0.2),
        b=torch.tensor(0.1),
        coef=torch.tensor([-1]),
        sigma=torch.tensor(5.0),
    )

@pytest.fixture
def dataset(synth_params) -> BerpDataset:
    stim = RandomStimulusGenerator(num_words=1000, num_phonemes=10, phoneme_voc_size=synth_params.confusion.shape[0],
                                   word_surprisal_params=(2.0, 0.5))
    ds_args = dict(
        response_type="gaussian",
        epoch_window=(0, 0.55), # TODO unused
        include_intercept=False,
        sample_rate=48)

    # HACK: resample until it doesn't error. should fix sloppy indexing at some point.
    while True:
        try:
            dataset = generator.sample_dataset(synth_params, stim, **ds_args)
        except RuntimeError: continue
        else: break

    # TODO test dataset
    return dataset


@pytest.fixture
def optim() -> Solver:
    return AdamSolver(early_stopping=False)

@pytest.fixture
def trf(dataset: BerpDataset, optim):
    return TemporalReceptiveField(
        tmin=0,
        tmax=0.55,
        sfreq=dataset.sample_rate,
        n_outputs=dataset.n_sensors,
        optim=optim,
        alpha=1e-2)


@pytest.fixture
def model_params(dataset: BerpDataset):
    return PartiallyObservedModelParameters(
        lambda_=torch.tensor(1.),
        confusion=torch.eye(dataset.n_phonemes),  # TODO better?
        threshold=torch.tensor(0.5))


@pytest.fixture
def model_param_grid(model_params):
    # Generate a grid of threshold values.
    grid = []
    for _ in range(50):
        grid.append(replace(model_params, threshold=torch.rand(1)))

    return grid


@pytest.fixture
def group_em_estimator(dataset: BerpDataset, trf: TemporalReceptiveField,
                       model_param_grid: List[PartiallyObservedModelParameters]):
    pipeline = GroupBerpTRFForwardPipeline(trf, model_param_grid)
    ret = BerpTRFEMEstimator(pipeline)

    # Make sure we run at least once with the dataset so that the pipeline is primed.
    dataset.name = "DKZ_1/derp"
    nested = NestedBerpDataset([dataset])
    ret.partial_fit(nested)

    return ret


def test_param_weights_distribute(group_em_estimator: BerpTRFEMEstimator):
    """
    When EM estimator responsibilities are updated, the constituent pipeline
    parameter weights should automatically update.
    """

    def _assert_param_weights_consistent(group_em_estimator: BerpTRFEMEstimator):
        torch.testing.assert_close(
            group_em_estimator.param_resp_,
            group_em_estimator.pipeline.param_weights
        )

        for pipe in group_em_estimator.pipeline.pipelines_.values():
            torch.testing.assert_close(
                pipe.param_weights,
                group_em_estimator.pipeline.param_weights
            )

    _assert_param_weights_consistent(group_em_estimator)

    group_em_estimator.param_resp_ = torch.rand_like(group_em_estimator.param_resp_)
    group_em_estimator.param_resp_ /= group_em_estimator.param_resp_.sum()

    _assert_param_weights_consistent(group_em_estimator)