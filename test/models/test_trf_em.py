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
from berp.util import time_to_sample


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
    return AdamSolver(early_stopping=False, n_batches=1)

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
    for _ in range(2):
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

    return ret, nested


def test_param_weights_distribute(group_em_estimator):
    """
    When EM estimator responsibilities are updated, the constituent pipeline
    parameter weights should automatically update.
    """

    def _assert_param_weights_consistent(est: BerpTRFEMEstimator):
        torch.testing.assert_close(
            est.param_resp_,
            est.pipeline.param_weights
        )

        for pipe in est.pipeline.pipelines_.values():
            torch.testing.assert_close(
                pipe.param_weights,
                est.pipeline.param_weights
            )

    est, dataset = group_em_estimator
    _assert_param_weights_consistent(est)

    est.param_resp_ = torch.rand_like(est.param_resp_)
    est.param_resp_ /= est.param_resp_.sum()

    _assert_param_weights_consistent(est)


def test_scatter_variable_edges(group_em_estimator):
    """
    For recognition points on the edge of time series, make sure that no indexing
    errors are triggered.
    """
    est: BerpTRFEMEstimator
    dataset: NestedBerpDataset
    est, dataset = group_em_estimator

    # Set up a long word at the very end of the time series with a recognition point
    # at its last phoneme
    ds = dataset.datasets[0]
    max_word_length = ds.phoneme_onsets.shape[1]
    ds.word_lengths[-1] = max_word_length
    last_time = ds.phoneme_onsets_global[-1, -1]
    # last_sample = time_to_sample(last_time, dataset.sample_rate)
    ds.phoneme_onsets[-1, max_word_length - 1] = last_time - ds.word_onsets[-1] - 0.001
    print(ds.phoneme_onsets[-1])

    # precondition
    assert len(est.pipeline.pipelines_) == 1
    est_sub = next(iter(est.pipeline.pipelines_.values()))

    recognition_points = torch.zeros_like(ds.word_onsets).long()
    recognition_points[-1] = max_word_length - 1

    primed = est_sub._prime(ds)

    # Should not raise IndexError.
    out = primed.design_matrix.clone()
    est_sub._scatter_variable(ds, recognition_points, out)

    # TODO check scatter result