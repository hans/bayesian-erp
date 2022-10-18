from copy import deepcopy
from dataclasses import replace
from typing import List, Tuple

import numpy as np
import pytest
import torch

from berp.cv import EarlyStopException
from berp.datasets import BerpDataset, NestedBerpDataset
from berp.generators import thresholded_recognition_simple as generator
from berp.generators.stimulus import RandomStimulusGenerator
from berp.models.reindexing_regression import PartiallyObservedModelParameters, ModelParameters
from berp.models.trf import TemporalReceptiveField
from berp.models.trf_em import BerpTRFEMEstimator, GroupBerpTRFForwardPipeline, GroupVanillaTRFForwardPipeline
from berp.solvers import Solver, AdamSolver, SGDSolver
from berp.util import time_to_sample


@pytest.fixture(scope="session")
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

def make_dataset(synth_params) -> BerpDataset:
    stim = RandomStimulusGenerator(num_words=1000, num_phonemes=10, phoneme_voc_size=synth_params.confusion.shape[0],
                                   word_surprisal_params=(2.0, 0.5))
    ds_args = dict(
        response_type="gaussian",
        epoch_window=(0, 0.55), # TODO unused
        include_intercept=False,
        sample_rate=48)


    dataset = generator.sample_dataset(synth_params, stim, **ds_args)

    # TODO test dataset
    return dataset


@pytest.fixture(scope="session")
def dataset(synth_params):
    return make_dataset(synth_params)


@pytest.fixture(scope="session")
def vanilla_dataset(dataset):
    dataset = deepcopy(dataset)
    tmin, tmax, sfreq = 0, 2, 1

    # three events which have an effect on Y at 1 time sample delay
    X = torch.tensor([[1],
                      [0],
                      [1],
                      [0]]).float()
    Y = torch.tensor([[0, 0],
                      [1, -1],
                      [0, 0],
                      [1, -1]]).float()

    # HACK: drop in and update dataset values
    # May need to update as interface changes
    dataset.name = 'DKZ_1/subj1'
    dataset.sample_rate = sfreq
    dataset.X_ts = X
    dataset.word_onsets = torch.tensor([0, 1, 2, 3]).float()
    dataset.X_variable = torch.zeros(len(dataset.word_onsets), 0).float()  # torch.tensor([0, 0, 0, 0]).float().unsqueeze(1)
    dataset.Y = Y

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


# NB some tests modify the underlying dataset, so scope this fixture at the function level
@pytest.fixture(scope="function")
def group_em_estimator(synth_params: ModelParameters, trf: TemporalReceptiveField,
                       model_param_grid: List[PartiallyObservedModelParameters]):
    pipeline = GroupBerpTRFForwardPipeline(trf, params=model_param_grid)
    ret = BerpTRFEMEstimator(pipeline)

    # Prime the pipeline with two datasets.
    ds1 = make_dataset(synth_params)
    ds1.name = "DKZ_1/subj1"
    ds2 = make_dataset(synth_params)
    ds2.name = "DKZ_1/subj2"
    nested = NestedBerpDataset([ds1, ds2])
    ret.prime(nested)

    return ret, nested


def test_param_weights_distribute(group_em_estimator):
    """
    When EM estimator responsibilities are updated, the pipeline
    parameter weights should automatically update.
    """

    def _assert_param_weights_consistent(est: BerpTRFEMEstimator):
        torch.testing.assert_close(
            est.param_resp_,
            est.pipeline.param_weights
        )

    est, dataset = group_em_estimator
    _assert_param_weights_consistent(est)

    est.param_resp_ = torch.rand_like(est.param_resp_)
    est.param_resp_ /= est.param_resp_.sum()

    _assert_param_weights_consistent(est)


def test_alpha_scatter(group_em_estimator):
    est: BerpTRFEMEstimator
    est, _ = group_em_estimator
    if len(est.pipeline.encoders_) < 2:
        pytest.skip("cannot check with fewer than 2 pipelines")

    def assert_alphas_consistent(est: BerpTRFEMEstimator):
        encoders = list(est.pipeline.encoders_.values())
        alpha = encoders[0].alpha

        for encoder in encoders[1:]:
            torch.testing.assert_close(alpha, encoder.alpha)

        return alpha

    assert_alphas_consistent(est)

    alpha = np.random.random()
    est.pipeline.set_params(encoder__alpha=alpha)
    ret = assert_alphas_consistent(est)
    np.testing.assert_approx_equal(ret, alpha)

    alpha = np.random.random()
    est.set_params(pipeline__encoder__alpha=alpha)
    ret = assert_alphas_consistent(est)
    np.testing.assert_approx_equal(ret, alpha)


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

    cache = est.pipeline._get_cache_for_dataset(ds)

    recognition_points = torch.zeros_like(ds.word_onsets).long()
    recognition_points[-1] = max_word_length - 1

    recognition_times = torch.gather(ds.phoneme_onsets_global, 1, recognition_points.unsqueeze(1)).squeeze(1)

    # Should not raise IndexError.
    out = cache.design_matrix.clone()
    est.pipeline._scatter_variable(ds, recognition_times, out)

    # TODO check scatter result


def test_vanilla_pipeline(vanilla_dataset: BerpDataset):
    nested = NestedBerpDataset([vanilla_dataset])
    tmin, tmax = 0, 2

    trf = TemporalReceptiveField(
        tmin, tmax, nested.sample_rate,
        alpha=0, n_outputs=nested.n_sensors)
    trf_pipe = GroupVanillaTRFForwardPipeline(encoder=trf)
    trf_pipe.prime(nested)
    trf_pipe.fit(nested)

    expected_coef = torch.tensor([[0, 0], [1, -1], [0, 0]]).float()
    coef = trf_pipe.encoders_["subj1"].coef_

    torch.testing.assert_close(coef[0], expected_coef)
    # assert torch.allclose(coef[1], torch.tensor(0.))


def test_vanilla_pipeline_partial(vanilla_dataset: BerpDataset):
    nested = NestedBerpDataset([vanilla_dataset])
    tmin, tmax = 0, 2

    trf = TemporalReceptiveField(
        tmin, tmax, nested.sample_rate,
        alpha=0, n_outputs=nested.n_sensors,
        optim=SGDSolver(learning_rate=0.5, n_batches=64, early_stopping=None))
    trf_pipe = GroupVanillaTRFForwardPipeline(encoder=trf)
    trf_pipe.prime(nested)
    trf_pipe.partial_fit(nested)

    expected_coef = torch.tensor([[0, 0], [1, -1], [0, 0]]).float()
    coef = trf_pipe.encoders_["subj1"].coef_

    torch.testing.assert_close(coef[0], expected_coef, atol=1e-3, rtol=1e-3)
    # assert torch.allclose(coef[1], torch.tensor(0.))


def test_vanilla_pipeline_twosubjs(vanilla_dataset: BerpDataset):
    # Clone dataset and change expected coefs for one subject.
    ds1 = deepcopy(vanilla_dataset)
    ds1.name = "DKZ_1/subj1"

    ds2 = deepcopy(vanilla_dataset)
    ds2.name = "DKZ_1/subj2"
    ds2.Y *= -1  # ******

    nested = NestedBerpDataset([ds1, ds2])
    tmin, tmax = 0, 2

    trf = TemporalReceptiveField(
        tmin, tmax, nested.sample_rate,
        alpha=0, n_outputs=nested.n_sensors)
    trf_pipe = GroupVanillaTRFForwardPipeline(encoder=trf)
    trf_pipe.prime(nested)
    trf_pipe.fit(nested)

    expected_coef = torch.tensor([[0, 0], [1, -1], [0, 0]]).float()
    encs = trf_pipe.encoders_

    torch.testing.assert_close(encs["subj1"].coef_[0], expected_coef)
    torch.testing.assert_close(encs["subj2"].coef_[0], -1 * expected_coef)


def test_vanilla_pipeline_partial_twosubjs(vanilla_dataset: BerpDataset):
    # Clone dataset and change expected coefs for one subject.
    ds1 = deepcopy(vanilla_dataset)
    ds1.name = "DKZ_1/subj1"

    ds2 = deepcopy(vanilla_dataset)
    ds2.name = "DKZ_1/subj2"
    ds2.Y *= -1  # ******

    nested = NestedBerpDataset([ds1, ds2])
    tmin, tmax = 0, 2

    trf = TemporalReceptiveField(
        tmin, tmax, nested.sample_rate,
        alpha=0, n_outputs=nested.n_sensors,
        optim=SGDSolver(learning_rate=0.5, n_batches=64, early_stopping=None))
    trf_pipe = GroupVanillaTRFForwardPipeline(encoder=trf)
    trf_pipe.prime(nested)
    trf_pipe.partial_fit(nested)

    expected_coef = torch.tensor([[0, 0], [1, -1], [0, 0]]).float()
    encs = trf_pipe.encoders_

    tols = dict(atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(encs["subj1"].coef_[0], expected_coef, **tols)
    torch.testing.assert_close(encs["subj2"].coef_[0], -1 * expected_coef, **tols)