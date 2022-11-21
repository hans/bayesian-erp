from copy import deepcopy
from dataclasses import replace
import pickle
from typing import List, Tuple

import numpy as np
import pytest
import torch
from torchtyping import TensorType  # type: ignore

from berp.cv import EarlyStopException
from berp.datasets import BerpDataset, NestedBerpDataset
from berp.generators import thresholded_recognition_simple as generator
from berp.generators.stimulus import RandomStimulusGenerator
from berp.models.reindexing_regression import PartiallyObservedModelParameters, ModelParameters
from berp.models.trf import TemporalReceptiveField, TRFDelayer, TRFDesignMatrix
from berp.models.trf_em import BerpTRFEMEstimator, GroupBerpFixedTRFForwardPipeline, GroupBerpTRFForwardPipeline, GroupVanillaTRFForwardPipeline
from berp.models import trf_em
from berp.solvers import Solver, AdamSolver, SGDSolver
from berp.tensorboard import Tensorboard
from berp.util import time_to_sample



# Avoid saving lots of spurious tensorboard events files
Tensorboard.disable()


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

def make_dataset(synth_params, sample_rate=48) -> BerpDataset:
    """
    Sample a synthetic dataset with random word/phoneme time series,
    where all events (phoneme onset/offset, word onset/offset) are
    aligned to the sample rate.
    """
    stim = RandomStimulusGenerator(num_words=1000, num_phonemes=10, phoneme_voc_size=synth_params.confusion.shape[0],
                                   word_surprisal_params=(2.0, 0.5))
    ds_args = dict(
        response_type="gaussian",
        epoch_window=(0, 0.55), # TODO unused
        include_intercept=False,
        sample_rate=sample_rate)

    dataset = generator.sample_dataset(synth_params, stim, **ds_args,
        stimulus_kwargs=dict(align_sample_rate=sample_rate))

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
    # NB this dataset is in an inconsistent state and will likely break Berp!
    # But it's fine for the vanilla model.
    dataset.name = 'DKZ_1/subj1'
    dataset.sample_rate = sfreq
    dataset.X_ts = X
    dataset.ts_feature_names = [f"ts_{i}" for i in range(X.shape[1])]
    dataset.word_onsets = torch.tensor([0, 1, 2, 3]).float()
    dataset.word_offsets = torch.tensor([1, 2, 3, 4]).float()
    dataset.phoneme_onsets = torch.tensor([0, 1, 2, 3]).float().unsqueeze(1)
    # dataset.X_variable = torch.randn(len(dataset.word_onsets), 2).float()  # torch.tensor([0, 0, 0, 0]).float().unsqueeze(1)
    dataset.X_variable = torch.zeros(len(dataset.word_onsets), 0).float()
    dataset.variable_feature_names = []
    dataset.Y = Y

    dataset.check_shapes()

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
def group_em_estimator(synth_params: ModelParameters, dataset: BerpDataset,
                       trf: TemporalReceptiveField,
                       model_param_grid: List[ModelParameters]):
    pipeline = GroupBerpTRFForwardPipeline(trf, params=model_param_grid,
        ts_feature_names=dataset.ts_feature_names,
        variable_feature_names=dataset.variable_feature_names)
    ret = BerpTRFEMEstimator(pipeline)

    # Prime the pipeline with two datasets.
    ds1 = make_dataset(synth_params)
    ds1.name = "DKZ_1/subj1"
    ds2 = make_dataset(synth_params)
    ds2.name = "DKZ_1/subj2"
    nested = NestedBerpDataset([ds1, ds2])
    ret.prime(nested)

    return ret, nested


@pytest.fixture(scope="function")
def group_fixed_estimator(synth_params: ModelParameters, dataset: BerpDataset,
                          trf: TemporalReceptiveField):
    pipe = GroupBerpFixedTRFForwardPipeline(
        trf,
        ts_feature_names=dataset.ts_feature_names,
        variable_feature_names=dataset.variable_feature_names,
        threshold=synth_params.threshold,
        confusion=synth_params.confusion,
        lambda_=synth_params.lambda_,
        scatter_point=np.random.random(),
        prior_scatter_index=np.random.choice([-3, -2, -1, 0]),
        prior_scatter_point=np.random.random(),
    )

    return pipe


@pytest.fixture(scope="function")
def group_cannon_estimator(synth_params: ModelParameters, dataset: BerpDataset,
                           trf: TemporalReceptiveField):
    pipe = trf_em.GroupBerpCannonTRFForwardPipeline(
        trf,
        ts_feature_names=dataset.ts_feature_names,
        variable_feature_names=dataset.variable_feature_names,
        threshold=synth_params.threshold,
        confusion=synth_params.confusion,
        lambda_=synth_params.lambda_,
        n_quantiles=3,
        scatter_point=np.random.random(),
        prior_scatter_index=np.random.choice([-3, -2, -1, 0]).item(),
        prior_scatter_point=np.random.random(),
    )

    return pipe


def test_variable_trf_zero_overflow(trf: TemporalReceptiveField, dataset: BerpDataset):
    # Should error when variable-onset TRF zero-constraint is too wide
    # given the width of the TRF encoder window.
    too_many_samples = int((trf.tmax - trf.tmin) * trf.sfreq) + 128
    left_zero = too_many_samples // 2
    right_zero = too_many_samples // 2

    with pytest.raises(ValueError):
        GroupBerpTRFForwardPipeline(
            trf,
            ts_feature_names=dataset.ts_feature_names,
            variable_feature_names=dataset.variable_feature_names,
            params=[PartiallyObservedModelParameters()],
            variable_trf_zero_left=left_zero,
            variable_trf_zero_right=right_zero
        )


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


@pytest.mark.usefixtures("group_fixed_estimator")
class TestGroupBerpFixed:

    check_params = [
        "threshold", "lambda_", "confusion",
        "scatter_point", "prior_scatter_point", "prior_scatter_index",
        "variable_trf_zero_left", "variable_trf_zero_right",
    ]

    def _eq(self, a, b):
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            torch.testing.assert_close(a, b)
        else:
            assert a == b

    @pytest.mark.parametrize("param", check_params)
    def test_parameters_distribute(self, group_fixed_estimator: GroupBerpFixedTRFForwardPipeline, param: str):
        """
        Parameter reads/writes should be synchronized
        """

        params = group_fixed_estimator.get_params()
        self._eq(getattr(group_fixed_estimator, param), params[param])
        if hasattr(group_fixed_estimator.params[0], param):
            self._eq(getattr(group_fixed_estimator.params[0], param), params[param])

        new_val = torch.tensor(np.random.random())
        setattr(group_fixed_estimator, param, new_val)
        self._eq(getattr(group_fixed_estimator, param), new_val)
        if hasattr(group_fixed_estimator.params[0], param):
            self._eq(getattr(group_fixed_estimator.params[0], param), new_val)

        new_val = torch.tensor(np.random.random())
        group_fixed_estimator.set_params(**{param: new_val})
        self._eq(getattr(group_fixed_estimator, param), new_val)
        if hasattr(group_fixed_estimator.params[0], param):
            self._eq(getattr(group_fixed_estimator.params[0], param), new_val)

    def test_pickle(self, group_fixed_estimator: GroupBerpFixedTRFForwardPipeline):
        pickled = pickle.dumps(group_fixed_estimator)
        unpickled = pickle.loads(pickled)

        for param in self.check_params:
            self._eq(getattr(group_fixed_estimator, param), getattr(unpickled, param))
            self._eq(getattr(group_fixed_estimator, param), unpickled.get_params()[param])

    @pytest.mark.usefixtures("dataset")
    def test_prior_recognition(self, group_fixed_estimator: GroupBerpFixedTRFForwardPipeline,
                               dataset: BerpDataset):
        group_fixed_estimator.prior_scatter_index = -1
        group_fixed_estimator.prior_scatter_point = 0.0

        # Hope we get some recognitions at prior :)
        group_fixed_estimator.threshold = torch.tensor(0.05)

        recognition_points, recognition_times = \
            group_fixed_estimator.get_recognition_times(dataset, group_fixed_estimator.params[0])
        
        recognized_at_prior = recognition_points == 0
        if not recognized_at_prior.any():
            pytest.skip("No recognitions at prior")

        recognition_times_relative = recognition_times - dataset.word_onsets
        recognition_times_relative = recognition_times_relative[recognized_at_prior]
        assert (recognition_times_relative < 0).all(), \
            "Words recognized at prior should have recognition times before word onset"


@pytest.mark.usefixtures("group_cannon_estimator")
class TestGroupCannon:

    def test_recognition_quantiles(self,
                                   group_cannon_estimator: trf_em.GroupBerpCannonTRFForwardPipeline,
                                   dataset: BerpDataset):
        # TODO
        pass

    def test_pre_transform(self,
                           group_cannon_estimator: trf_em.GroupBerpCannonTRFForwardPipeline,
                           dataset: BerpDataset):
        """
        Test that the Cannon pipeline works
        """
        params = group_cannon_estimator.params[0]
        quantiles = group_cannon_estimator._get_recognition_quantiles(dataset, params)

        assert quantiles.min() >= 0
        assert quantiles.max() < group_cannon_estimator.n_quantiles

        ######

        dataset.name = "DKZ_1/subj1"
        group_cannon_estimator.prime(dataset)
        design_matrix, _ = group_cannon_estimator.pre_transform(dataset)
        assert design_matrix.shape[1] == dataset.n_ts_features + (1 + group_cannon_estimator.n_quantiles) * dataset.n_variable_features
        for quantile_i in range(group_cannon_estimator.n_quantiles):
            print(quantile_i)
            mask = quantiles == quantile_i
            onsets_i = dataset.word_onsets[mask]

            expected_features = torch.zeros((len(onsets_i), (1 + group_cannon_estimator.n_quantiles) * dataset.n_variable_features))
            expected_features[:, :dataset.n_variable_features] = dataset.X_variable[mask]
            expected_features[:, (quantile_i + 1) * dataset.n_variable_features:(quantile_i + 2) * dataset.n_variable_features] = \
                dataset.X_variable[mask]

            check_lagged_features(
                design_matrix[:, dataset.n_ts_features:, :],
                time_to_sample(onsets_i, dataset.sample_rate),
                expected_features
            )



class TestGroupVanilla:

    def _make_trf_pipe(self, nested: NestedBerpDataset, **kwargs):
        kwargs = dict(tmin=0, tmax=2, sfreq=nested.sample_rate,
                      alpha=0, n_outputs=nested.n_sensors) | kwargs
        trf = TemporalReceptiveField(**kwargs)
        trf_pipe = GroupVanillaTRFForwardPipeline(encoder=trf, ts_feature_names=nested.ts_feature_names,
                                                  variable_feature_names=nested.variable_feature_names)
        trf_pipe.prime(nested)
        return trf_pipe

    def test_vanilla_pipeline(self, vanilla_dataset: BerpDataset):
        nested = NestedBerpDataset([vanilla_dataset])
        trf_pipe = self._make_trf_pipe(nested)
        trf_pipe.fit(nested)

        expected_coef = torch.tensor([[0, 0], [1, -1], [0, 0]]).float()
        coef = trf_pipe.encoders_["subj1"].coef_

        torch.testing.assert_close(coef[0], expected_coef)
        # assert torch.allclose(coef[1], torch.tensor(0.))

    def test_vanilla_pipeline_partial(self, vanilla_dataset: BerpDataset):
        nested = NestedBerpDataset([vanilla_dataset])
        trf_pipe = self._make_trf_pipe(nested, 
            optim=SGDSolver(learning_rate=0.5, n_batches=64, early_stopping=None))
        trf_pipe.partial_fit(nested)

        expected_coef = torch.tensor([[0, 0], [1, -1], [0, 0]]).float()
        coef = trf_pipe.encoders_["subj1"].coef_

        torch.testing.assert_close(coef[0], expected_coef, atol=1e-3, rtol=1e-3)
        # assert torch.allclose(coef[1], torch.tensor(0.))

    def test_vanilla_pipeline_twosubjs(self, vanilla_dataset: BerpDataset):
        # Clone dataset and change expected coefs for one subject.
        ds1 = deepcopy(vanilla_dataset)
        ds1.name = "DKZ_1/subj1"

        ds2 = deepcopy(vanilla_dataset)
        ds2.name = "DKZ_1/subj2"
        ds2.Y *= -1  # ******

        nested = NestedBerpDataset([ds1, ds2])
        trf_pipe = self._make_trf_pipe(nested)
        trf_pipe.fit(nested)

        expected_coef = torch.tensor([[0, 0], [1, -1], [0, 0]]).float()
        encs = trf_pipe.encoders_

        torch.testing.assert_close(encs["subj1"].coef_[0], expected_coef)
        torch.testing.assert_close(encs["subj2"].coef_[0], -1 * expected_coef)

    def test_vanilla_pipeline_partial_twosubjs(self, vanilla_dataset: BerpDataset):
        # Clone dataset and change expected coefs for one subject.
        ds1 = deepcopy(vanilla_dataset)
        ds1.name = "DKZ_1/subj1"

        ds2 = deepcopy(vanilla_dataset)
        ds2.name = "DKZ_1/subj2"
        ds2.Y *= -1  # ******

        nested = NestedBerpDataset([ds1, ds2])
        trf_pipe = self._make_trf_pipe(nested,
            optim=SGDSolver(learning_rate=0.5, n_batches=64, early_stopping=None))
        trf_pipe.partial_fit(nested)

        expected_coef = torch.tensor([[0, 0], [1, -1], [0, 0]]).float()
        encs = trf_pipe.encoders_

        tols = dict(atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(encs["subj1"].coef_[0], expected_coef, **tols)
        torch.testing.assert_close(encs["subj2"].coef_[0], -1 * expected_coef, **tols)

    def test_scores(self, vanilla_dataset: BerpDataset):
        # Clone dataset and change expected coefs for one subject.
        ds1 = deepcopy(vanilla_dataset)
        ds1.name = "DKZ_1/subj1"

        ds2 = deepcopy(vanilla_dataset)
        ds2.name = "DKZ_1/subj2"
        ds2.Y *= -1  # ******
        ds2.Y += torch.randn_like(ds2.Y)

        nested = NestedBerpDataset([ds1, ds2])
        trf_pipe = self._make_trf_pipe(nested)
        trf_pipe.fit(nested)

        aggregate_score = trf_pipe.score(nested)
        assert isinstance(aggregate_score, float)

        scores = trf_pipe.score_multidimensional(nested)
        assert scores.shape == (len(nested.datasets), vanilla_dataset.n_sensors)

        np.testing.assert_allclose(scores.mean(), aggregate_score)


def check_lagged_features(X: TRFDesignMatrix, samples: TensorType["batch", torch.long],
                          values: TensorType["batch", torch.float]):
    """
    Check that there is accurate representation of lagged features with onset
    at `samples` and corresponding `values` in the given TRF design matrix.
    """

    for delay in range(X.shape[2]):
        samples_i = samples + delay

        # don't index past time series edge.
        overflow_mask = samples_i >= X.shape[0]

        torch.testing.assert_allclose(
            X[samples_i[~overflow_mask], :, delay],
            values[~overflow_mask],
        )


@pytest.mark.parametrize("epoch_window", [(0, 0.5)])
def test_scatter_variable(dataset: BerpDataset, epoch_window: Tuple[float, float]):
    tmin, tmax = epoch_window
    delayer = TRFDelayer(tmin, tmax, dataset.sample_rate)
    # Prepare dummy design matrix.
    design_matrix = trf_em.make_dummy_design_matrix(dataset, delayer)

    # Events should be one sample after each word onset.
    shift = 1
    times = dataset.word_onsets + shift / dataset.sample_rate

    # NB assumes word onsets are nicely aligned to sample rate,
    # which is true for our synthetic data.
    from pprint import pprint
    torch.testing.assert_allclose(
        time_to_sample(times, dataset.sample_rate) - time_to_sample(dataset.word_onsets, dataset.sample_rate),
        shift * torch.ones(len(times), dtype=torch.long),
        msg="Word onsets not aligned to sample rate. Test precondition failed.")
    assert shift / dataset.sample_rate < tmax, "Test precondition failed."

    trf_em.scatter_variable(dataset, times, design_matrix, 1.0)
    variable_feature_start_idx = dataset.n_ts_features

    check_lagged_features(
        design_matrix[:, variable_feature_start_idx:, :],
        time_to_sample(dataset.word_onsets, dataset.sample_rate) + shift,
        dataset.X_variable)


def test_scatter_add_edges(dataset: BerpDataset):
    # epoch window
    tmin, tmax = 0, 0.5
    delayer = TRFDelayer(tmin, tmax, dataset.sample_rate)
    # Prepare dummy design matrix.
    design_matrix = trf_em.make_dummy_design_matrix(dataset, delayer)
    design_matrix_old = design_matrix.clone()

    # Scatter-add at an edge
    target_samples = torch.tensor([design_matrix.shape[0] - 5])

    # Expect scatter to work (scatter-add) for 5 samples, and then work
    # not trigger indexing error for the rest.
    trf_em.scatter_add(design_matrix[:, 1:, :], target_samples, torch.tensor([[1.3]]))

    from pprint import pprint
    torch.testing.assert_allclose(
        design_matrix[torch.arange(design_matrix.shape[0] - 5, design_matrix.shape[0]),
                      1, torch.arange(5)],
        torch.tensor([1.3] * 5))

    expected = torch.zeros_like(design_matrix)
    expected[torch.arange(design_matrix.shape[0] - 5, design_matrix.shape[0]),
             1, torch.arange(5)] = 1.3
    torch.testing.assert_allclose(design_matrix, design_matrix_old + expected)


@pytest.mark.parametrize("epoch_window", [(0, 0.5)])
def test_scatter_variable_with_mask(dataset: BerpDataset, epoch_window: Tuple[float, float]):
    tmin, tmax = epoch_window
    delayer = TRFDelayer(tmin, tmax, dataset.sample_rate)
    # Prepare dummy design matrix.
    design_matrix = trf_em.make_dummy_design_matrix(dataset, delayer)

    # Events should be one sample after each word onset.
    shift = 1
    times = dataset.word_onsets + shift / dataset.sample_rate

    # NB assumes word onsets are nicely aligned to sample rate,
    # which is true for our synthetic data.
    from pprint import pprint
    torch.testing.assert_allclose(
        time_to_sample(times, dataset.sample_rate) - time_to_sample(dataset.word_onsets, dataset.sample_rate),
        shift * torch.ones(len(times), dtype=torch.long),
        msg="Word onsets not aligned to sample rate. Test precondition failed.")
    assert shift / dataset.sample_rate < tmax, "Test precondition failed."

    lag_mask = torch.randint(0, 2, (design_matrix.shape[2],), dtype=torch.bool)
    trf_em.scatter_variable(dataset, times, design_matrix, 1.0, lag_mask=lag_mask)
    variable_feature_start_idx = dataset.n_ts_features
    for delay in range(design_matrix.shape[2]):
        expected = dataset.X_variable if lag_mask[delay] else torch.zeros_like(dataset.X_variable)

        samples = time_to_sample(dataset.word_onsets, dataset.sample_rate) + shift + delay
        # Don't index past time series edge.
        overflow_mask = samples >= design_matrix.shape[0]

        torch.testing.assert_allclose(
            torch.gather(
                design_matrix[:, variable_feature_start_idx:, delay],
                0,
                samples[~overflow_mask].unsqueeze(1),
            ),
            expected[~overflow_mask]
        )