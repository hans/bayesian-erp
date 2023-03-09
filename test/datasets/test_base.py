from copy import deepcopy
import uuid

import numpy as np
import pytest
from sklearn.model_selection import KFold
import torch

from berp.datasets.base import BerpDataset, NestedBerpDataset
from berp.datasets.base import assert_concatenatable, assert_compatible
from berp.generators.stimulus import RandomStimulusGenerator


def make_dataset():
    stim_gen = RandomStimulusGenerator()
    stim = stim_gen()

    t_max = stim.phoneme_onsets_global[-1, -1]
    sfreq = 48
    sigma = 1.
    num_sensors = 3
    sensor_names = ["a", "b", "c"]
    num_ts_features = 3
    num_variable_features = 3

    Y = torch.normal(0, sigma, size=(int(np.ceil(t_max * sfreq)), num_sensors))
    X_ts = Y * 0.01 + torch.normal(0, 1, size=(Y.shape[0], num_ts_features))
    X_variable = torch.normal(0, 1, size=(stim.word_lengths.shape[0], num_variable_features))

    return BerpDataset(
        name=uuid.uuid4().hex,
        stimulus_name=uuid.uuid4().hex,
        sample_rate=sfreq,
        phonemes=[chr(i) for i in range(stim_gen.num_phonemes)],

        p_candidates=stim.p_candidates,
        word_lengths=stim.word_lengths,
        candidate_phonemes=stim.candidate_phonemes,
        word_onsets=stim.word_onsets,
        word_offsets=stim.word_offsets,
        phoneme_onsets=stim.phoneme_onsets,

        X_ts=X_ts,
        X_variable=X_variable,
        Y=Y,
        sensor_names=sensor_names,
    )


def test_offsets():
    ds = make_dataset()
    assert ds.phoneme_offsets_global.shape == ds.phoneme_onsets_global.shape

    assert torch.allclose(ds.phoneme_offsets_global[:, -1], ds.word_offsets)
    assert (ds.word_offsets[:-1] <= ds.word_onsets[1:]).all(), "No overlapping words"


def test_slice_name():
    ds = make_dataset()
    sliced = ds[0:10]
    assert sliced.name.startswith(f"{ds.name}/slice:")


def test_slice_indices():
    ds = make_dataset()
    assert ds.global_slice_indices is None

    sliced1 = ds[39:96]
    assert sliced1.global_slice_indices == (39, 96)

    sliced2 = sliced1[1:19]
    assert sliced2.global_slice_indices == (39 + 1, 39 + 1 + 18)

    sliced3 = sliced2[9:18]
    assert sliced3.global_slice_indices == (39 + 1 + 9, 39 + 1 + 9 + 9)

    # Try a second slice from sliced1
    sliced2_2 = sliced1[8:10]
    assert sliced2_2.global_slice_indices == (39 + 8, 39 + 8 + (10 - 8))


@pytest.fixture
def single_nested_dataset():
    ds = make_dataset()
    n_splits = 4
    nested = NestedBerpDataset([ds], n_splits=n_splits)
    return nested


@pytest.fixture
def nested_dataset_2():
    dss = [make_dataset(), make_dataset()]
    n_splits = 4
    nested = NestedBerpDataset(dss, n_splits=n_splits)
    return nested


def test_nested_shape(nested_dataset_2):
    assert nested_dataset_2.flat_idxs.shape == (2 * nested_dataset_2.n_splits, 3)


def test_nested_getitem_contiguous(nested_dataset_2):
    """
    When indexing into a nested dataset with contiguous indices, the
    resulting nested dataset should keep items maximally contiguous.
    """
    # Aim to slice all of the first dataset, then one element of the second
    idxs = np.arange(nested_dataset_2.n_splits + 1)
    sliced = nested_dataset_2[idxs]

    print([d.name for d in sliced.datasets])
    assert len(sliced.datasets) == 2, "First dataset should be kept contiguous"
    assert sliced.datasets[0].name == nested_dataset_2.datasets[0].name, "First dataset should be kept contiguous"
    assert sliced.datasets[1].name.startswith(f"{nested_dataset_2.datasets[1].name}/slice:"), "Second dataset should be sliced"


@pytest.mark.parametrize("sensor_spec", [["a", "c"], [0, 2]])
def test_subset_sensors(sensor_spec):
    ds = make_dataset()
    ds2 = deepcopy(ds)
    ds2.subset_sensors(sensor_spec)

    assert ds.Y.shape[0] == ds2.Y.shape[0]
    assert ds.Y.shape[1] == 3
    assert ds2.Y.shape[1] == 2
    assert ds2.sensor_names == ["a", "c"]
    torch.testing.assert_allclose(ds2.Y, ds.Y[:, [0, 2]])


def test_average_sensors():
    ds = make_dataset()

    ds2 = deepcopy(ds)
    ds2.average_sensors()

    assert ds.Y.shape[0] == ds2.Y.shape[0]
    assert ds.Y.shape[1] == 3
    assert ds2.Y.shape[1] == 1
    assert ds2.sensor_names == ["average_a_b_c"]
    torch.testing.assert_allclose(ds2.Y, ds.Y.mean(dim=1, keepdim=True))


def test_average_unnamed_sensors():
    ds = make_dataset()
    ds.sensor_names = None

    ds2 = deepcopy(ds)
    ds2.average_sensors()

    assert ds.Y.shape[0] == ds2.Y.shape[0]
    assert ds.Y.shape[1] == 3
    assert ds2.Y.shape[1] == 1
    assert ds2.sensor_names == ["average"]
    torch.testing.assert_allclose(ds2.Y, ds.Y.mean(dim=1, keepdim=True))


def test_select_features():
    dataset = make_dataset()
    dataset.ts_feature_names = [f"ts{x}" for x in range(dataset.n_ts_features)]
    dataset.X_variable = torch.concat([dataset.X_variable, 2 * dataset.X_variable], dim=1)
    dataset.variable_feature_names = ["var1", "var2"]

    dataset2 = deepcopy(dataset)
    dataset2.select_features(ts=["ts0"], variable=["var2"])
    dataset2_int = deepcopy(dataset)
    dataset2_int.select_features(ts=[0], variable=[1])
    torch.testing.assert_allclose(dataset2.X_ts, dataset.X_ts[:, [0]])
    torch.testing.assert_allclose(dataset2.X_ts, dataset2_int.X_ts)
    torch.testing.assert_allclose(dataset2.X_variable, dataset.X_variable[:, [1]])
    torch.testing.assert_allclose(dataset2.X_variable, dataset2_int.X_variable)


    # Test get_features here too
    ds2_ts, ds2_variable = dataset.get_features(ts=["ts0"], variable=["var2"])
    torch.testing.assert_allclose(ds2_ts, dataset2.X_ts)
    torch.testing.assert_allclose(ds2_variable, dataset2.X_variable)

    ds2_int_ts, ds2_int_variable = dataset.get_features(ts=[0], variable=[1])
    torch.testing.assert_allclose(ds2_int_ts, dataset2.X_ts)
    torch.testing.assert_allclose(ds2_int_variable, dataset2.X_variable)


def test_select_features_drop_all_variable():
    dataset = make_dataset()
    dataset.ts_feature_names = [f"ts{x}" for x in range(dataset.n_ts_features)]
    dataset.X_variable = torch.concat([dataset.X_variable, 2 * dataset.X_variable], dim=1)
    dataset.variable_feature_names = ["var1", "var2"]

    dataset2 = deepcopy(dataset)
    dataset2.select_features(ts=None, variable=[])
    torch.testing.assert_allclose(dataset2.X_ts, dataset.X_ts)
    assert dataset2.X_variable.shape == (dataset.n_words, 0)

    # Test get_features here too
    ds2_ts, ds2_variable = dataset.get_features(ts=None, variable=[])
    torch.testing.assert_allclose(ds2_ts, dataset2.X_ts)
    assert ds2_variable.shape == (dataset.n_words, 0)


def test_nested_different_sensors():
    ds = make_dataset()

    ds2 = make_dataset()
    ds2.Y = ds2.Y[:, 1:]
    ds2.sensor_names = ["c", "d"]

    assert_compatible(ds, ds2)
    with pytest.raises(AssertionError):
        assert_concatenatable(ds, ds2)

    # Shouldn't bork.
    nested = NestedBerpDataset([ds, ds2])