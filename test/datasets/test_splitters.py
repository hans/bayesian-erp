from typing import List

import numpy as np
import pytest
import torch

from berp.datasets import BerpDataset, NestedBerpDataset
from berp.datasets import splitters
from berp.generators import thresholded_recognition_simple as generator
from berp.generators.stimulus import RandomStimulusGenerator
from berp.models.reindexing_regression import ModelParameters
from berp.tensorboard import Tensorboard


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

def make_datasets(synth_params, n=1, sample_rate=48) -> List[BerpDataset]:
    """
    Sample N synthetic datasets with random word/phoneme time series,
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

    stim = stim(align_sample_rate=sample_rate)
    stim_thunk = lambda: stim

    datasets = [
        generator.sample_dataset(synth_params, stim_thunk, **ds_args)
        for _ in range(n)
    ]
    return datasets


def test_kfold(synth_params):
    datasets = make_datasets(synth_params, n=6)
    nested = NestedBerpDataset(datasets, n_splits=4)

    # TODO account for case where datasets are of different lengths
    # TODO test that returned kfold has maximally contiguous

    def summarize_dataset(ds: NestedBerpDataset):
        return [(x.name.split("/")[0], x.global_slice_indices) for x in ds]

    dataset_max_len = max(len(x) for x in datasets)
    dataset_name2idx = {x.name: idx for idx, x in enumerate(datasets)}

    test_fold_assignments = np.zeros((nested.n_datasets, dataset_max_len), dtype=np.int32)

    for fold_i, (train, test) in enumerate(splitters.kfold(nested, n_splits=4)):
        train_assignment = np.zeros((nested.n_datasets, dataset_max_len), dtype=bool)

        train_draws = summarize_dataset(train)
        test_draws = summarize_dataset(test)
        print(f"Train {fold_i}: {repr(train)}")
        print(f"Test {fold_i}: {repr(test)}")

        for ds in train:
            source_dataset = ds.name.split("/")[0]
            start, end = ds.global_slice_indices
            train_assignment[dataset_name2idx[source_dataset], start:end] = True

        for ds in test:
            source_dataset = ds.name.split("/")[0]
            start, end = ds.global_slice_indices
            assert not np.any(train_assignment[dataset_name2idx[source_dataset], start:end]), \
                "Test data overlaps with train data"

            assert np.all(test_fold_assignments[dataset_name2idx[source_dataset], start:end] == 0), \
                    f"Test fold {fold_i} overlaps with prior test fold"
            test_fold_assignments[dataset_name2idx[source_dataset], start:end] = fold_i + 1

    np.testing.assert_array_less(0, test_fold_assignments,
                                 err_msg="Some samples were not assigned to a test fold")


def test_recursive_kfold(synth_params):
    """
    Test recursive k-fold splitting (where the training fold is used to
    again generate k-fold splits).
    """

    datasets = make_datasets(synth_params, n=6)
    nested = NestedBerpDataset(datasets, n_splits=4)

    dataset_max_len = max(len(x) for x in datasets)
    dataset_name2idx = {x.name: idx for idx, x in enumerate(datasets)}

    for fold_i, (train, test) in enumerate(splitters.kfold(nested, n_splits=4)):
        outer_test_assignment = np.zeros((nested.n_datasets, dataset_max_len), dtype=bool)
        # Ensure that each sample appears in exactly one test fold.
        inner_test_assignment = np.zeros((nested.n_datasets, dataset_max_len), dtype=int)

        for ds in test:
            source_dataset = ds.name.split("/")[0]
            start, end = ds.global_slice_indices
            outer_test_assignment[dataset_name2idx[source_dataset], start:end] = True

        for fold_ij, (train2, test2) in enumerate(splitters.kfold(train, n_splits=4)):
            train_assignment = np.zeros((nested.n_datasets, dataset_max_len), dtype=bool)

            print(f"Test fold {fold_i}/{fold_ij}: {repr(test2)}")

            for ds in train2:
                source_dataset = ds.name.split("/")[0]
                start, end = ds.global_slice_indices
                train_assignment[dataset_name2idx[source_dataset], start:end] = True

                assert not np.any(outer_test_assignment[dataset_name2idx[source_dataset], start:end]), \
                    "Outer fold test data overlaps with train data"

            for ds in test2:
                source_dataset = ds.name.split("/")[0]
                start, end = ds.global_slice_indices

                assert np.all(inner_test_assignment[dataset_name2idx[source_dataset], start:end] == 0), \
                    f"Test fold {fold_ij} overlaps with prior test fold"
                inner_test_assignment[dataset_name2idx[source_dataset], start:end] = fold_ij + 1

                assert not np.any(train_assignment[dataset_name2idx[source_dataset], start:end]), \
                    "Test data overlaps with train data"

                assert not np.any(outer_test_assignment[dataset_name2idx[source_dataset], start:end]), \
                    "Outer fold test data overlaps with inner fold test data"

        np.testing.assert_array_equal(0, inner_test_assignment[outer_test_assignment],
            err_msg="No samples in the outer test fold should appear in an inner test fold")
        np.testing.assert_array_less(0, inner_test_assignment[~outer_test_assignment],
            err_msg="All samples not in the outer test fold should appear in an inner test fold")


def test_train_test_split(synth_params):
    datasets = make_datasets(synth_params, n=6)
    nested = NestedBerpDataset(datasets, n_splits=4)

    dataset_max_len = max(len(x) for x in datasets)
    dataset_name2idx = {x.name: idx for idx, x in enumerate(datasets)}

    sample_assignments = np.zeros((nested.n_datasets, dataset_max_len), dtype=np.int32)

    train, test = splitters.train_test_split(nested, test_size=0.25)

    for ds in train:
        source_dataset = ds.name.split("/")[0]
        start, end = ds.global_slice_indices
        sample_assignments[dataset_name2idx[source_dataset], start:end] = 1
    
    for ds in test:
        source_dataset = ds.name.split("/")[0]
        start, end = ds.global_slice_indices

        np.testing.assert_array_equal(0, sample_assignments[dataset_name2idx[source_dataset], start:end],
            err_msg="Test data overlaps with train data")

        sample_assignments[dataset_name2idx[source_dataset], start:end] = 2

    np.testing.assert_array_less(0, sample_assignments,
        err_msg="Some samples were not assigned to a test or train fold")