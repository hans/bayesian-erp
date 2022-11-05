import pytest
import torch

from berp.datasets import BerpDataset, NestedBerpDataset
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

def make_datasets(synth_params, n=1, sample_rate=48) -> BerpDataset:
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

    stim = stim()
    stim_thunk = lambda: stim

    datasets = [
        generator.sample_dataset(
            synth_params, stim_thunk, **ds_args,
            stimulus_kwargs=dict(align_sample_rate=sample_rate))
        for _ in range(n)
    ]
    return datasets


def test_nested_kfold(synth_params):
    datasets = make_datasets(synth_params, n=6)
    nested = NestedBerpDataset(datasets, n_splits=4)

    # TODO track what is getting returned in each KFold/GroupKFold draw by default