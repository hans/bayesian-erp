from argparse import ArgumentParser
import re
from typing import List, Tuple, NamedTuple, Callable

from typeguard import typechecked

import torch
from torch.nn.functional import pad
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import pyro.poutine as poutine

from berp.generators import thresholded_recognition_simple as generator
from berp.models import reindexing_regression as rr
from berp.typing import is_log_probability, DIMS


def get_parameters():
    # Sample model parameters.
    coef_mean = torch.tensor([1., -1.])
    return rr.ModelParameters(
        lambda_=torch.tensor(1.0),
        confusion=generator.phoneme_confusion,
        threshold=pyro.sample("threshold",
                              dist.Beta(1.2, 1.2)),
        a=torch.tensor(0.4),
        b=torch.tensor(0.1),
        coef=pyro.sample("coef", dist.Normal(coef_mean, 0.1)),
        sigma=torch.tensor(1.0),
    )


def fit(dataset: rr.RRDataset):
    nuts_kernel = NUTS(rr.model)
    mcmc = MCMC(nuts_kernel,
                num_samples=400,
                warmup_steps=100,
                num_chains=4)

    mcmc.run(dataset.params,
        p_word=dataset.p_word,
        candidate_phonemes=dataset.candidate_phonemes,
        phoneme_onsets=dataset.phoneme_onsets,
        word_lengths=dataset.word_lengths,

        X_epoched=dataset.X_epoch,
        Y_epoched=dataset.Y_epoch,

        sample_rate=dataset.sample_rate,
        epoch_window=dataset.epoch_window)

    mcmc.summary(prob=0.8)


def main(args):
    epoch_window = (-0.1, 1.0)

    dataset = generator.sample_dataset(params=get_parameters(),
                                       num_words=150,
                                       epoch_window=epoch_window)

    if args.mode == "fit":
        fit(dataset)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("-m", "--mode", choices=["fit"],
                   default="fit")

    main(p.parse_args())
