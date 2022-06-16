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
    coef_mean = torch.tensor([0., -1.])
    coef_sigma = torch.tensor([1e-6, 0.1])
    return rr.ModelParameters(
        lambda_=torch.tensor(1.0),
        confusion=generator.phoneme_confusion,
        threshold=pyro.sample("threshold",
                              dist.Beta(1.2, 1.2)),
        a=torch.tensor(0.4),
        b=torch.tensor(0.1),
        coef=pyro.deterministic("coef", coef_mean),  # pyro.sample("coef", dist.Normal(coef_mean, coef_sigma)),
        sigma=torch.tensor(0.1),
    )


def fit(dataset: rr.RRDataset):
    nuts_kernel = NUTS(rr.model_for_dataset)
    mcmc = MCMC(nuts_kernel,
                num_samples=400,
                warmup_steps=100,
                num_chains=1)

    mcmc.run(dataset, get_parameters)

    mcmc.summary(prob=0.8)


def main(args):
    epoch_window = (-0.1, 1.0)

    dataset = generator.sample_dataset(params=get_parameters(),
                                       num_words=150,
                                       epoch_window=epoch_window)
    from pprint import pprint
    pprint(dataset.params)

    if args.mode == "fit":
        fit(dataset)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("-m", "--mode", choices=["fit"],
                   default="fit")

    main(p.parse_args())
