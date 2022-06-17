from argparse import ArgumentParser
from copy import deepcopy
import re
from typing import List, Tuple, NamedTuple, Callable

from typeguard import typechecked

import numpy as np
import torch
from torch.distributions import constraints
from torch.nn.functional import pad
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, EmpiricalMarginal, TracePosterior
import pyro.poutine as poutine

from berp.generators import thresholded_recognition_simple as generator
from berp.infer import Importance
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
        sigma=pyro.sample("sigma", dist.Uniform(1.0, 2.0))
        # sigma=torch.tensor(1.0),
    )


def get_parameters_mle():
    coef_mean = torch.tensor([0., -1.])
    # return rr.ModelParameters(
    #     lambda_=torch.tensor(1.0),
    #     confusion=generator.phoneme_confusion,
    #     threshold=pyro.param("threshold", torch.tensor(0.5),
    #                          constraint=constraints.unit_interval),
    #     a=torch.tensor(0.4),
    #     b=torch.tensor(0.1),
    #     coef=pyro.deterministic("coef", coef_mean),  # pyro.sample("coef", dist.Normal(coef_mean, coef_sigma)),
    #     sigma=pyro.param("sigma", torch.tensor(1.0),
    #                      constraint=constraints.interval(1.0, 2.0))
    # )
    return rr.ModelParameters(
        lambda_=torch.tensor(1.0),
        confusion=generator.phoneme_confusion,
        threshold=pyro.param("threshold",
                             torch.tensor(0.5), constraint=constraints.unit_interval),
        a=torch.tensor(0.4),
        b=torch.tensor(0.1),
        coef=pyro.deterministic("coef", coef_mean),  # pyro.sample("coef", dist.Normal(coef_mean, coef_sigma)),
        sigma=pyro.deterministic("sigma", torch.tensor(1.0)),
        # sigma=torch.tensor(1.0),
    )


def fit(dataset: rr.RRDataset):
    from pyro.infer import HMC
    kernel = HMC(rr.model_for_dataset, step_size=1.)
    # kernel = NUTS(rr.model_for_dataset)
    mcmc = MCMC(kernel,
                num_samples=50,
                warmup_steps=50,
                num_chains=1)

    mcmc.run(dataset, get_parameters)

    mcmc.summary()


def fit_map(dataset: rr.RRDataset):
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoDelta
    from pyro.optim import Adam
    # autoguide = AutoDelta(model)

    kwargs = dict(
        p_word=dataset.p_word,
        candidate_phonemes=dataset.candidate_phonemes,
        phoneme_onsets=dataset.phoneme_onsets,
        word_lengths=dataset.word_lengths,
        X_epoched=dataset.X_epoch,
        Y_epoched=dataset.Y_epoch,
        sample_rate=dataset.sample_rate,
        epoch_window=dataset.epoch_window,
    )
    def model(**kwargs):
        params = get_parameters_mle()
        print(params.threshold)
        return rr.model(params=params, **kwargs)

    def guide(*args, **kwargs): pass

    pyro.clear_param_store()
    opt = Adam({"lr": 0.01})
    svi = SVI(model, guide, opt, loss=Trace_ELBO())

    for step in range(200):
        loss = svi.step(**kwargs)
        if step % 100 == 0:
            print(step, loss)

    # print(autoguide.median())
    print(pyro.param("threshold"))


def slice_trace_posterior(tp: TracePosterior, num_samples: int) -> TracePosterior:
    if tp.num_chains > 1:
        raise NotImplementedError()

    ret = deepcopy(tp)
    ret.log_weights = ret.log_weights[:num_samples]
    ret.exec_traces = ret.exec_traces[:num_samples]
    ret.chain_ids = ret.chain_ids[:num_samples]

    return ret


def fit_importance(dataset: rr.RRDataset):
    kwargs = dict(
        p_word=dataset.p_word,
        candidate_phonemes=dataset.candidate_phonemes,
        phoneme_onsets=dataset.phoneme_onsets,
        word_lengths=dataset.word_lengths,
        X_epoched=dataset.X_epoch,
        Y_epoched=dataset.Y_epoch,
        sample_rate=dataset.sample_rate,
        epoch_window=dataset.epoch_window,
    )

    def model(**kwargs):
        params = get_parameters()
        rr.model(params=params, **kwargs)

        return params.threshold

    importance = Importance(model, num_samples=5000)
    emp_marginal = EmpiricalMarginal(importance.run(**kwargs))

    print(emp_marginal.mean)

    # Evaluate parameter estimate on a sliding window of samples to
    # understand how many samples we actually needed.
    sample_points = np.linspace(1, importance.num_samples, 10).round().astype(int)
    for num_samples in sample_points:
        im_sliced = slice_trace_posterior(importance, num_samples)
        emp_marginal_sliced = EmpiricalMarginal(im_sliced)
        print(num_samples, emp_marginal_sliced.mean)

    return importance, emp_marginal


def main(args):
    epoch_window = (-0.1, 1.0)

    dataset = generator.sample_dataset(params=get_parameters(),
                                       num_words=500,
                                       epoch_window=epoch_window)
    from pprint import pprint
    pprint(dataset.params)

    if args.mode == "fit":
        fit(dataset)
    elif args.mode == "fit_map":
        fit_map(dataset)
    elif args.mode == "fit_importance":
        fit_importance(dataset)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("-m", "--mode", choices=["fit", "fit_map", "fit_importance"],
                   default="fit")

    main(p.parse_args())
