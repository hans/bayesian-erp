"""
Run a reindexing linear regression on synthetic data.
"""

from argparse import ArgumentParser
from pprint import pprint

import pandas as pd
import pyro
from pyro import poutine
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, infer_discrete, config_enumerate
from pyro.infer.autoguide import AutoDiagonalNormal, AutoDelta, AutoNormal
import torch

from berp.models import reindexing_regression
import berp.synthesize_n400 as syn


def synthesize_dataset(size, **kwargs):
    """
    Generate, epoch, and torch-ify synthetic dataset.
    """
    X_word, X_phon, y = syn.sample_dataset_with_phons(size)

    epochs_df = syn.dataset_to_epochs(X_phon, y, **kwargs)
    epochs_df["value"] = epochs_df.value_N400 - epochs_df.baseline_N400
    y_all = epochs_df.value.unstack("phon_idx").fillna(0.)

    merged_df = pd.merge(y_all, X_word[["surprisal"]],
                         how="inner",
                         left_index=True, right_index=True)

    X = merged_df.surprisal.values
    y = merged_df[y_all.columns].values

    X = torch.tensor(X).unsqueeze(1).float()
    y = torch.tensor(y).float()

    return X, y


def train(svi, X, y, num_iterations):
    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(X, y)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(X)))


def do_discrete_inference(model, guide, X, y):
    guide_trace = poutine.trace(guide).get_trace(X, y)  # record the globals
    trained_model = poutine.replay(model, trace=guide_trace)  # replay the globals

    def classifier(X, y, temperature=0):
        inferred_model = infer_discrete(model, temperature=temperature,
                                        first_available_dim=-2)  # avoid conflict with data plate
        trace = poutine.trace(inferred_model).get_trace(X, y)
        # pprint(trace.nodes["obs"])
        # pprint(y)
        return trace.nodes["index"]["value"]

    return classifier(X, y).numpy()


def init_loc_fn(site):
    if site["name"] == "coef":
        return torch.tensor(0.)
    elif site["name"] == "weights":
        return torch.ones(5) / 5.
    raise ValueError(site["name"])


def eval(args):
    # test_window = (args.test_window_left, args.test_window_right)
    # X, y = synthesize_dataset(args.size, test_window=test_window)

    # DEV: try this with fake data first.
    N, d = 50, 5
    import numpy as np
    import time
    np.random.seed(int(time.time()))
    y = np.random.uniform(-10., 10., size=(N, d))
    # shuffled = np.arange(n)
    # np.random.shuffle(shuffled)
    # y = -0.5 * np.diag(X)[:, shuffled]
    # idxs = np.random.randint(d, size=N)
    idxs = np.ones(N).astype(int)
    X = y[np.arange(N), idxs]
    X = torch.tensor(X).float()
    y = torch.tensor(y).float()

    model, guide = reindexing_regression.build_model_guide(args.bayesian)
    # guide = AutoDiagonalNormal(poutine.block(model, hide=["index"]))
    # guide = AutoDelta(poutine.block(model, expose=["coef"]))
    # guide = reindexing_regression.guide

    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    adam = Adam({"lr": 0.1})

    def initialize(X, y, seed, model, guide, optim, elbo):
        global svi
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        # guide = AutoNormal(poutine.block(model, expose=["coef", "weights"]),
        #                    init_loc_fn=init_loc_fn)
        svi = SVI(model, guide, optim, loss=elbo)
        return svi.loss(model, guide, X, y), seed

    loss, seed = min(initialize(X, y, seed, model, guide, adam, elbo)
                     for seed in range(100))
    print(f"seed = {seed}, initial_loss = {loss}")
    initialize(X, y, seed, model, guide, adam, elbo)

    train(svi, X, y, num_iterations=args.n_iter)

    # map_estimates = guide(X, y)
    # pprint(map_estimates)
    for k, v in pyro.get_param_store().items():
        print(k, v)

    # TODO how do I get parameter estimates for individual training samples?
    # as I understand it, I established the "weights" parameter under a plate,
    # so that should mean I have N weight vectors.

    index_hat = do_discrete_inference(model, guide, X, y)

    print(idxs)
    print(index_hat)
    print((idxs == index_hat).mean())

    return idxs, index_hat


def main(args):
    accs = []
    for _ in range(100):
        pyro.clear_param_store()
        idxs, index_hat = eval(args)
        accs.append((idxs == index_hat).mean())
    print(accs)
    print(np.mean(accs))


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("--test_window_left", type=float, default=0.3)
    p.add_argument("--test_window_right", type=float, default=0.5)

    p.add_argument("--size", type=int, default=10)
    p.add_argument("--level", choices=["word", "phoneme"], default="word")

    p.add_argument("--bayesian", default=False, action="store_true")

    p.add_argument("--n_iter", type=int, default=200)

    main(p.parse_args())
