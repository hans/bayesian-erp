"""
Run a basic linear regression on synthetic data.
"""

from argparse import ArgumentParser
from pprint import pprint

import pandas as pd
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
import torch

from berp.models.basic import RegressionModel
import berp.synthesize_n400 as syn


def synthesize_dataset(size, level="word", **kwargs):
    """
    Generate, epoch, and torch-ify synthetic dataset.
    """
    X_word, X_phon, y = syn.sample_dataset_with_phons(size)
    X = X_word if level == "word" else X_phon

    epochs_df = syn.dataset_to_epochs(X, y, **kwargs)
    merged_df = pd.merge(epochs_df, X[["surprisal"]],
                         left_index=True, right_index=True)

    X = merged_df.surprisal.values
    y = (merged_df.value_N400 - merged_df.baseline_N400).values

    X = torch.tensor(X).unsqueeze(1)
    y = torch.tensor(y)

    return X, y


def train(svi, X, y, num_iterations):
    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(X, y)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(X)))


def main(args):
    test_window = (args.test_window_left, args.test_window_right)
    X, y = synthesize_dataset(args.size, level=args.level,
                              test_window=test_window)

    model = RegressionModel()
    guide = AutoDiagonalNormal(model)

    adam = Adam({"lr": 0.1, "betas": [0.8, 0.99]})
    svi = SVI(model, guide, adam, loss=Trace_ELBO(max_plate_nesting=1))

    train(svi, X, y, num_iterations=args.n_iter)

    guide.requires_grad_(False)
    pprint(guide.quantiles([0.25, 0.5, 0.75]))


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("--test_window_left", type=float, default=0.3)
    p.add_argument("--test_window_right", type=float, default=0.5)

    p.add_argument("--size", type=int, default=10)
    p.add_argument("--level", choices=["word", "phoneme"], default="word")

    p.add_argument("--n_iter", type=int, default=200)

    main(p.parse_args())
