"""
Fit RR latent parameters by EM.
"""


from argparse import ArgumentParser
from pprint import pprint
import re
from typing import List, Tuple, NamedTuple, Callable

import numpy as np
import pandas as pd
import torch
from torch.distributions import constraints
from torch.nn.functional import pad
from torchtyping import TensorType
import pyro
from pyro import poutine
import pyro.distributions as dist
from sklearn.base import clone, BaseEstimator
from tqdm.auto import tqdm, trange

from berp.generators import stimulus
from berp.generators import thresholded_recognition_simple as generator
from berp.models import reindexing_regression as rr
from berp.models.trf import TemporalReceptiveField


def generate_sentences() -> List[str]:
    text = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, `and what is the use of a book,' thought Alice `without pictures or conversation?'
So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, `Oh dear! Oh dear! I shall be late!' (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again.

The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled `ORANGE MARMALADE', but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody, so she managed to put it into one of the cupboards as she fell past it.

`Well,' thought Alice to herself, `after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!' (Which was very likely true.)

Down, down, down. Would the fall never come to an end! `I wonder how many miles I've fallen by this time?' she said aloud. `I must be getting somewhere near the center of the earth. Let me see: that would be four thousand miles down, I think--' (for, you see, Alice had learnt several things of this sort in her lessons in the schoolroom, and though this was not a very good opportunity for showing off her knowledge, as there was no one to listen to her, still it was good practice to say it over) `--yes, that's about the right distance--but then I wonder what Latitude or Longitude I've got to?'
Alice had no idea what Latitude was, or Longitude either, but thought they were nice grand words to say.
""".strip()
    sentences = [s.strip().replace("\n", "") for s in re.split(r"[.?!]", text)]
    sentences = [s for s in sentences if s]
    return sentences


def get_parameters():
    # Sample model parameters.
    coef_mean = torch.tensor([-1.])
    coef_sigma = torch.tensor([0.1])
    return rr.ModelParameters(
        # lambda_=pyro.sample("lambda", dist.Uniform(0.8, 1.2)),  # torch.tensor(1.0),
        lambda_=pyro.deterministic("lambda", torch.tensor(1.0)),
        confusion=generator.phoneme_confusion,
        threshold=pyro.sample("threshold",
                              dist.Beta(1.2, 1.2)),

        # NB only used for generation, not in model
        a=pyro.deterministic("a", torch.tensor(0.2)),  # unif_categorical_rv("a", torch.tensor([0.3, 0.4, 0.5])),
        b=pyro.deterministic("b", torch.tensor(0.1)), # unif_categorical_rv("b", torch.tensor([0.05, 0.1, 0.15, 0.2])),
        coef=pyro.deterministic("coef", coef_mean),  # pyro.sample("coef", dist.Normal(coef_mean, coef_sigma)),
        sigma=pyro.sample("sigma", dist.Uniform(0.5, 1.2))
        # sigma=torch.tensor(1.0),
    )


def forward(params: rr.ModelParameters, encoder: TemporalReceptiveField,
            dataset: rr.RRDataset):
    _, _, design_matrix = rr.scatter_model(params, dataset)
    Y_pred = encoder.predict(design_matrix)
    return Y_pred


def weighted_design_matrix(weights: torch.Tensor, param_grid: List[rr.ModelParameters],
                           dataset: rr.RRDataset):
    num_features = dataset.X_variable.shape[1] + dataset.X_ts.shape[1]
    X_mixed = torch.zeros((dataset.Y.shape[0], num_features))
    for weight, params in zip(weights, param_grid):
        _, _, design_matrix = rr.scatter_model(params, dataset)
        X_mixed += weight * design_matrix

    return X_mixed


def e_step_grid(encoder: TemporalReceptiveField, dataset: rr.RRDataset,
                param_grid: List[rr.ModelParameters]):
    """
    Compute distribution over latent parameters conditioned on regression model.
    """
    results = torch.zeros(len(param_grid))
    for i, params in enumerate(param_grid):
        _, _, design_matrix = rr.scatter_model(params, dataset)
        test_ll = encoder.log_prob(design_matrix, dataset.Y).sum()
        results[i] = test_ll

    # Convert to probabilities
    results -= results.max()
    results = results.exp()
    weights = results / results.sum()

    return weights


def m_step(encoder: TemporalReceptiveField, weights: torch.Tensor, dataset: rr.RRDataset,
           param_grid: List[rr.ModelParameters]):
    """
    Estimate encoder which maximizes likelihood of data under expected design matrix.
    """
    X_mixed = weighted_design_matrix(weights, param_grid, dataset)
    return encoder.fit(X_mixed, dataset.Y)


def fit_em(dataset: rr.RRDataset, param_grid: List[rr.ModelParameters],
           val_dataset=None, n_iter=10, trf_alpha=None,
           early_stopping_patience=None):
    # TODO generalize
    tmin, tmax = 0.1, 0.4
    all_features = \
        [f"var_{i}" for i in range(dataset.X_variable.shape[1])] + \
        [f"ts_{i}" for i in range(dataset.X_ts.shape[1])]
    encoder = TemporalReceptiveField(
        tmin, tmax, dataset.sample_rate,
        feature_names=all_features,
        alpha=trf_alpha)

    def evaluate(weights: torch.Tensor, encoder: TemporalReceptiveField,
                 dataset: rr.RRDataset):
        with poutine.block():
            X_mixed = weighted_design_matrix(weights, param_grid, dataset)
            Y_pred = encoder.predict(X_mixed)

        mse = (Y_pred - dataset.Y).pow(2).mean()
        # for sensor_pred_obs in torch.stack([Y_pred, dataset.Y]).transpose(0, 2):
        #     # corrs.append(torch.corrcoef(sensor_pred_obs)[0, 1])
        #     corrs.append(torch.tensor(0.))
        # corr = torch.stack(corrs).mean()
        return mse

    point_estimate_keys = ["threshold", "lambda_"]
    metric_names = ["mse", "corr"]

    # if test_dataset is not None:
    #     # Calculate expected MSE of random model parameters.
    #     weight_dist = dist.Dirichlet(torch.tensor([0.1]))
    #     n_evaluations = 50
    #     metrics = np.zeros((n_evaluations, len(metric_names)))
    #     for i in trange(n_evaluations):
    #         weights = weight_dist.sample_n(len(param_grid))
    #         weights /= weights.sum()
    #         metrics[i] = evaluate(weights, encoder, test_dataset)
        
    #     print("Random model evaluation:")
    #     pprint(list(zip(metric_names, metrics.mean(axis=0))))

    weights = torch.zeros((n_iter, len(param_grid)))
    coefs = [encoder.coef_]

    best_val_loss = np.inf
    patience_counter = 0
    with trange(n_iter) as titer:
        for i in titer:
            weights[i] = e_step_grid(encoder, dataset, param_grid)
            encoder = m_step(encoder, weights[i], dataset, param_grid)
            coefs.append(encoder.coef_)

            train_loss = evaluate(weights[i], encoder, dataset)
            iter_results = dict(train_loss=train_loss.item())
            if val_dataset is not None:
                val_loss = evaluate(weights[i], encoder, val_dataset)
                iter_results["val_loss"] = val_loss.item()

                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            break

            titer.set_postfix(**iter_results)

    # Trim weights if we stopped early.
    weights = weights[:i + 1]

    return weights, coefs


def main(args):
    epoch_window = (-0.1, 1.0)

    # TODO off by one with padding token? does this matter?

    if args.stim == "random":
        stim = stimulus.RandomStimulusGenerator(
            phoneme_voc_size=len(generator.phoneme2idx))
    elif args.stim == "sentences":
        sentences = generate_sentences()
        phonemes = list("abcdefghijklmnopqrstuvwxyz") + ["_"]
        stim = stimulus.NaturalLanguageStimulusGenerator(
            phonemes=phonemes,
            hf_model="gpt2"
        )

        from functools import partial
        stim = partial(stim, sentences)
    else:
        raise ValueError("Unknown stimulus type: {}".format(args.stim))

    dataset = generator.sample_dataset(get_parameters(),
                                       stim,
                                       response_type=args.response_type,
                                       epoch_window=epoch_window)
    test_dataset = None
    if args.stim == "random":
        test_dataset = generator.sample_dataset(dataset.params, stim,
                                                response_type=args.response_type,
                                                epoch_window=epoch_window)

    from pprint import pprint
    pprint(dataset.params)

    param_grid = [get_parameters() for _ in range(args.grid_size)]

    if args.mode == "fit":
        fit_em(dataset, param_grid, val_dataset=test_dataset)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("-m", "--mode", choices=["fit", "fit_map", "fit_importance"],
                   default="fit")
    p.add_argument("-s", "--stim", choices=["random", "sentences"],
                   default="random")
    p.add_argument("-r", "--response_type", choices=["gaussian", "square", "n400"],
                   default="gaussian")
    p.add_argument("-g", "--grid_size", type=int, default=100)

    main(p.parse_args())
