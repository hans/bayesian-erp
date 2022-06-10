from argparse import ArgumentParser
import re
from typing import List, Tuple, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from tqdm.notebook import tqdm
from icecream import ic
from typeguard import typechecked
import pytest

import torch
from torch.nn.functional import pad
from torchtyping import TensorType
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import pyro.poutine as poutine

from berp.generators import thresholded_recognition as generator
from berp.models import reindexing_regression as rr
from berp.typing import is_log_probability, DIMS


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
    return sentences  # DEV


def pad_phoneme_data(dataset) -> Tuple[TensorType[DIMS.B, DIMS.N_C, DIMS.N_P], ...]:
    # we will pass flattened data representations, where each word in each
    # item is an independent sample. to do this, we have to pad N_P to be
    # equivalent across items.
    max_n_p = max(cand.shape[2] for cand in dataset.candidate_phonemes)

    candidate_phonemes = torch.cat([
        pad(cand, (0, max_n_p - cand.shape[2], 0, 0, 0, 0),
            value=generator.phoneme2idx["_"])
        for cand in dataset.candidate_phonemes
    ])

    # NB there are often longer representations (more phonemes necessary) in
    # candidate_phonemes, because the candidate words are longer than the ground
    # truth word. phoneme_onsets is just as long as the ground truth longest word.
    max_n_gt_p = max(onsets.shape[1] for onsets in dataset.phoneme_onsets)
    phoneme_onsets = torch.cat([
        pad(onsets, (0, max_n_gt_p - onsets.shape[1], 0, 0), value=0.)
        for onsets in dataset.phoneme_onsets
    ])

    phoneme_mask = ...

    return candidate_phonemes, phoneme_onsets, phoneme_mask


def preprocess_dataset(dataset: generator.RRDataset, epoch_window):
    # We will flatten all observations across item and word
    p_word = torch.cat(dataset.p_word)

    n_words, n_candidate_words = dataset.candidate_ids[0].shape

    candidate_phonemes, phoneme_onsets, phoneme_mask = pad_phoneme_data(dataset)
    word_lengths = torch.tensor([word_length for item in dataset.word_lengths
                                 for word_length in item])

    # prepare epoched response
    if phoneme_onsets.max() > epoch_window[1]:
        raise ValueError(f"Some words have phoneme onsets outside the word "
                         f"epoch window {epoch_window} (max onset {phoneme_onsets.max()}). "
                         f"This won't work -- increase the epoch window.")
    epochs_df = generator.dataset_to_epochs(dataset.X_word, dataset.y,
                                            epoch_window=epoch_window)
    epochs_df["epoch_sample_idx"] = epochs_df.groupby(["item", "token_idx"]).cumcount()

    # this yields a df with two index levels (item and sample_idx).
    # sorting is retained. the
    # underlying ndarray is flattened in a rep where sample_idx varies within
    # item. so it's safe to just steal this flattened matrix and use it
    Y = epochs_df.droplevel("sample_idx") \
        .pivot(columns="epoch_sample_idx",
               values="signal")
    Y = torch.tensor(Y.values)[..., np.newaxis].float()

    # compute predictors: surprisal, baseline
    X = -p_word[:, 0] / np.log(2)
    baseline = epochs_df[epochs_df.epoch_time <= 0] \
        .groupby(["item", "token_idx"]).signal.mean()
    baseline = torch.tensor(baseline.values)
    X = torch.stack([baseline, X], dim=1).float()

    return p_word, word_lengths, candidate_phonemes, phoneme_onsets, X, Y


@typechecked
def model(params: rr.ModelParameters,
          p_word, word_lengths,
          candidate_phonemes, phoneme_onsets,
          X, Y, sample_rate, epoch_window):

    # TODO move to reindexing_regression?

    p_word_posterior = rr.predictive_model(p_word,
                                           candidate_phonemes,
                                           params.confusion,
                                           params.lambda_)
    rec = rr.recognition_point_model(p_word_posterior,
                                     word_lengths,
                                     params.threshold)
    response = rr.epoched_response_model(X=X,
                                         coef=params.coef,
                                         recognition_points=rec,
                                         phoneme_onsets=phoneme_onsets,
                                         Y=Y, a=params.a, b=params.b,
                                         sample_rate=sample_rate,
                                         epoch_window=epoch_window)


def build_model(*args, **kwargs):
    # Sample model parameters.
    coef_mean = torch.tensor([1., -1.])
    params = rr.ModelParameters(
        lambda_=torch.tensor(1.0),
        confusion=generator.phoneme_confusion,
        threshold=pyro.sample("threshold",
                              dist.Beta(1.2, 1.2)),
        a=torch.tensor(0.4),
        b=torch.tensor(0.1),
        coef=pyro.sample("coef", dist.Normal(coef_mean, 0.1)),
    )
    return model(params, *args, **kwargs)


def fit(dataset, args, epoch_window, **preprocess_args):
    input_data = preprocess_dataset(dataset, epoch_window, **preprocess_args)

    # build_model(*input_data, sample_rate=dataset.sample_rate)

    nuts_kernel = NUTS(build_model)
    mcmc = MCMC(nuts_kernel,
                num_samples=400,
                warmup_steps=100,
                num_chains=4)
    mcmc.run(*input_data, sample_rate=dataset.sample_rate,
             epoch_window=epoch_window)

    mcmc.summary(prob=0.8)



def main(args):
    sentences = generate_sentences()
    dataset = generator.sample_dataset(sentences)

    epoch_window = (-0.1, 2.5)
    if args.mode == "fit":
        fit(dataset, args, epoch_window=epoch_window)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("-m", "--mode", choices=["fit"],
                   default="fit")

    main(p.parse_args())
