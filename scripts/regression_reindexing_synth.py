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

import torch
from torchtyping import TensorType
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

from berp.generators import thresholded_recognition as generator
from berp.models import reindexing_regression as rr
from berp.typing import is_log_probability


TT = TensorType


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
    return sentences[:2]


def preprocess_dataset(dataset: generator.RRDataset):
    # DEV: Just work with first item.
    X_phon = dataset.X_phon.loc[0]
    p_word = dataset.p_word[0]

    candidate_tokens = dataset.candidate_tokens[0]
    n_words, n_candidate_words = dataset.candidate_ids[0].shape
    candidate_phonemes = dataset.candidate_phonemes[0]
    phoneme_onsets = dataset.phoneme_onsets[0]

    # compute surprisal data
    X = torch.tensor(dataset.p_word[0][:, 0]).view(-1, 1)
    X = -X / np.log(2)

    # prepare epoched response
    epochs_df = generator.dataset_to_epochs(
        dataset.X_word.xs(0, drop_level=False),
        dataset.y.xs(0, drop_level=False))
    assert len(set(epochs_df.groupby("token_idx").size())) == 1
    epochs_df["epoch_sample_idx"] = epochs_df.groupby(["item", "token_idx"]).cumcount()
    # DEV: jst first item.
    Y = epochs_df.droplevel(["item", "sample_idx"]) \
        .pivot(columns="epoch_sample_idx",
               values="signal")
    Y = torch.tensor(Y.values)[..., np.newaxis].float()

    return p_word, candidate_phonemes, phoneme_onsets, X, Y


def build_model(p_word, candidate_phonemes, phoneme_onsets, X, Y, sample_rate):
    # Parameters
    lambda_ = torch.tensor(1.0)
    threshold = pyro.sample("threshold",
                            dist.Beta(5, 5))
    a = torch.tensor(0.4)
    b = torch.tensor(0.2)
    coef_mean = torch.tensor([-1.])
    coef = pyro.sample("coef", dist.Normal(coef_mean, 0.1))

    # TODO check that masking is handled correctly

    p_word_posterior = rr.predictive_model(p_word,
                                           candidate_phonemes,
                                           generator.phoneme_confusion,
                                           lambda_)
    rec = rr.recognition_point_model(p_word_posterior,
                                     candidate_phonemes,
                                     generator.phoneme_confusion,
                                     lambda_,
                                     threshold)
    response = rr.epoched_response_model(X=X,
                                         coef=coef,
                                         recognition_points=rec,
                                         phoneme_onsets=phoneme_onsets,
                                         Y=Y, a=a, b=b,
                                         sample_rate=sample_rate)


def main(args):
    sentences = generate_sentences()
    dataset = generator.sample_dataset(sentences)

    input_data = preprocess_dataset(dataset)

    nuts_kernel = NUTS(build_model)
    mcmc = MCMC(nuts_kernel,
                num_samples=400,
                warmup_steps=100,
                num_chains=4)
    mcmc.run(*input_data, sample_rate=dataset.sample_rate)

    mcmc.summary(prob=0.8)


if __name__ == "__main__":
    p = ArgumentParser()

    main(p.parse_args())
