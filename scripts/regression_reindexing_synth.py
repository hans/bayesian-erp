from argparse import ArgumentParser
from pathlib import Path
import pickle
import re
from typing import List, Tuple, NamedTuple, Callable

from typeguard import typechecked

import torch
from torch.nn.functional import pad
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import pyro.poutine as poutine

from berp.generators import thresholded_recognition as generator
import berp.infer
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


def fit_importance(dataset: rr.RRDataset):
    def model():
        result = rr.model_wrapped(get_parameters, dataset)
        return result.params.threshold

    importance, slice_means = berp.infer.fit_importance(
        model, guide=None, num_samples=5000
    )

    # Evaluate parameter estimate on a sliding window of samples to
    # understand how many samples we actually needed.
    from pprint import pprint
    pprint(slice_means)

    return importance



def main(args):
    epoch_window = (-0.1, 2.5)

    if args.dataset is not None and args.dataset.exists():
        with args.dataset.open("rb") as f:
            dataset = pickle.load(f)
    else:
        sentences = generate_sentences()
        dataset = generator.sample_dataset(sentences,
                                           epoch_window=epoch_window)

        if args.dataset is not None:
            with args.dataset.open("wb") as f:
                pickle.dump(dataset, f)

    if args.mode == "fit":
        fit(dataset)
    elif args.mode == "fit_importance":
        fit_importance(dataset)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("-m", "--mode", choices=["fit", "fit_importance"],
                   default="fit")
    p.add_argument("-d", "--dataset", type=Path)

    main(p.parse_args())
