from argparse import ArgumentParser
from pprint import pprint
import re
from typing import List, Tuple, NamedTuple, Callable

import numpy as np
import pandas as pd
import torch
from torch.distributions import constraints
from torch.nn.functional import pad
import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, EmpiricalMarginal, TracePosterior

from berp.generators import stimulus
from berp.generators import thresholded_recognition_simple as generator
import berp.infer
from berp.models import reindexing_regression as rr


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


def unif_categorical_rv(name, choices: torch.Tensor):
    one_hot = dist.OneHotCategorical(torch.ones_like(choices) / choices.numel())
    one_hot = pyro.sample(name, one_hot)
    return (one_hot * choices).sum()


def get_parameters():
    # Sample model parameters.
    coef_mean = torch.tensor([0., -1.])
    coef_sigma = torch.tensor([1e-6, 0.1])
    return rr.ModelParameters(
        # lambda_=pyro.sample("lambda", dist.Uniform(0.8, 1.2)),  # torch.tensor(1.0),
        lambda_=pyro.deterministic("lambda", torch.tensor(1.0)),
        confusion=generator.phoneme_confusion,
        threshold=pyro.sample("threshold",
                              dist.Beta(1.2, 1.2)),
        a=pyro.sample("a", dist.Uniform(0.3, 0.5)),  # unif_categorical_rv("a", torch.tensor([0.3, 0.4, 0.5])),
        b=pyro.sample("b", dist.Uniform(0.05, 0.2)), # unif_categorical_rv("b", torch.tensor([0.05, 0.1, 0.15, 0.2])),
        coef=pyro.deterministic("coef", coef_mean),  # pyro.sample("coef", dist.Normal(coef_mean, coef_sigma)),
        sigma=pyro.sample("sigma", dist.Uniform(0.5, 1.2))
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
                             torch.tensor(0.5),
                             constraint=constraints.unit_interval),  # type: ignore
        a=torch.tensor(0.4),
        b=torch.tensor(0.1),
        coef=pyro.deterministic("coef", coef_mean),  # pyro.sample("coef", dist.Normal(coef_mean, coef_sigma)),
        sigma=pyro.deterministic("sigma", torch.tensor(1.0)),
        # sigma=torch.tensor(1.0),
    )


def fit(dataset: rr.RRDataset):
    from pyro.infer import HMC
    kernel = HMC(rr.model_wrapped, step_size=1.)
    # kernel = NUTS(rr.model_for_dataset)
    mcmc = MCMC(kernel,
                num_samples=50,
                warmup_steps=50,
                num_chains=1)

    mcmc.run(get_parameters, dataset)

    mcmc.summary()


def fit_map(dataset: rr.RRDataset):
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoDelta
    from pyro.optim import Adam  # type: ignore
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


def fit_importance(dataset: rr.RRDataset, test_dataset=None):
    def evaluate(params: rr.ModelParameters, dataset: rr.RRDataset):
        with poutine.block():
            test_result = rr.model(params, dataset)
        mse = ((test_result.q_obs - test_result.q_pred) ** 2).mean()
        corr = torch.corrcoef(torch.stack([test_result.q_obs, test_result.q_pred]))[0, 1]
        return [mse, corr]

    point_estimate_keys = ["threshold", "lambda_", "a", "b", "sigma"]
    metric_names = ["mse", "corr"]

    def model():
        # Evaluate parameter point estimates.
        result = rr.model_wrapped(get_parameters, dataset)
        ret = [getattr(result.params, key) for key in point_estimate_keys]

        if test_dataset is not None:
            # Evaluate MSE on test dataset points.
            ret.extend(evaluate(result.params, test_dataset))

        return torch.stack(ret)

    if test_dataset is not None:
        # Calculate expected MSE of random model parameters.    
        random_metrics = np.array(
            [evaluate(get_parameters(), test_dataset) for _ in range(100)])
        print("Random model evaluation:")
        pprint(list(zip(metric_names, random_metrics.mean(axis=0))))

    importance, slice_means = berp.infer.fit_importance(
        model, guide=None, num_samples=5000)

    result_labels = point_estimate_keys
    if test_dataset is not None:
        result_labels += metric_names
    
    slice_means = [(idx, dict(zip(result_labels, values.numpy())))
                   for idx, values in slice_means]

    # Evaluate parameter estimate on a sliding window of samples to
    # understand how many samples we actually needed.
    df = pd.DataFrame.from_dict(dict(slice_means), orient="index")
    df.index.name = "num_samples"
    print(df)

    return importance


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

    if args.mode == "fit":
        fit(dataset)
    elif args.mode == "fit_map":
        fit_map(dataset)
    elif args.mode == "fit_importance":
        fit_importance(dataset, test_dataset=test_dataset)


if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("-m", "--mode", choices=["fit", "fit_map", "fit_importance"],
                   default="fit")
    p.add_argument("-s", "--stim", choices=["random", "sentences"],
                   default="random")
    p.add_argument("-r", "--response_type", choices=["gaussian", "square", "n400"],
                   default="gaussian")

    main(p.parse_args())
