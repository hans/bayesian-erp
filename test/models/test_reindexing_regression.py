from pathlib import Path
import re
from typing import Dict, Callable

from icecream import ic
import numpy as np
import pyro
from pyro import poutine
from pyro import distributions as dist
import pytest
import torch

from berp.generators import thresholded_recognition_simple as generator
from berp.generators import thresholded_recognition as generator2
from berp.models import reindexing_regression as rr


def get_parameters():
    coef_mean = torch.tensor([0., -1.])

    # NB we establish deterministic sites below so that they can be conditioned
    return rr.ModelParameters(
        lambda_=torch.tensor(1.0),
        confusion=generator.phoneme_confusion,
        threshold=pyro.deterministic("threshold", torch.tensor(0.7)),
        a=torch.tensor(0.4),
        b=torch.tensor(0.1),
        coef=pyro.deterministic("coef", coef_mean),
        sigma=torch.tensor(1.0),
    )


def get_parameters2():
    coef_mean = torch.tensor([1., -1.])

    # NB we establish deterministic sites below so that they can be conditioned
    return rr.ModelParameters(
        lambda_=torch.tensor(1.0),
        confusion=generator2.phoneme_confusion,
        threshold=pyro.deterministic("threshold", torch.tensor(0.7)),
        a=torch.tensor(0.4),
        b=torch.tensor(0.2),
        coef=pyro.deterministic("coef", coef_mean),
        sigma=torch.tensor(0.1),
    )


@pytest.fixture(scope="session")
def soundness_dataset1():
    return (generator.sample_dataset(get_parameters(), num_words=200),
            get_parameters)


@pytest.fixture(scope="session")
def sentences():
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


@pytest.fixture(scope="session")
def soundness_dataset2(sentences):
    return (generator2.sample_dataset(sentences=sentences,
                                      params=get_parameters2()),
            get_parameters2)


def trace_conditional(conditioning: Dict, *args, **kwargs):
    """
    Evaluate model trace for given conditioning set.
    """
    conditioned_model = poutine.condition(rr.model_for_dataset, conditioning)
    return poutine.trace(conditioned_model).get_trace(*args, **kwargs)  # type: ignore


def model_forward(dataset, parameters, conditioning=None):
    if conditioning is None:
        conditioning = {}

    model_trace = trace_conditional(conditioning, dataset, parameters)
    return model_trace


def _run_soundness_check(conditions, background_condition,
                         dataset: rr.RRDataset,
                         parameters: Callable[[], rr.ModelParameters]):
    """
    Verify that the probability of the ground truth data is greatest under the
    generating parameters (compared to perturbations thereof).
    """

    condition_logprobs = []
    condition_traces = []
    test_keys = set(key for condition_dict in conditions
                    for key in condition_dict.keys())
    for condition in conditions:
        condition.update(background_condition)
        print(condition)
        trace = model_forward(dataset, parameters, condition)
        log_joint = trace.log_prob_sum()

        if log_joint.isinf():
            raise RuntimeError(str(condition))
        
        condition_logprobs.append(log_joint)
        condition_traces.append(trace)

    # Renormalize+exponentiate.
    from pprint import pprint
    
    condition_logprobs = torch.stack(condition_logprobs)
    pprint(sorted(zip(conditions, condition_logprobs.numpy()),
                  key=lambda x: -x[1]))

    condition_logprobs = (condition_logprobs - condition_logprobs.max()).exp()
    condition_logprobs /= condition_logprobs.sum()

    from pprint import pprint
    result = sorted(zip(conditions, condition_logprobs.numpy()),
                    key=lambda x: -x[1])
    pprint(result)

    map_result = result[0][0]
    gt_result = conditions[0]
    for test_key in test_keys:
        assert torch.equal(map_result[test_key], gt_result[test_key]), \
            f"Ground truth parameter is the MAP choice ({test_key})"

    return result, condition_traces


@pytest.mark.parametrize("dataset_fixture", ["soundness_dataset1", "soundness_dataset2"])
def test_soundness_threshold(request, dataset_fixture):
    dataset, parameters = request.getfixturevalue(dataset_fixture)

    background_condition = {"coef": dataset.params.coef}

    gt_condition = {"threshold": dataset.params.threshold}
    alt_conditions = [{"threshold": x}
                      for x in torch.rand(10)]
    all_conditions = [gt_condition] + alt_conditions

    _run_soundness_check(all_conditions, background_condition,
                         dataset, parameters)


@pytest.mark.parametrize("dataset_fixture", ["soundness_dataset1", "soundness_dataset2"])
def test_soundness_coef(request, dataset_fixture):
    dataset, parameters = request.getfixturevalue(dataset_fixture)

    background_condition = {"threshold": dataset.params.threshold}

    gt_condition = {"coef": dataset.params.coef}
    alt_conditions = [{"coef": torch.tensor(x)}
                      for x in [[1., 1.],
                                [1., -5.],
                                [-1., -1.]]]
    all_conditions = [gt_condition] + alt_conditions

    _run_soundness_check(all_conditions, background_condition,
                         dataset, parameters)


def test_recognition_logic(soundness_dataset1):
    """
    For possible thresholds T2 > T1, inferred recognition points K2, K1 should
    obey K2 >= K1. Corresponding onsets O2 >= O1 as well.
    """
    dataset, params = soundness_dataset1

    t1 = dataset.params.threshold
    t2 = np.sqrt(dataset.params.threshold)
    assert t2 > t1

    trace_1 = model_forward(dataset, params, {"threshold": t1})
    trace_2 = model_forward(dataset, params, {"threshold": t2})

    rec_1 = trace_1.nodes["recognition_point"]["value"]
    rec_2 = trace_2.nodes["recognition_point"]["value"]
    ic(rec_2 - rec_1)
    ic(np.where(rec_2 < rec_1))
    assert bool((rec_2 >= rec_1).all()), "Recognition points should not decrease when threshold increases"

    rec_onset_1 = trace_1.nodes["recognition_onset"]["value"]
    rec_onset_2 = trace_2.nodes["recognition_onset"]["value"]
    ic(rec_onset_2 - rec_onset_1)
    ic(np.where(rec_onset_2 < rec_onset_1))
    # NB if this assertion fails and the above passes, it's an error in the
    # onset data representation / indexing logic.
    assert bool((rec_onset_2 >= rec_onset_1).all()), "Recognition onsets should not decrease when threshold increases"
