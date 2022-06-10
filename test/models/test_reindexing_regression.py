from pathlib import Path
from typing import Dict

from icecream import ic
import numpy as np
import pyro
from pyro import poutine
from pyro import distributions as dist
import pytest
import torch

from berp.generators import thresholded_recognition_simple as generator
from berp.models import reindexing_regression as rr


def get_parameters():
    coef_mean = torch.tensor([0., -1.])

    # NB we establish deterministic sites below so that they can be conditioned
    return rr.ModelParameters(
        lambda_=torch.tensor(1.0),
        confusion=generator.phoneme_confusion,
        threshold=pyro.deterministic("threshold", torch.tensor(0.7)),
        a=torch.tensor(0.4),
        b=torch.tensor(0.05),
        coef=pyro.deterministic("coef", coef_mean),
    )


@pytest.fixture(scope="session")
def soundness_dataset():
    return generator.sample_dataset(get_parameters(), num_words=50)


def build_model(d: generator.RRDataset):
    # hacky: re-call get_parameters so that we get the conditioned parameter values
    params = get_parameters()

    p_word_posterior = rr.predictive_model(d.p_word,
                                           d.candidate_phonemes,
                                           params.confusion,
                                           params.lambda_)
    rec = rr.recognition_point_model(p_word_posterior,
                                     d.word_lengths,
                                     params.threshold)
    response = rr.epoched_response_model(X=d.X_epoch,
                                         coef=params.coef,
                                         recognition_points=rec,
                                         phoneme_onsets=d.phoneme_onsets,
                                         Y=d.Y_epoch,
                                         a=d.params.a, b=d.params.b,
                                         sigma=torch.tensor(1.),
                                         sample_rate=d.sample_rate,
                                         epoch_window=d.epoch_window)

    return response


def trace_conditional(conditioning: Dict, *args, **kwargs):
    """
    Evaluate model trace for given conditioning set.
    """
    conditioned_model = poutine.condition(build_model, conditioning)
    return poutine.trace(conditioned_model).get_trace(*args, **kwargs)


def model_forward(dataset, conditioning=None):
    if conditioning is None:
        conditioning = {}

    model_trace = trace_conditional(conditioning, dataset)
    return model_trace


def _run_soundness_check(conditions, background_condition,
                         dataset):
    """
    Verify that the probability of the ground truth data is greatest under the
    generating parameters (compared to perturbations thereof).
    """
    condition_logprobs = []
    test_keys = set(key for condition_dict in conditions
                    for key in condition_dict.keys())
    for condition in conditions:
        condition.update(background_condition)
        print(condition)
        log_joint = model_forward(dataset, condition).log_prob_sum()

        if log_joint.isinf():
            raise RuntimeError(str(condition))
        condition_logprobs.append(log_joint)

    # Renormalize+exponentiate.
    condition_logprobs = torch.stack(condition_logprobs)
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


def test_soundness_threshold(soundness_dataset):
    background_condition = {"coef": torch.tensor([1., -1])}

    gt_condition = {"threshold": soundness_dataset.params.threshold}
    alt_conditions = [{"threshold": x}
                      for x in torch.rand(10)]
    all_conditions = [gt_condition] + alt_conditions

    _run_soundness_check(all_conditions, background_condition,
                         soundness_dataset)


def test_soundness_coef(soundness_dataset):
    background_condition = {"threshold": soundness_dataset.params.threshold}

    gt_condition = {"coef": torch.tensor([0., -1])}
    alt_conditions = [{"coef": torch.tensor(x)}
                      for x in [[1., 1.],
                                [1., -5.],
                                [-1., -1.]]]
    all_conditions = [gt_condition] + alt_conditions

    _run_soundness_check(all_conditions, background_condition,
                         soundness_dataset)


def test_recognition_logic(soundness_dataset):
    """
    For possible thresholds T2 > T1, inferred recognition points K2, K1 should
    obey K2 >= K1. Corresponding onsets O2 >= O1 as well.
    """

    t1 = soundness_dataset.params.threshold
    t2 = np.sqrt(soundness_dataset.params.threshold)
    assert t2 > t1

    trace_1 = model_forward(soundness_dataset, {"threshold": t1})
    trace_2 = model_forward(soundness_dataset, {"threshold": t2})

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
