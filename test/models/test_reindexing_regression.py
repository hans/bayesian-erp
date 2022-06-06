from pathlib import Path
from typing import Dict

from icecream import ic
import numpy as np
import pyro
from pyro import poutine
import pytest
import torch

from berp.generators import thresholded_recognition as generator

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))
# steal dataset preprocessor, model wrapper from regression script
from regression_reindexing_synth import build_model, generate_sentences, preprocess_dataset


@pytest.fixture(scope="session")
def sentences():
    limit = 3
    return generate_sentences()[:limit]


@pytest.fixture(scope="session")
def soundness_dataset(sentences):
    return generator.sample_dataset(sentences)


def trace_conditional(conditioning: Dict, *args, **kwargs):
    """
    Evaluate model trace for given conditioning set.
    """
    conditioned_model = poutine.condition(build_model, conditioning)
    return poutine.trace(conditioned_model).get_trace(*args, **kwargs)


EPOCH_WINDOW = (-0.1, 2.9)
def model_forward(dataset, conditioning=None, **preprocess_args):
    if conditioning is None:
        conditioning = {}

    preprocess_args.setdefault("epoch_window", EPOCH_WINDOW)
    input_data = preprocess_dataset(dataset, **preprocess_args)

    model_trace = trace_conditional(
        conditioning,
        *input_data, sample_rate=dataset.sample_rate
    )

    return model_trace


def _run_soundness_check(conditions, background_condition,
                         dataset, **preprocess_args):
    """
    Verify that the probability of the ground truth data is greatest under the
    generating parameters (compared to perturbations thereof).
    """
    condition_logprobs = []
    for condition in conditions:
        condition.update(background_condition)
        ic(condition)
        log_joint = model_forward(dataset, condition, **preprocess_args).log_prob_sum()

        if log_joint.isinf():
            raise RuntimeError(str(condition))
        condition_logprobs.append(log_joint)

    from pprint import pprint
    result = sorted(zip(conditions, condition_logprobs),
                    key=lambda x: -x[1])
    pprint(result)

    assert result[0][0] == conditions[0], "Ground truth parameter is the MAP choice"


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

    gt_condition = {"coef": torch.tensor([1., -1])}
    alt_conditions = [{"coef": torch.tensor(x)}
                      for x in [[1., 1.],
                                [1., -2.],
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
