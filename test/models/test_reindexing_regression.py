from pathlib import Path

from icecream import ic
import pyro
from pyro import poutine
import pytest
import torch

from berp.generators import thresholded_recognition as generator

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))
print(sys.path)
# steal dataset preprocessor, model wrapper from regression script
from regression_reindexing_synth import build_model, generate_sentences, preprocess_dataset


@pytest.fixture(scope="session")
def sentences():
    limit = 3
    return generate_sentences()[:limit]


@pytest.fixture(scope="session")
def soundness_dataset(sentences):
    return generator.sample_dataset(sentences)


EPOCH_WINDOW = (-0.1, 2.9)
def model_logprob(dataset, conditioning=None, **preprocess_args):
    preprocess_args.setdefault("epoch_window", EPOCH_WINDOW)
    input_data = preprocess_dataset(dataset, **preprocess_args)

    conditioned_model = poutine.condition(build_model, conditioning)
    model_trace = poutine.trace(conditioned_model).get_trace(
        *input_data, sample_rate=dataset.sample_rate
    )

    log_joint = model_trace.log_prob_sum()
    return log_joint


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
        log_joint = model_logprob(dataset, condition, **preprocess_args)

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
