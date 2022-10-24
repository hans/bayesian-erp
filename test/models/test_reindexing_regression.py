from dataclasses import replace
from functools import partial
import re
from typing import Dict, Callable

from icecream import ic
import numpy as np
import pyro
import pytest
import torch

from berp.generators import stimulus
from berp.generators import thresholded_recognition_simple as generator
from berp.models import reindexing_regression as rr


def get_parameters():
    coef_mean = torch.tensor([0., -1.])

    # NB we establish deterministic sites below so that they can be conditioned
    return rr.ModelParameters(
        lambda_=torch.tensor(1.0),
        confusion=generator.phoneme_confusion,
        threshold=pyro.deterministic("threshold", torch.tensor(0.7)),
        a=pyro.deterministic("a", torch.tensor(0.4)),
        b=pyro.deterministic("b", torch.tensor(0.1)),
        coef=pyro.deterministic("coef", coef_mean),
        sigma=torch.tensor(1.0),
    )


def get_parameters2():
    coef_mean = torch.tensor([0., -1.])

    # NB we establish deterministic sites below so that they can be conditioned
    return rr.ModelParameters(
        lambda_=torch.tensor(1.0),
        confusion=generator.phoneme_confusion,
        threshold=pyro.deterministic("threshold", torch.tensor(0.7)),
        a=torch.tensor(0.4),
        b=torch.tensor(0.2),
        coef=pyro.deterministic("coef", coef_mean),
        sigma=torch.tensor(0.1),
    )


@pytest.fixture(scope="session")
def soundness_dataset1():
    stim = stimulus.RandomStimulusGenerator(phoneme_voc_size=len(generator.phoneme2idx))
    params = get_parameters()
    dataset = generator.sample_dataset(params, stim, response_type="square")
    print(dataset.Y.shape)
    return (dataset, params)


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
    phonemes = list("abcdefghijklmnopqrstuvwxyz_")
    stim = stimulus.NaturalLanguageStimulusGenerator(
        phonemes=phonemes,
        hf_model="hf-internal-testing/tiny-xlm-roberta")  # TODO use gpt2 instead?
    stim = partial(stim, sentences)
    params = get_parameters2()
    return (generator.sample_dataset(params, stim), params)


def model_forward(params, dataset):
    p_candidates_posterior = rr.predictive_model(
        dataset.p_candidates, dataset.candidate_phonemes,
        params.confusion, params.lambda_)
    rec_points = rr.recognition_point_model(
        p_candidates_posterior, dataset.word_lengths, params.threshold
    )
    return p_candidates_posterior, rec_points


def test_predictive_prior(soundness_dataset1):
    """
    Verify that the prior predictive is correctly integrated into the
    posterior predictive.
    """
    dataset, params = soundness_dataset1
    p_candidates_posterior, rec_points = model_forward(params, dataset)

    assert p_candidates_posterior.shape == (dataset.n_words, dataset.max_n_phonemes + 1,)

    # Posterior at k=0 should be the same as the ground truth word prior.
    torch.testing.assert_allclose(p_candidates_posterior[:, 0].log(), dataset.p_candidates[:, 0])


def test_recognition_edge_cases(soundness_dataset1):
    """
    Test recognition point inferences for literal edge cases, where no incremental
    posterior values pass threshold.
    """

    p_candidates_posterior = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ])

    word_lengths = torch.tensor([3, 2, 1])

    # Threshold is 0.5, so no recognition point.
    threshold = torch.tensor(0.5)

    rec_points = rr.recognition_point_model(p_candidates_posterior, word_lengths, threshold)
    # Recognition points should never exceed word length.
    torch.testing.assert_allclose(rec_points, torch.tensor([3, 2, 1]))


@pytest.fixture
def dummy_onsets():
    phoneme_onsets_global = torch.tensor(
       [[ 1.2879,  1.4607,  1.6087,  1.6087,  1.6087],                                                                                                                                                               
        [ 1.7484,  1.8518,  1.9802,  1.9802,  1.9802],                                                                                                                                                               
        [ 2.1424,  2.3097,  2.4258,  2.5167,  2.5167],                                                                                                                                                               
        [ 2.7228,  2.7228,  2.7228,  2.7228,  2.7228],                                                                                                                                                               
        [ 3.0338,  3.2246,  3.3319,  3.3319,  3.3319],                                                                                                                                                               
        [ 3.4800,  3.6465,  3.8436,  3.8436,  3.8436],                                                                                                                                                               
        [ 4.2200,  4.3135,  4.4947,  4.5873,  4.5873],                                                                                                                                                               
        [ 4.9134,  5.1128,  5.2600,  5.2600,  5.2600],                                                                                                                                                               
        [ 5.5410,  5.7359,  5.9243,  6.0812,  6.2804],                                                                                                                                                               
        [ 6.5199,  6.6307,  6.7280,  6.7280,  6.7280]])
    phoneme_offsets_global = torch.tensor(
       [[ 1.4607,  1.6087,  1.6087,  1.6087,  1.6087],
        [ 1.8518,  1.9802,  1.9802,  1.9802,  1.9802],
        [ 2.3097,  2.4258,  2.5167,  2.5167,  2.5167],
        [ 2.7228,  2.7228,  2.7228,  2.7228,  2.7228],
        [ 3.2246,  3.3319,  3.3319,  3.3319,  3.3319],
        [ 3.6465,  3.8436,  3.8436,  3.8436,  3.8436],
        [ 4.3135,  4.4947,  4.5873,  4.5873,  4.5873],
        [ 5.1128,  5.2600,  5.2600,  5.2600,  5.2600],
        [ 5.7359,  5.9243,  6.0812,  6.2804,  6.2804],
        [ 6.6307,  6.7280,  6.7280,  6.7280,  6.7280]])

    # TODO verify this is what we get from synth/real data, too.
    word_lengths = torch.tensor([2, 2, 3, 1, 2, 2, 2, 2, 4, 2])

    return phoneme_onsets_global, phoneme_offsets_global, word_lengths


def test_recognition_points_to_times(dummy_onsets):
    phoneme_onsets_global, phoneme_offsets_global, word_lengths = dummy_onsets

    recognition_points = torch.tensor([1, 0, 1, 0, 1, 1, 2])
    N = len(recognition_points)
    phoneme_onsets_global = phoneme_onsets_global[:N]
    phoneme_offsets_global = phoneme_offsets_global[:N]
    word_lengths = word_lengths[:N]

    def go(scatter_point, prior_scatter_point):
        return rr.recognition_points_to_times(
            recognition_points, phoneme_onsets_global, phoneme_offsets_global,
            word_lengths,
            scatter_point=scatter_point,
            prior_scatter_point=prior_scatter_point)

    times1 = go(0.0, (0, 0.0))
    torch.testing.assert_allclose(times1, torch.tensor([1.2879, 1.7484, 2.1424, 2.7228, 3.0338, 3.4800, 4.3135]))

    # test scatter = 1.0
    times2 = go(1.0, (0, 1.0))
    torch.testing.assert_allclose(times2, torch.tensor([1.4607, 1.8518, 2.3097, 2.7228, 3.2246, 3.6465, 4.4947]))

    times4 = go(0.5, (0, 0.0))
    torch.testing.assert_allclose(times4, torch.tensor([
        1.2879 + 0.5 * (1.4607 - 1.2879),
        1.7484, ## rec=0 so different scatter
        2.1424 + 0.5 * (2.3097 - 2.1424),
        2.7228, ## rec=0 so different scatter
        3.0338 + 0.5 * (3.2246 - 3.0338),
        3.4800 + 0.5 * (3.6465 - 3.4800),
        4.3135 + 0.5 * (4.4947 - 4.3135),
    ]))

    # test scatter prior at negative index
    times3 = go(0.0, (-1, 0.0))
    torch.testing.assert_allclose(times3, torch.tensor([
        1.2879,
        phoneme_onsets_global[0, word_lengths[0] - 1],
        2.1424,
        phoneme_onsets_global[2, word_lengths[2] - 1],
        3.0338, 3.4800, 4.3135]))

    # test scatter prior at negative index with partial scatter
    times5 = go(0.5, (-1, 0.5))
    torch.testing.assert_allclose(times5, torch.tensor([
        1.2879 + 0.5 * (1.4607 - 1.2879),
        phoneme_onsets_global[0, word_lengths[0] - 1] + 0.5 * (
            phoneme_offsets_global[0, word_lengths[0] - 1] - phoneme_onsets_global[0, word_lengths[0] - 1]),
        2.1424 + 0.5 * (2.3097 - 2.1424),
        phoneme_onsets_global[2, word_lengths[2] - 1] + 0.5 * (
            phoneme_offsets_global[2, word_lengths[2] - 1] - phoneme_onsets_global[2, word_lengths[2] - 1]),
        3.0338 + 0.5 * (3.2246 - 3.0338),
        3.4800 + 0.5 * (3.6465 - 3.4800),
        4.3135 + 0.5 * (4.4947 - 4.3135)]))


def test_recognition_logic(soundness_dataset1):
    """
    For possible thresholds T2 > T1, inferred recognition points K2, K1 should
    obey K2 >= K1.
    """
    dataset, params = soundness_dataset1

    t1 = params.threshold
    t2 = np.sqrt(params.threshold)
    assert t2 > t1

    _, rec_1 = model_forward(replace(params, threshold=t1), dataset)
    _, rec_2 = model_forward(replace(params, threshold=t2), dataset)

    ic(rec_2 - rec_1)
    ic(np.where(rec_2 < rec_1))
    assert bool((rec_2 >= rec_1).all()), "Recognition points should not decrease when threshold increases"

    # rec_onset_1 = trace_1.nodes["recognition_onset"]["value"]
    # rec_onset_2 = trace_2.nodes["recognition_onset"]["value"]
    # ic(rec_onset_2 - rec_onset_1)
    # ic(np.where(rec_onset_2 < rec_onset_1))
    # # NB if this assertion fails and the above passes, it's an error in the
    # # onset data representation / indexing logic.
    # assert bool((rec_onset_2 >= rec_onset_1).all()), "Recognition onsets should not decrease when threshold increases"
