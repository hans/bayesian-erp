"""
Defines a simple three-parameter latent-onset model. Latent onset indices
are a deterministic function of data and these three parameters.
"""


from icecream import ic
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torchtyping import TensorType
from typeguard import typechecked

from berp.typing import Probability, is_probability, is_log_probability
from berp.util import sample_to_time, time_to_sample


# Define TensorType axis labels
B = "batch"
N_C = "n_candidate_words"
N_P = "n_phonemes"  # maximum length of observed word, in phonemes
V_P = "v_phonemes"  # size of phoneme vocabulary
T = "n_times"  # number of EEG samples
S = "n_sensors"  # number of EEG sensors


@typechecked
def predictive_model(p_word: TensorType[B, N_C, is_log_probability],
                     phonemes: TensorType[B, N_C, N_P, int],
                     confusion: TensorType[V_P, V_P, is_probability],
                     lambda_: TensorType[float],
                     ground_truth_word_idx=0
                     ) -> TensorType[B, N_P, is_probability]:
    """
    Computes the next-word distribution

        $$P(w = w_j \mid w_{<j}, I_{\le k})$$

    for all $k$.

    This decomposes into a prior $P(w \mid w_{<j})$, derived from input language
    model probabilities, and a phoneme likelihood

        $$P(I_{\le k} \mid w = w_j)$$

    Args:
        p_word: Next-word predictive distribution from a language model for a
            limited set of top-k candidates `n_candidate_words`. Column axis
            should be a proper distribution.
        phonemes: Phoneme sequence for all examples and all candidate words.
            Each element is an index into a row/column of `confusion`.
        confusion: Confusion parameter matrix used to define likelihood.
            Each column is a proper probability distribution, defining
            probability of observing phone j given ground truth phone i.
        lambda_: Temperature parameter for likelihood.
        ground_truth_word_idx: Specifies an index into second axis of `p_word`
            and `phonemes` that corresponds to the ground truth word.

    Returns:
        `batch * n_phonemes` log-probability matrix, defining the next-word
        distribution evaluated for each example at each conditioning point.
    """

    # Compute likelihood for each candidate and each phoneme position.
    ground_truth_phonemes = phonemes[:, ground_truth_word_idx, :].unsqueeze(1)
    phoneme_likelihoods: TensorType[B, N_C, N_P, is_log_probability] = \
        confusion[phonemes, ground_truth_phonemes].log()
    incremental_word_likelihoods: TensorType[B, N_C, N_P, is_log_probability] = \
        phoneme_likelihoods.cumsum(axis=2)

    # Combine with prior and normalize.
    bayes_p_word = (p_word.unsqueeze(-1) + phoneme_likelihoods).exp()
    bayes_p_word /= bayes_p_word.sum(axis=1, keepdim=True)

    p_ground_truth = bayes_p_word[:, ground_truth_word_idx, :]
    return p_ground_truth


def recognition_point_model(p_word_posterior: TensorType[B, N_P, is_probability],
                            phonemes: TensorType[B, N_C, N_P, int],
                            confusion: TensorType[V_P, V_P, is_probability],
                            lambda_: TensorType[float],
                            threshold: Probability
                            ) -> TensorType[B, int]:
    """
    Computes the latent onset / recognition point for each example.
    """
    passes_threshold = p_word_posterior >= threshold

    # Find first phoneme index for which predictive distribution passes
    # threshold.
    rec_point = passes_threshold.int().argmax(axis=1)
    return rec_point


@typechecked
def epoched_response_model(recognition_points: TensorType[B, int],
                           phoneme_onsets: TensorType[B, N_P, float],
                           Y: TensorType[B, T, S, float],
                           a: TensorType[float],
                           b: TensorType[float],
                           sample_rate: TensorType[int],
                           sigma: TensorType[float] = torch.tensor(1.),
                           time_reduction_fn=torch.mean,
                           sensor_reduction_fn=torch.mean
                           ) -> TensorType[B, float]:
    """
    Computes the distribution over observable response to word $w_j$

        $$q ~ P(Y_j \mid k_j, w_j)$$

    Args:
        recognition_points:
        phoneme_onsets: Onset (in seconds, relative to t = sample = 0 in `Y`)
            of each phoneme of each example.
        a: Test window offset from recognition point, in seconds.
        b: Test window width, in seconds.
        sample_rate: Number of samples per second
        sigma: Standard deviation parameter for observations
    """
    # Compute start of range slice into Y for each example.
    recognition_onset = torch.gather(phoneme_onsets, 1, recognition_points.unsqueeze(1)).squeeze(1)
    assert recognition_onset[2] == phoneme_onsets[2, recognition_points[2]]
    recognition_onset_samp = time_to_sample(recognition_onset,
                                            sample_rate)

    slice_width = time_to_sample(b, sample_rate)
    # TODO there must be a cleaner way to do this with slicing?
    # Generate sample index range for each example.
    slice_idxs = torch.arange(slice_width).tile((recognition_onset.shape[0], 1)) \
        + recognition_onset_samp.unsqueeze(1)
    # Now tile indices across sensors.
    # TODO is this necessary? I want to use slice `:`
    n_sensors = Y.shape[2]
    slice_idxs = slice_idxs.unsqueeze(2).tile((1, 1, n_sensors))

    # Gather.
    Y_sliced = torch.gather(Y, 1, slice_idxs)
    assert Y_sliced[2, 0, 1] == Y[2, slice_idxs[2, 0, 1], 1]

    q = time_reduction_fn(Y_sliced, axis=1, keepdim=True)
    q = sensor_reduction_fn(q, axis=2, keepdim=True)
    q = q.squeeze()
    ic(q.shape)

    return pyro.sample("q", dist.Normal(q, sigma))


if __name__ == "__main__":
    b = 5
    n_c = 10
    n_p = 4
    v_p = 7

    t = 100
    s = 2

    confusion = torch.rand(v_p, v_p)
    confusion /= confusion.sum(axis=1)

    p_word = torch.rand(b, n_c)
    p_word /= p_word.sum(axis=1, keepdim=True)
    p_word_ground_truth = p_word[:, 0]

    phonemes = torch.randint(0, v_p, (b, n_c, n_p))
    phoneme_onsets = torch.rand(b, n_p).cumsum(axis=1)

    lambda_ = torch.tensor(1.)
    threshold = torch.tensor(0.15)

    Y = torch.rand(b, t, s)
    sample_rate = torch.tensor(32)

    a = torch.tensor(0.1)
    b = torch.tensor(0.2)


    p_word_posterior = predictive_model(p_word.log(), phonemes, confusion, lambda_)
    rec = recognition_point_model(p_word_posterior, phonemes, confusion, lambda_, threshold)
    epoched_response_model(rec, phoneme_onsets, Y, a, b, sample_rate)
