"""
Defines a simple three-parameter latent-onset model. Latent onset indices
are a deterministic function of data and these three parameters.
"""

from typing import NamedTuple

from icecream import ic
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torchtyping import TensorType
from typeguard import typechecked

from berp.typing import Probability, is_probability, is_log_probability, DIMS
from berp.util import sample_to_time, time_to_sample, variable_position_slice


# Define TensorType axis labels
TT = TensorType
B, N_C, N_P, N_F, V_P, T, S = \
    DIMS.B, DIMS.N_C, DIMS.N_P, DIMS.N_F, DIMS.V_P, DIMS.T, DIMS.S


class ModelParameters(NamedTuple):
    lambda_: TT[float]
    confusion: TT[V_P, V_P, float]
    threshold: TT[float]

    a: TT[float]
    b: TT[float]
    coef: TT[N_F, float]


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

    # Temperature adjustment
    confusion = confusion.pow(1 / lambda_)
    confusion /= confusion.sum(dim=0, keepdim=True)

    # Compute likelihood for each candidate and each phoneme position.
    ground_truth_phonemes = phonemes[:, ground_truth_word_idx, :].unsqueeze(1)
    phoneme_likelihoods: TensorType[B, N_C, N_P, is_log_probability] = \
        confusion[phonemes, ground_truth_phonemes].log()
    incremental_word_likelihoods: TensorType[B, N_C, N_P, is_log_probability] = \
        phoneme_likelihoods.cumsum(axis=2)

    # Combine with prior and normalize.
    bayes_p_word = (p_word.unsqueeze(-1) + phoneme_likelihoods).exp()
    bayes_p_word /= bayes_p_word.sum(dim=1, keepdim=True)

    p_ground_truth = bayes_p_word[:, ground_truth_word_idx, :]
    return p_ground_truth


@typechecked
def recognition_point_model(p_word_posterior: TensorType[B, N_P, is_probability],
                            word_lengths: TensorType[B, torch.long],
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
    rec_point = passes_threshold.int().argmax(dim=1)

    # Don't allow recognition point to go past final ground-truth phoneme.
    rec_point = pyro.deterministic("recognition_point",
                                   torch.minimum(rec_point, word_lengths - 1))

    return rec_point


@typechecked
def epoched_response_model(X: TensorType[B, N_F, float],
                           coef: TensorType[N_F, float],
                           recognition_points: TensorType[B, int],
                           phoneme_onsets: TensorType[B, N_P, float],
                           Y: TensorType[B, T, S, float],
                           a: TensorType[float],
                           b: TensorType[float],
                           sample_rate: int,
                           sigma: TensorType[float] = torch.tensor(1.),
                           sensor_reduction_fn=torch.mean
                           ) -> TensorType[B, float]:
    """
    Computes the distribution over observable response to word $w_j$

        $$q ~ P(Y_j \mid k_j, w_j)$$

    Args:
        X: word-level features describing each word
        coef: Linear regression coefficients for each feature
        recognition_points:
        phoneme_onsets: Onset (in seconds, relative to t = sample = 0 in `Y`)
            of each phoneme of each example.
        a: Test window offset from recognition point, in seconds.
        b: Test window width, in seconds.
        sample_rate: Number of samples per second
        sigma: Standard deviation parameter for observations
    """
    # Compute start of range slice into Y for each example.
    # print("recognition_points", recognition_points)
    recognition_onset = pyro.deterministic(
        "recognition_onset",
        torch.gather(phoneme_onsets, 1, recognition_points.unsqueeze(1)).squeeze(1))
    assert recognition_onset[2] == phoneme_onsets[2, recognition_points[2]]

    recognition_onset_samp = time_to_sample(recognition_onset,
                                            sample_rate)

    slice_width = int(time_to_sample(b, sample_rate))
    Y_sliced, Y_mask = variable_position_slice(Y, recognition_onset_samp, slice_width)

    # Compute observed q.
    # Average over time, accounting for possibly variable length sequences.
    sample_counts = Y_mask.int().sum(dim=1)
    # print("sample_counts", sample_counts)
    q = Y_sliced.sum(dim=1, keepdim=True) / torch.maximum(sample_counts, torch.tensor(1))
    # Average over sensors.
    q = sensor_reduction_fn(q, dim=2, keepdim=True)
    q = q.squeeze()

    q_pred = torch.matmul(X, coef)
    # print("q_pred", q_pred)
    # print("q", q)
    return pyro.sample("q", dist.Normal(q_pred, sigma),
                       obs=q)


if __name__ == "__main__":
    batch = 5
    n_c = 10
    n_p = 4
    v_p = 7

    t = 100
    s = 2

    confusion = torch.rand(v_p, v_p)
    confusion /= confusion.sum(dim=1)

    p_word = torch.rand(batch, n_c)
    p_word /= p_word.sum(dim=1, keepdim=True)
    p_word_ground_truth = p_word[:, 0]

    phonemes = torch.randint(0, v_p, (batch, n_c, n_p))
    phoneme_onsets = torch.rand(batch, n_p).cumsum(dim=1)

    lambda_ = torch.tensor(1.)
    threshold = torch.tensor(0.15)

    Y = torch.rand(batch, t, s)
    sample_rate = torch.tensor(32)

    a = torch.tensor(0.1)
    b = torch.tensor(0.2)


    p_word_posterior = predictive_model(p_word.log(), phonemes, confusion, lambda_)
    rec = recognition_point_model(p_word_posterior, phonemes, confusion, lambda_, threshold)
    epoched_response_model(rec, phoneme_onsets, Y, a, b, sample_rate)
