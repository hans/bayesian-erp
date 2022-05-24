"""
Defines a simple three-parameter latent-onset model. Latent onset indices
are a deterministic function of data and these three parameters.
"""


from icecream import ic
import numpy as np
import pyro
import torch
from torchtyping import TensorType
from typeguard import typechecked

from berp.typing import Probability, is_probability, is_log_probability


# Define TensorType axis labels
B = "batch"
N_C = "n_candidate_words"
N_P = "n_phonemes"  # maximum length of observed word, in phonemes
V_P = "v_phonemes"  # size of phoneme vocabulary
T = "n_times"  # number of EEG samples
S = "n_sensors"  # number of EEG sensors


@typechecked
def predictive_model(p_word: TensorType[B, N_C],
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


def onset_model(p_word: TensorType[B, N_C],
                phonemes: TensorType[B, N_C, N_P, int],
                threshold: Probability
                ) -> TensorType[B, int]:
    """
    Computes the latent onset / recognition point for each example.
    """
    pass


def epoched_response_model(p_word: TensorType[B, is_log_probability],
                           onsets: TensorType[B, int],
                           Y: TensorType[B, T, S, float],
                           a: TensorType[float],
                           b: TensorType[float]
                           ) -> TensorType[B, float]:
    """
    Computes the distribution over observable response to word $w_j$

        $$P(Y_j \mid k_j, w_j)$$
    """
    pass


if __name__ == "__main__":
    b = 5
    n_c = 10
    n_p = 4
    v_p = 7

    confusion = torch.rand(v_p, v_p)
    confusion /= confusion.sum(axis=1)

    p_word = torch.rand(b, n_c)
    p_word /= p_word.sum(axis=1, keepdim=True)

    phonemes = torch.randint(0, v_p, (b, n_c, n_p))

    lambda_ = torch.tensor(1.)

    predictive_model(p_word, phonemes, confusion, lambda_)
