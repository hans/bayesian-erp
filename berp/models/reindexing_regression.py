"""
Defines a simple three-parameter latent-onset model. Latent onset indices
are a deterministic function of data and these three parameters.
"""


import pyro
import torch
from torchtyping import TensorType

from berp.typing import Probability, is_probability, is_log_probability


def predictive_model(p_word: TensorType["batch"],
                     phonemes: TensorType["batch", "n_phonemes", int],
                     confusion: TensorType["v_phonemes", "v_phonemes", is_probability],
                     lambda_: TensorType[float]
                     ) -> TensorType["batch", "n_phonemes"]:
    """
    Computes the next-word distribution

        $$P(w = w_j \mid w_{<j}, I_{\le k})$$

    for all $k$.

    This decomposes into a prior $P(w \mid w_{<j})$, derived from input language
    model probabilities, and a phoneme likelihood

        $$P(I_{\le k} \mid w = w_j)$$
    """
    pass


def onset_model(p_word: TensorType["batch"],
                phonemes: TensorType["batch", "n_phonemes", int],
                threshold: Probability
                ) -> TensorType["batch", int]:
    """
    Computes the latent onset / recognition point for each example.
    """
    pass


def epoched_response_model(p_word: TensorType["batch", is_log_probability],
                           onsets: TensorType["batch", int],
                           Y: TensorType["batch", "n_times", "n_sensors", float],
                           a: TensorType[float],
                           b: TensorType[float]
                           ) -> TensorType["batch", float]:
    """
    Computes the distribution over observable response to word $w_j$

        $$P(Y_j \mid k_j, w_j)$$
    """
    pass
