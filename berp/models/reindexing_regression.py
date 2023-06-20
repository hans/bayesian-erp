"""
Defines a simple three-parameter latent-onset model. Latent onset indices
are a deterministic function of data and these three parameters.
"""

from dataclasses import dataclass
from typing import NamedTuple, Tuple, Callable, List, Optional, Union
from typing_extensions import TypeAlias

from icecream import ic
from jaxtyping import Float, Integer, UInt64, UInt
import numpy as np
import torch
from typeguard import typechecked

from berp.datasets import BerpDataset
from berp.util import sample_to_time, time_to_sample


T: TypeAlias = torch.Tensor
FloatScalar: TypeAlias = Float[T, ""]
Probability: TypeAlias = FloatScalar
ConfusionMatrix: TypeAlias = Float[T, "vocab_phonemes vocab_phonemes"]


@dataclass
class ModelParameters:
    lambda_: FloatScalar
    confusion: ConfusionMatrix
    threshold: FloatScalar

    a: FloatScalar
    b: FloatScalar
    coef: Float[T, "num_variable_features"]
    sigma: FloatScalar


def PartiallyObservedModelParameters(*args, **kwargs):
    kwargs.setdefault("lambda_", torch.tensor(1.0))
    kwargs.setdefault("confusion", torch.tensor(1.0))
    kwargs.setdefault("threshold", torch.tensor(0.75))

    return ModelParameters(*args,
        a = torch.tensor(0.0),
        b = torch.tensor(0.0),
        coef = torch.tensor([0.0]),
        sigma = torch.tensor(0.0),
        **kwargs)


# @typechecked
def predictive_model(p_candidates: Float[T, "batch num_candidates"],
                     phonemes: UInt64[T, "batch num_candidates num_phonemes"],
                     confusion: ConfusionMatrix,
                     lambda_: Float[T, ""],
                     ground_truth_word_idx=0,
                     return_gt_only=True,
                     ) -> Union[Float[T, "num_phonemes+1"],
                                Float[T, "batch num_candidates num_phonemes+1"]]:
    r"""
    Computes the next-word probability estimate

        $$P(w = w_j \mid w_{<j}, I_{\le k})$$

    for all $k$, including the initial case when there is no input.

    This decomposes into a prior $P(w \mid w_{<j})$, derived from input language
    model probabilities, and a phoneme likelihood

        $$P(I_{\le k} \mid w = w_j)$$

    In the initial case when there is no input, this posterior is equivalent to the
    prior alone.

    Args:
        p_candidates: Next-word predictive distribution from a language model for a
            limited set of top-k candidates `n_candidate_words`. Column axis
            should be a proper distribution.
        phonemes: Phoneme sequence for all examples and all candidate words.
            Each element is an index into a row/column of `confusion`.
        confusion: Confusion parameter matrix used to define likelihood.
            Each column is a proper probability distribution, defining
            probability of observing phone j given ground truth phone i.
        lambda_: Temperature parameter for likelihood.
        ground_truth_word_idx: Specifies an index into second axis of `p_candidates`
            and `phonemes` that corresponds to the ground truth word.

    Returns:
        `batch * n_phonemes + 1` probability values, defining the next-word
        posterior probability evaluated for each example at each conditioning
        point. Column `i` corresponds to next-word posterior probability after
        observing `i` phonemes (so column 0 corresponds to zero input, column 1
        corresponds to first phoneme input, and so on).
    """

    # Temperature adjustment
    confusion = confusion.to(p_candidates).pow(1 / lambda_)
    confusion /= confusion.sum(dim=0, keepdim=True)

    # Compute likelihood for each candidate and each phoneme position.
    ground_truth_phonemes = phonemes[:, ground_truth_word_idx, :].unsqueeze(1)
    phoneme_likelihoods: Float[T, "batch num_candidates num_phonemes"] = \
        confusion[phonemes, ground_truth_phonemes].log()
    incremental_word_likelihoods: Float[T, "batch num_candidates num_phonemes"] = \
        phoneme_likelihoods.cumsum(dim=2)

    # Add an initial column of zeros corresponding to likelihood prior to input.
    incremental_word_likelihoods = torch.cat(
        [torch.zeros_like(incremental_word_likelihoods[:, :, :1]),
         incremental_word_likelihoods], dim=2)

    # Combine with prior and normalize.
    bayes_p_candidates = (p_candidates.unsqueeze(-1) + incremental_word_likelihoods).exp()
    bayes_p_candidates /= bayes_p_candidates.sum(dim=1, keepdim=True)

    if return_gt_only:
        p_ground_truth = bayes_p_candidates[:, ground_truth_word_idx, :]
        return p_ground_truth
    else:
        return bayes_p_candidates


# @typechecked
def recognition_point_model(p_candidates_posterior: Float[T, "batch num_phonemes+1"],
                            word_lengths: UInt64[T, "batch"],
                            threshold: Probability
                            ) -> UInt64[T, "batch"]:
    """
    Computes the recognition point of the ground-truth word for each example,
    given incremental posterior probabilities over the ground-truth word as
    returned by `predictive_model`.

    If a particular example never exceeds the given threshold, the recognition
    point is set to the final phoneme in the ground-truth word.

    Returns:
        Recognition point for each example, where `0` corresponds to recognition
        without perceptual input, and a value `i` corresponds to recognition after
        observing `i` phonemes.
    """

    assert p_candidates_posterior.shape[1] >= word_lengths.max() + 1

    passes_threshold = p_candidates_posterior >= threshold

    # Find first phoneme index for which predictive distribution passes
    # threshold.
    rec_point = passes_threshold.int().argmax(dim=1).long()

    # Special case: no phoneme passes threshold. We then say recognition happens
    # at the final phoneme in the ground truth word.
    # This is not correct, but nothing else is feasible with the given setup.
    valid_point = passes_threshold.any(dim=1).long()
    rec_point = valid_point * rec_point + (1 - valid_point) * word_lengths

    # Don't allow recognition point to go past final ground-truth phoneme.
    rec_point = torch.minimum(rec_point, word_lengths)

    return rec_point


# @typechecked
def recognition_points_to_times(recognition_points: UInt64[T, "batch"],
                                phoneme_onsets_global: Float[T, "batch num_phonemes"],
                                phoneme_offsets_global: Float[T, "batch num_phonemes"],
                                word_lengths: UInt[T, "batch"],
                                scatter_point: float = 0.0,
                                prior_scatter_index: int = 0,
                                prior_scatter_point: float = 0.0,
                                ) -> Float[T, "batch"]:
    """
    Convert integer recognition point indices to continuous times using phoneme
    onset data. Configurably scatters recognition points within a phoneme's
    onset/offset duration.

    Also configurably scatters the edge case of recognition point == 0, indicating
    that the word was recognized prior to input.

    Args:
        recognition_points: As returned by `recognition_point_model`. One per word.
            Zero values indicate word is recognized prior to input onset; otherwise
            a value of `i` indicates recognition occurs after consuming `i` phonemes.
        phoneme_onsets_global: See `BerpDataset`.
        phoneme_offsets_global: See `BerpDataset`.
        word_lengths: Length of each ground-truth word, in phonemes.
        scatter_point:  If a word is recognized at phoneme p_i
            which has onset time t_i and offset time t_{i+1}, then declare
            that the word's recognition point is
            $$t_i + (t_{i+1} - t_i) * recognition_scatter_point$$
            or equivalently
            $$recognition_scatter_point * t_{i+1} + (1 - recognition_scatter_point) * t_i$$
        prior_scatter_index: If a word is recognized prior to any perceptual
            input, index its recognition point onto the given phoneme.
            The phoneme index can be negative! For example, a value
            of `-1` indicates that recognition should be indexed into the
            onset/offset window of the final phoneme of the PREVIOUS
            word. `0` indicates that recognition should be indexed to onset/offset
            window of the first phoneme of the target word.
        prior_scatter_point: If a word is recognized prior to any perceptual
            input, index its recognition point at this fraction progress
            through the phoneme onset/offset window specified by by
            `prior_scatter_index`.
            This spec overrides any setting of `scatter_point`.
    """
    if scatter_point < 0.0 or scatter_point > 1.0:
        raise ValueError("Scatter point must be in [0, 1]")
    if prior_scatter_point < 0.0 or prior_scatter_point > 1.0:
        raise ValueError("Prior scatter point must be in [0, 1]")

    # All indexing will happen on a flattened array, in order to easily support
    # indexing across words.
    # But do this sensitive to word lengths so that we aren't indexing over padding
    # elements.
    phoneme_onsets_flat = torch.cat([
        ons[:word_lengths[i]]
        for i, ons in enumerate(phoneme_onsets_global)
    ])
    phoneme_offsets_flat = torch.cat([
        offs[:word_lengths[i]]
        for i, offs in enumerate(phoneme_offsets_global)
    ])
    assert len(phoneme_onsets_flat) == word_lengths.sum()
    
    # Convert recognition points to flat indices.
    rec_point_flat = recognition_points + word_lengths.cumsum(dim=0) - word_lengths
    assert rec_point_flat[0] == recognition_points[0]

    ret = torch.zeros_like(recognition_points).to(phoneme_offsets_flat)
    switch = recognition_points == 0

    # Handle edge case: recognition point is zero, indicating recognition pre input
    # onset.
    # Anchor phoneme: index of phoneme for whose window we'll do scattering.
    anchor_phoneme_idx = rec_point_flat[switch] + prior_scatter_index
    # Avoid index errors at start and end of time series.
    anchor_phoneme_idx = torch.clamp(anchor_phoneme_idx, 0, len(phoneme_onsets_flat) - 1)
    ret[switch] = prior_scatter_point * phoneme_offsets_flat[anchor_phoneme_idx] + \
        (1 - prior_scatter_point) * phoneme_onsets_flat[anchor_phoneme_idx]

    # Handle normal case: recognition point is non-zero, indicating recognition
    # at a given phoneme.
    # NB we subtract 1 from recognition point because we want to index into the
    # phoneme onset/offset arrays, which are zero-indexed.
    ret[~switch] = scatter_point * phoneme_offsets_flat[rec_point_flat[~switch] - 1] + \
        (1 - scatter_point) * phoneme_onsets_flat[rec_point_flat[~switch] - 1]

    return ret