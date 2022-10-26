"""
Defines a simple three-parameter latent-onset model. Latent onset indices
are a deterministic function of data and these three parameters.
"""

from dataclasses import dataclass
from typing import NamedTuple, Tuple, Callable, List, Optional, Union

from icecream import ic
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torchtyping import TensorType
from typeguard import typechecked

from berp.datasets import BerpDataset
from berp.typing import Probability, is_probability, is_log_probability, is_positive, DIMS
from berp.util import sample_to_time, time_to_sample, variable_position_slice


# Define TensorType axis labels
B, N_C, N_P, N_F, N_F_T, V_P, T, S = \
    DIMS.B, DIMS.N_C, DIMS.N_P, DIMS.N_F, DIMS.N_F_T, DIMS.V_P, DIMS.T, DIMS.S


@dataclass
class ModelParameters:
    lambda_: TensorType[float]
    confusion: TensorType[V_P, V_P, float]
    threshold: TensorType[float]

    a: TensorType[float]
    b: TensorType[float]
    coef: TensorType[N_F, float]
    sigma: TensorType[float]


def PartiallyObservedModelParameters(*args, **kwargs):
    return ModelParameters(*args,
        a = torch.tensor(0.0),
        b = torch.tensor(0.0),
        coef = torch.tensor([0.0]),
        sigma = torch.tensor(0.0),
        **kwargs)


class RRResult(NamedTuple):

    params: ModelParameters
    dataset: BerpDataset

    q_pred: TensorType[B, T, S, float]
    q_obs: TensorType[B, T, S, float]


# @typechecked
def predictive_model(p_candidates: TensorType[B, N_C, is_log_probability],
                     phonemes: TensorType[B, N_C, N_P, int],
                     confusion: TensorType[V_P, V_P, is_probability],
                     lambda_: TensorType[float],
                     ground_truth_word_idx=0,
                     return_gt_only=True,
                     ) -> Union[TensorType[B, "num_phonemes_plus_one", is_probability],
                                TensorType[B, N_C, "num_phonemes_plus_one", is_probability]]:
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
    confusion = confusion.pow(1 / lambda_)
    confusion /= confusion.sum(dim=0, keepdim=True)

    # Compute likelihood for each candidate and each phoneme position.
    ground_truth_phonemes = phonemes[:, ground_truth_word_idx, :].unsqueeze(1)
    phoneme_likelihoods: TensorType[B, N_C, N_P, is_log_probability] = \
        confusion[phonemes, ground_truth_phonemes].log()
    incremental_word_likelihoods: TensorType[B, N_C, N_P, is_log_probability] = \
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


@typechecked
def recognition_point_model(p_candidates_posterior: TensorType[B, N_P, is_probability],
                            word_lengths: TensorType[B, torch.long, is_positive],
                            threshold: Probability
                            ) -> TensorType[B, torch.long]:
    """
    Computes the latent onset / recognition point for each example.
    """

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
    rec_point = pyro.deterministic("recognition_point",
                                   torch.minimum(rec_point, word_lengths))

    return rec_point


@typechecked
def recognition_points_to_times(recognition_points: TensorType[B, torch.long],
                                phoneme_onsets_global: TensorType[B, N_P, float],
                                phoneme_offsets_global: TensorType[B, N_P, float],
                                word_lengths: TensorType[B, int],
                                scatter_point: float = 0.0,
                                prior_scatter_index: int = 0,
                                prior_scatter_point: float = 0.0,
                                ) -> TensorType[B, float]:
    """
    Convert integer recognition point indices to continuous times using phoneme
    onset data. Configurably scatters recognition points within a phoneme's
    onset/offset duration.

    Also configurably scatters the edge case of recognition point == 0, indicating
    that the word was recognized prior to input.

    Args:
        recognition_points: As returned by `recognition_point_model`. One per word.
            Zero values indicate word is recognized prior to input onset; otherwise
            a value of `i` indicates recognition occurs at phoneme `i`,
            one-indexed.
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

    ret = torch.zeros_like(recognition_points, dtype=torch.float)
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


@typechecked
def scatter_response_model(
    X_variable: TensorType[B, N_F, float],
    X_ts: TensorType[T, N_F_T, float],
    recognition_points: TensorType[B, int],
    phoneme_onsets: TensorType[B, N_P, float],
    sample_rate: int,
    total_samples: int,
    ) -> TensorType[T, "n_total_features", float]:
    """
    Shallow response model which simply generates a design matrix
    to be fed to e.g. a receptive field estimator.
    Scatters word-level features onto their respective recognition onset point.

    In the case of two words with the same recognition point, their features are
    added in the resulting design matrix.
    """

    # Compute recognition onset times.
    recognition_onset = pyro.deterministic(
        "recognition_onset",
        torch.gather(phoneme_onsets, 1, recognition_points.unsqueeze(1)).squeeze(1))

    # Compute recognition onset as global index
    recognition_onsets_global_samp = time_to_sample(recognition_onset, sample_rate)

    # Scatter variable-onset data.
    ret_variable = torch.zeros(total_samples, X_variable.shape[1],
                               dtype=X_variable.dtype)
    for X_variable_i, rec_onset_samp in zip(X_variable, recognition_onsets_global_samp):
        ret_variable[rec_onset_samp] += X_variable_i

    # Concatenate with time-series data.
    ret = torch.concat([ret_variable, X_ts], dim=1)

    return ret


@typechecked
def general_response_model(
    X: TensorType[B, N_F, float],
    recognition_points: TensorType[B, int],
    phoneme_onsets: TensorType[B, N_P, float],
    Y: TensorType[T, S, float],
    irf_fn: Callable[[TensorType[N_F, float]],
                     Tuple[TensorType["irf_times", S, float],
                           TensorType["irf_times", S, bool]]],
    sample_rate: int,
    sigma: TensorType[float] = torch.tensor(0.1),
    ) -> Tuple[TensorType[T, S, float],
               TensorType[T, S, float]]:
    """
    Generate a response time series for the given stimulus and response parameters.
    Accepts an arbitrary impulse response function `irf_fn`, which accepts a feature
    vector and returns a tuple of a response time series and a mask over that time
    series. The mask indicates which regions of the time series are considered
    a commitment by the model; this response model will use the mask to assign log-prob
    only to the un-masked regions.
    """

    # Compute recognition onset times.
    recognition_onset = pyro.deterministic(
        "recognition_onset",
        torch.gather(phoneme_onsets, 1, recognition_points.unsqueeze(1)).squeeze(1))
    assert recognition_onset[2] == phoneme_onsets[2, recognition_points[2]]

    # Compute recognition onset as global index
    recognition_onsets_global_samp = time_to_sample(recognition_onset, sample_rate)

    # Generate continuous signal stream.
    Y_pred = torch.normal(0, sigma, size=Y.shape)
    # Which prediction indices do we actually mind?
    # (Don't penalize the model for making predictions outside of the mask
    # specified by the IRF.)
    mask = torch.zeros(*Y.shape, dtype=torch.bool)

    # Add characteristic response after each recognition onset.
    for X_i, rec_onset_samp in zip(X, recognition_onsets_global_samp):
        response_values, response_mask = irf_fn(X_i)
        start_idx = rec_onset_samp
        end_idx = start_idx + len(response_values)
        Y_pred[start_idx:end_idx] += response_values

        mask[start_idx:end_idx] = response_mask

    Y_obs = pyro.sample(
        "Y",
        dist.Normal(Y_pred, sigma),
        obs=Y,
        obs_mask=mask
    )

    return (Y_obs, Y_pred)


@typechecked
def response_model(X: TensorType[B, N_F, float],
                           coef: TensorType[N_F, float],
                           recognition_points: TensorType[B, int],
                           phoneme_onsets: TensorType[B, N_P, float],
                           Y: TensorType[B, T, S, float],
                           a: TensorType[float],
                           b: TensorType[float],
                           sample_rate: int,
                           epoch_window: Tuple[float, float],
                           sigma: TensorType[float] = torch.tensor(0.1),
                           predictive_distribution: str = "student_t",
                           sensor_reduction_fn=torch.mean
                           ) -> Tuple[TensorType[B, float], TensorType[B, float]]:
    """
    Computes the distribution over observable response to word $w_j$

        $$q ~ P(Y_j \\mid k_j, w_j)$$

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
        predictive_distribution: `student_t` or `normal`. If `normal`, outliers
            (values outside 95% percentile) are dropped from probability calculation

    Returns:
        q_obs:
        q_pred:
    """
    assert np.abs(Y.shape[1] - int(np.floor((epoch_window[1] - epoch_window[0]) * sample_rate))) <= 1

    # Compute recognition onset time.
    recognition_onset = pyro.deterministic(
        "recognition_onset",
        torch.gather(phoneme_onsets, 1, recognition_points.unsqueeze(1)).squeeze(1))
    assert recognition_onset[2] == phoneme_onsets[2, recognition_points[2]]

    # Expected response is `a` seconds later. Get the left edge of this window.
    recognition_erp_left_samp = time_to_sample(recognition_onset + a, sample_rate,
                                               epoch_window[0])

    slice_width = int(time_to_sample(b, sample_rate))
    Y_sliced, Y_mask = variable_position_slice(Y, recognition_erp_left_samp, slice_width)

    # Compute observed q.
    # Average over time, accounting for possibly variable length sequences.
    sample_counts = Y_mask.int().sum(dim=1)
    # print("sample_counts", sample_counts)
    q = Y_sliced.sum(dim=1, keepdim=True) / torch.maximum(sample_counts, torch.tensor(1))
    # Average over sensors.
    q = sensor_reduction_fn(q, dim=2, keepdim=True)
    q = q.squeeze()

    q_pred = pyro.deterministic("q_pred", torch.matmul(X, coef))
    # print("q_pred", q_pred[:5])
    # print("q", q[:5])

    if predictive_distribution == "student_t":
        q = pyro.sample("q", dist.StudentT(4, q_pred, sigma), obs=q)  # type: ignore
    elif predictive_distribution == "normal":
        # HACK: Drop regression outliers.
        # Ideally we'd 1) have a more robust regression or 2) account for correlations between
        # items.
        resids = (q - q_pred) ** 2
        q95 = torch.quantile(resids, 0.95)
        outlier_mask = resids > q95
        # Effectively drop outliers from probability estimate by assigning a high
        # predictive variance.
        sigma_vec = torch.ones_like(q) * sigma
        sigma_vec[outlier_mask] = 100.

    return (q, q_pred)


def scatter_model(params: ModelParameters,
                  dataset: BerpDataset
                  ) -> Tuple[ModelParameters, BerpDataset, TensorType[T, N_F, float]]:
    """
    Execute scatter model forward pass. Returns a design matrix to be fed
    to a receptive field estimator.
    """
    p_candidates_posterior = predictive_model(
        dataset.p_candidates, dataset.candidate_phonemes,
        params.confusion, params.lambda_)
    rec = recognition_point_model(
        p_candidates_posterior, dataset.word_lengths,
        params.threshold)
    scatter = scatter_response_model(
        X_variable=dataset.X_variable,
        X_ts=dataset.X_ts,
        recognition_points=rec,
        phoneme_onsets=dataset.phoneme_onsets + dataset.word_onsets.unsqueeze(1),
        sample_rate=dataset.sample_rate,
        total_samples=dataset.Y.shape[0],  # type: ignore
    )

    return params, dataset, scatter