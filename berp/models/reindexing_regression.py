"""
Defines a simple three-parameter latent-onset model. Latent onset indices
are a deterministic function of data and these three parameters.
"""

from typing import NamedTuple, Tuple, Callable, List, Optional

from icecream import ic
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from torchtyping import TensorType
from typeguard import typechecked

from berp.typing import Probability, is_probability, is_log_probability, is_positive, DIMS
from berp.util import sample_to_time, time_to_sample, variable_position_slice


# Define TensorType axis labels
B, N_C, N_P, N_F, V_P, T, S = \
    DIMS.B, DIMS.N_C, DIMS.N_P, DIMS.N_F, DIMS.V_P, DIMS.T, DIMS.S


class ModelParameters(NamedTuple):
    lambda_: TensorType[float]
    confusion: TensorType[V_P, V_P, float]
    threshold: TensorType[float]

    a: TensorType[float]
    b: TensorType[float]
    coef: TensorType[N_F, float]
    sigma: TensorType[float]


class RRDataset(NamedTuple):
    """
    Defines an epoched dataset for reindexing regression, generated with the given
    ground-truth parameters ``params``.

    Each element on the batch axis corresponds to a single word within a single
    item.

    All tensors are padded on the N_P axis on the right to the maximum word length.
    """

    params: ModelParameters
    sample_rate: int
    epoch_window: Tuple[float, float]

    phonemes: List[str]
    """
    Phoneme vocabulary.
    """

    p_word: TensorType[B, N_C, is_log_probability]
    """
    Predictive distribution over expected candidate words at each time step,
    derived from a language model.
    """

    word_lengths: TensorType[B, int]
    """
    Length of ground-truth words in phonemes. Can be used to unpack padded
    ``N_P`` axes.
    """

    candidate_phonemes: TensorType[B, N_C, N_P, int]
    """
    Phoneme ID sequence for each word and alternate candidate set.
    """

    word_onsets: TensorType[B, float]
    """
    Onset of each word in seconds, relative to the start of the sequence.
    """

    phoneme_onsets: TensorType[B, N_P, float]
    """
    Onset of each phoneme within each word in seconds, relative to the start of
    the corresponding word.
    """

    X_epoch: TensorType[B, N_F, float]
    """
    Epoch features.
    """

    Y_epoch: TensorType[B, T, S, float]
    """
    Epoched response data (raw; not baselined).
    """

    Y: Optional[TensorType[..., S, float]] = None
    """
    Original response data stream.
    """

    recognition_points: Optional[TensorType[B, int]] = None
    """
    Ground-truth recognition points (phoneme indices) for each word.
    Useful for debugging.
    """

    recognition_onsets: Optional[TensorType[B, float]] = None
    """
    Ground-truth recognition onset (seconds, relative to word onset) for each
    word. Useful for debugging.
    """


class RRResult(NamedTuple):

    params: ModelParameters
    dataset: RRDataset

    q_pred: TensorType[B, T, S, float]
    q_obs: TensorType[B, T, S, float]


@typechecked
def predictive_model(p_word: TensorType[B, N_C, is_log_probability],
                     phonemes: TensorType[B, N_C, N_P, int],
                     confusion: TensorType[V_P, V_P, is_probability],
                     lambda_: TensorType[float],
                     ground_truth_word_idx=0
                     ) -> TensorType[B, N_P, is_probability]:
    r"""
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
        phoneme_likelihoods.cumsum(dim=2)

    # Combine with prior and normalize.
    bayes_p_word = (p_word.unsqueeze(-1) + incremental_word_likelihoods).exp()
    bayes_p_word /= bayes_p_word.sum(dim=1, keepdim=True)

    p_ground_truth = bayes_p_word[:, ground_truth_word_idx, :]
    return p_ground_truth


@typechecked
def recognition_point_model(p_word_posterior: TensorType[B, N_P, is_probability],
                            word_lengths: TensorType[B, torch.long, is_positive],
                            threshold: Probability
                            ) -> TensorType[B, int]:
    """
    Computes the latent onset / recognition point for each example.
    """

    passes_threshold = p_word_posterior >= threshold

    # Find first phoneme index for which predictive distribution passes
    # threshold.
    rec_point = passes_threshold.int().argmax(dim=1)

    # Special case: no phoneme passes threshold. We then call the recognition
    # point the n+1 th phoneme in our representation.
    valid_point = passes_threshold.any(dim=1).int()
    rec_point = valid_point * rec_point + (1 - valid_point) * (word_lengths - 1)
    # print("recognition_pct", rec_point / word_lengths)

    # Don't allow recognition point to go past final ground-truth phoneme.
    rec_point = pyro.deterministic("recognition_point",
                                   torch.minimum(rec_point, word_lengths - 1))

    return rec_point


@typechecked
def scatter_response_model(
    X: TensorType[B, N_F, float],
    recognition_points: TensorType[B, int],
    phoneme_onsets: TensorType[B, N_P, float],
    sample_rate: int,
    total_samples: int,
    ) -> TensorType[T, N_F, float]:
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
    assert recognition_onset[2] == phoneme_onsets[2, recognition_points[2]]

    # Compute recognition onset as global index
    recognition_onsets_global_samp = time_to_sample(recognition_onset, sample_rate)

    num_features = X.shape[1]
    ret = torch.zeros(total_samples, num_features, dtype=X.dtype)

    for X_i, rec_onset_samp in zip(X, recognition_onsets_global_samp):
        ret[rec_onset_samp] += X_i

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
                  dataset: RRDataset
                  ) -> Tuple[ModelParameters, RRDataset, TensorType[T, N_F, float]]:
    """
    Execute scatter model forward pass. Returns a design matrix to be fed
    to a receptive field estimator.
    """
    p_word_posterior = predictive_model(
        dataset.p_word, dataset.candidate_phonemes,
        params.confusion, params.lambda_)
    rec = recognition_point_model(
        p_word_posterior, dataset.word_lengths,
        params.threshold)
    scatter = scatter_response_model(
        X=dataset.X_epoch,
        recognition_points=rec,
        phoneme_onsets=dataset.phoneme_onsets + dataset.word_onsets.unsqueeze(1),
        sample_rate=dataset.sample_rate,
        total_samples=dataset.Y.shape[0],  # type: ignore
    )

    return params, dataset, scatter


# def model_wrapped(params: Callable[[], ModelParameters],
#                   dataset: RRDataset) -> RRResult:
#     """
#     Execute full forward model, wrapped in a function that allows for
#     parameterization.
#     """
#     return model(params(), dataset)