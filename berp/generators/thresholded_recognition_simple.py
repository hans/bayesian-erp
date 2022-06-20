from typing import List, NamedTuple, Tuple

import numpy as np
import pyro.distributions as dist
import torch
from torch.nn.functional import pad
from torchtyping import TensorType
from tqdm.notebook import tqdm
from typeguard import typechecked

from berp.models import reindexing_regression as rr
from berp.typing import DIMS, is_log_probability
from berp.util import sample_to_time, time_to_sample, gaussian_window

PHONEMES = np.array(list("abcdefghijklmnop_"))
phoneme2idx = {p: idx for idx, p in enumerate(PHONEMES)}
def random_word(length):
    return np.random.choice(PHONEMES, size=length)

phoneme_confusion = torch.diag(torch.ones(len(PHONEMES))) + \
    0.1 * torch.rand(len(PHONEMES), len(PHONEMES))
phoneme_confusion /= phoneme_confusion.sum(dim=0, keepdim=True)


# Type variables
B, N_W, N_C, N_F, N_P, V_W = DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_P, DIMS.V_W
T, S = DIMS.T, DIMS.S


def rand_unif(low, high, *shape) -> torch.Tensor:
    return torch.rand(*shape) * (high - low) + low


class Stimulus(NamedTuple):

    word_lengths: TensorType[N_W, int]
    phoneme_onsets: TensorType[N_W, N_P, float]
    phoneme_onsets_global: TensorType[N_W, N_P, float]
    word_onsets: TensorType[N_W, float]
    word_surprisals: TensorType[N_W, float]
    p_word: TensorType[N_W, N_C, float]
    candidate_phonemes: TensorType[N_W, N_C, N_P, int]


@typechecked
def sample_stimulus(num_words: int = 100,
                   num_candidates: int = 10,
                   num_phonemes: int = 5,
                   phon_delay_range: Tuple[float, float] = (0.04, 0.1),
                   word_delay_range: Tuple[float, float] = (0.01, 0.1),
                   word_surprisal_params: Tuple[float, float] = (1., 0.5),
                   first_onset=1.0,
                   # TODO rm first_onset silliness
                   ) -> Stimulus:
    word_lengths = torch.tensor([num_phonemes for _ in range(num_words)])

    phoneme_onsets = rand_unif(*phon_delay_range, num_words, num_phonemes)
    phoneme_onsets[:, 0] = 0.
    phoneme_onsets = phoneme_onsets.cumsum(1)
    word_delays = rand_unif(*word_delay_range, num_words)
    word_onsets = (torch.cat([torch.tensor([first_onset]),
                              first_onset + phoneme_onsets[1:, -1]])
                              + word_delays).cumsum(0)
    # Make phoneme_onsets global (not relative to word onset).
    phoneme_onsets_global = phoneme_onsets + word_onsets.view(-1, 1)

    word_surprisals: torch.Tensor = dist.LogNormal(*word_surprisal_params).sample((num_words,))  # type: ignore

    # Calculate p_word using surprisal; allocate remainder randomly
    p_gt_word = (-word_surprisals).exp()
    remainder = 1 - p_gt_word
    p_candidates = (remainder / (num_candidates - 1)).view(-1, 1) \
        * torch.ones(num_words, num_candidates - 1)
    p_word = torch.cat([p_gt_word.view(-1, 1), p_candidates], dim=1) \
        .log()

    candidate_phonemes = torch.randint(0, len(PHONEMES) - 2,
                                       (num_words, num_candidates, num_phonemes))

    return Stimulus(word_lengths, phoneme_onsets, phoneme_onsets_global,
                    word_onsets, word_surprisals, p_word, candidate_phonemes)


@typechecked
def sample_dataset(params: rr.ModelParameters,
                   num_sensors: int = 1,
                   sample_rate: int = 128,
                   epoch_window: Tuple[float, float] = (-0.1, 1.0),
                   noise_params: Tuple[float, float] = (0., 0.5),
                   **stimulus_params,
                   ) -> rr.RRDataset:
    
    stim = sample_stimulus(**stimulus_params)

    ############

    p_word_posterior = rr.predictive_model(stim.p_word, stim.candidate_phonemes,
                                           confusion=params.confusion,
                                           lambda_=params.lambda_)
    recognition_points = rr.recognition_point_model(p_word_posterior,
                                                    stim.word_lengths,
                                                    threshold=params.threshold)
                                                    
    ############

    # Compute recognition onset, relative to word onset
    recognition_onsets = torch.gather(stim.phoneme_onsets_global, 1,
                                      recognition_points.unsqueeze(1)).squeeze(1)
    # Compute recognition onset as global index
    recognition_onsets_samp = time_to_sample(recognition_onsets, sample_rate)

    # Generate continuous signal stream.
    t_max = stim.phoneme_onsets_global[-1, -1] + (epoch_window[1] - epoch_window[0])
    Y = torch.normal(*noise_params,
                     size=(int(np.ceil(t_max * sample_rate)), num_sensors))

    # Sample a standardized response, which will be scaled by per-word surprisal.
    # TODO check that window size is sufficient to cover this
    window_std = params.b  # NB this means there's nontrivial response outside of b window.
    window_center = params.a + window_std / 2
    unit_response_xs, unit_response_ys = gaussian_window(window_center.item(), window_std.item(),
                                                         sample_rate=sample_rate)
    response_width = len(unit_response_xs)
    unit_response_ys = torch.tensor(unit_response_ys).view(-1, 1).tile((1, num_sensors))

    # Add delta response after each recognition onset.
    # response_delay = time_to_sample(params.a, sample_rate)
    # response_width = time_to_sample(params.b, sample_rate)
    for word_surp, rec_onset_samp in zip(stim.word_surprisals, recognition_onsets_samp):
        start_idx = rec_onset_samp
        end_idx = start_idx + response_width
        Y[start_idx:end_idx] += params.coef[1] * word_surp * unit_response_ys

    #############
    # Run epoching.

    epoch_tmin, epoch_tmax = torch.tensor(epoch_window)
    epoch_samples = time_to_sample(epoch_tmax - epoch_tmin, sample_rate)
    num_words = stim.word_lengths.shape[0]
    X_epoch = torch.stack([torch.ones(num_words), stim.word_surprisals], dim=1)
    Y_epoch = torch.empty(num_words, epoch_samples, num_sensors)
    for i, word_onset in enumerate(stim.word_onsets):
        start_idx = time_to_sample(word_onset + epoch_tmin, sample_rate)
        end_idx = time_to_sample(word_onset + epoch_tmax, sample_rate)

        val = Y[start_idx:end_idx, :]
        # Crop / pad if necessary
        if val.shape[0] > epoch_samples:
            val = val[(val.shape[0] - epoch_samples):, :]
        elif val.shape[0] < epoch_samples:
            # Pad on right.
            val = pad(val, (0, 0, 0, epoch_samples - val.shape[0]))

        Y_epoch[i, :, :] = val

    return rr.RRDataset(
        params=params,
        sample_rate=sample_rate,
        epoch_window=epoch_window,
        phonemes=PHONEMES.tolist(),

        p_word=stim.p_word,
        word_lengths=stim.word_lengths,
        candidate_phonemes=stim.candidate_phonemes,

        word_onsets=stim.word_onsets,
        phoneme_onsets=stim.phoneme_onsets,

        recognition_points=recognition_points,
        recognition_onsets=recognition_onsets,

        Y=Y,
        X_epoch=X_epoch,
        Y_epoch=Y_epoch,
    )
