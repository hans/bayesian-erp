from typing import List, NamedTuple, Tuple, Optional, Dict, Union, Callable

import numpy as np
import pyro.distributions as dist
import torch
from torch.nn.functional import pad
from torchtyping import TensorType
from tqdm.notebook import tqdm
from typeguard import typechecked

from berp.generators.stimulus import Stimulus, StimulusGenerator
from berp.models import reindexing_regression as rr
from berp.typing import DIMS, is_log_probability
from berp.util import sample_to_time, time_to_sample, gaussian_window

PHONEMES = np.array(list("abcdefghijklmnopqrstuvwxyz_"))
phoneme2idx = {p: idx for idx, p in enumerate(PHONEMES)}
def random_word(length):
    return np.random.choice(PHONEMES, size=length)

phoneme_confusion = torch.diag(torch.ones(len(PHONEMES))) + \
    0.1 * torch.rand(len(PHONEMES), len(PHONEMES))
phoneme_confusion /= phoneme_confusion.sum(dim=0, keepdim=True)


# Type variables
B, N_W, N_C, N_F, N_P, V_W = DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_P, DIMS.V_W
T, S = DIMS.T, DIMS.S


def response_model(stim: Stimulus,
                   recognition_onsets: TensorType[N_W, float],
                   params: rr.ModelParameters,
                   num_sensors: int = 1,
                   sample_rate: int = 128,
                   noise_params: Tuple[float, float] = (0., 0.5),
                   right_padding: float = 0.,
                   ) -> torch.Tensor:
    """
    Generate a response time series for the given stimulus and response parameters.
    """

    # Compute recognition onset as global index
    recognition_onsets_global_samp = time_to_sample(stim.word_onsets + recognition_onsets, sample_rate)

    # Generate continuous signal stream.
    t_max = stim.phoneme_onsets_global[-1, -1] + right_padding
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

    # Add characteristic response after each recognition onset.
    for word_surp, rec_onset_samp in zip(stim.word_surprisals, recognition_onsets_global_samp):
        start_idx = rec_onset_samp
        end_idx = start_idx + response_width
        Y[start_idx:end_idx] += params.coef[1] * word_surp * unit_response_ys

    return Y


@typechecked
def sample_dataset(params: rr.ModelParameters,
                   stimulus_generator: Union[StimulusGenerator, Callable[[], Stimulus]],
                   num_sensors: int = 1,
                   sample_rate: int = 128,
                   epoch_window: Tuple[float, float] = (-0.1, 1.0),
                   noise_params: Tuple[float, float] = (0., 0.5),
                   ) -> rr.RRDataset:
    
    stim = stimulus_generator()

    ############

    p_word_posterior = rr.predictive_model(stim.p_word, stim.candidate_phonemes,
                                           confusion=params.confusion,
                                           lambda_=params.lambda_)
    recognition_points = rr.recognition_point_model(p_word_posterior,
                                                    stim.word_lengths,
                                                    threshold=params.threshold)
    
    # Compute recognition onset, relative to word onset
    recognition_onsets = torch.gather(stim.phoneme_onsets, 1,
                                      recognition_points.unsqueeze(1)).squeeze(1)
                                                    
    ############

    epoch_width = epoch_window[1] - epoch_window[0]
    Y = response_model(stim, recognition_onsets, params, num_sensors, sample_rate,
                       noise_params=noise_params, right_padding=epoch_width)

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
