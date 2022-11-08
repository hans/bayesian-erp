from functools import partial
import itertools
from typing import List, NamedTuple, Tuple, Optional, Dict, Union, Callable
import uuid

import numpy as np
import torch
from torch.nn.functional import pad
from torchtyping import TensorType
from tqdm.notebook import tqdm
from typeguard import typechecked

from berp.datasets import BerpDataset
from berp.generators import response
from berp.generators.stimulus import Stimulus, StimulusGenerator
from berp.models import reindexing_regression as rr
from berp.typing import DIMS, is_log_probability
from berp.util import sample_to_time, time_to_sample, gaussian_window

PHONEMES = np.array(list("abcdefghijklmnopqrstuvwxyz_"))
phoneme2idx = {p: idx for idx, p in enumerate(PHONEMES)}
def random_word(length):
    return np.random.choice(PHONEMES, size=length)

vowels = list("aeiou")
consonants = list("bcdfghjklmnpqrstvwxyz")
phoneme_confusion = torch.diag(torch.ones(len(PHONEMES)))
# Add confusion between sets of phonemes
confusion_constant = 0.2
for confusion_set in [vowels, consonants]:
    for p_i, p_j in itertools.product(confusion_set, repeat=2):
        if p_i == p_j: continue
        phoneme_confusion[phoneme2idx[p_i], phoneme2idx[p_j]] += confusion_constant / len(confusion_set)
phoneme_confusion /= phoneme_confusion.sum(dim=0, keepdim=True)


# Type variables
B, N_W, N_C, N_F, N_P, V_W = DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_P, DIMS.V_W
T, S = DIMS.T, DIMS.S


def response_model(stim: Stimulus,
                   recognition_onsets: TensorType[N_W, float],
                   params: rr.ModelParameters,
                   num_sensors: int = 1,
                   sample_rate: int = 128,
                   response_type: str = "gaussian",
                   right_padding: float = 0.,
                   ) -> torch.Tensor:
    """
    Generate a response time series for the given stimulus and response parameters.
    """

    # Compute recognition onset as global index
    recognition_onsets_global_samp = time_to_sample(stim.word_onsets + recognition_onsets, sample_rate)

    # Generate continuous signal stream.
    t_max = stim.phoneme_onsets_global[-1, -1] + right_padding
    Y = torch.normal(0, params.sigma,
                     size=(int(np.ceil(t_max * sample_rate)), num_sensors))

    # Sample a standardized response, which will be scaled by per-word surprisal.
    if response_type == "gaussian":
        _, unit_response_ys = response.simple_gaussian(params.b, params.a, sample_rate)
        response_fn = lambda surprisal: params.coef[-1] * unit_response_ys * surprisal
    elif response_type == "square":
        _, unit_response_ys = response.simple_peak(params.b, params.a, sample_rate)
        response_fn = lambda surprisal: params.coef[-1] * unit_response_ys * surprisal
    elif response_type == "n400":
        response_fn = lambda surprisal: response.n400_like(surprisal, sample_rate=sample_rate)[1]
    else:
        raise ValueError("Unknown response type: {}".format(response_type))

    # Add characteristic response after each recognition onset.
    for word_surp, rec_onset_samp in zip(stim.word_surprisals, recognition_onsets_global_samp):
        response_values = response_fn(word_surp)
        if response_values.ndim == 1:
            # Tile across sensors.
            response_values = response_values.view(-1, 1).tile((1, num_sensors))

        start_idx = rec_onset_samp
        end_idx = start_idx + len(response_values)

        if end_idx >= Y.shape[0]:
            response_values = response_values[:Y.shape[0] - start_idx]
        Y[start_idx:end_idx] += response_values

    return Y


# @typechecked
def sample_dataset(params: rr.ModelParameters,
                   stimulus_generator: Union[StimulusGenerator, Callable[[], Stimulus]],
                   num_sensors: int = 1,
                   sample_rate: int = 128,
                   response_type: str = "gaussian",
                   epoch_window: Tuple[float, float] = (-0.1, 1.0),
                   include_intercept=True,
                   stimulus_kwargs=None,
                   ) -> BerpDataset:
    
    stim = stimulus_generator(**(stimulus_kwargs or {}))

    ############

    p_candidates_posterior = rr.predictive_model(stim.p_candidates, stim.candidate_phonemes,
                                           confusion=params.confusion,
                                           lambda_=params.lambda_)
    recognition_points = rr.recognition_point_model(p_candidates_posterior,
                                                    stim.word_lengths,
                                                    threshold=params.threshold)
    
    # Compute recognition onset, relative to word onset
    recognition_onsets = torch.gather(stim.phoneme_onsets, 1,
                                      recognition_points.unsqueeze(1)).squeeze(1)
                                                    
    ############

    epoch_width = epoch_window[1] - epoch_window[0]
    Y = response_model(stim, recognition_onsets, params, num_sensors, sample_rate,
                       response_type=response_type)

    #############
    # Run epoching.

    epoch_tmin, epoch_tmax = torch.tensor(epoch_window)
    epoch_samples = time_to_sample(epoch_tmax - epoch_tmin, sample_rate)
    num_words = stim.word_lengths.shape[0]

    if include_intercept:
        X_variable = torch.stack([torch.ones(num_words), stim.word_surprisals], dim=1)
    else:
        X_variable = stim.word_surprisals[:, None]

    # TODO any features to add here?
    X_ts = torch.zeros((Y.shape[0], 1))

    return BerpDataset(
        name=uuid.uuid4().hex[:8],
        # TODO store this on Stimulus and draw on it here
        stimulus_name=uuid.uuid4().hex[:8],
        sample_rate=sample_rate,
        phonemes=PHONEMES.tolist(),

        p_candidates=stim.p_candidates,
        word_lengths=stim.word_lengths,
        candidate_phonemes=stim.candidate_phonemes,

        word_onsets=stim.word_onsets,
        word_offsets=stim.word_offsets,
        phoneme_onsets=stim.phoneme_onsets,

        Y=Y,
        X_variable=X_variable,
        X_ts=X_ts,
    )
