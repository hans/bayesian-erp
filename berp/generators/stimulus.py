from collections import Counter
import itertools
import logging
import re
from typing import Callable, List, NamedTuple, Tuple, Optional, Dict

import numpy as np
import torch.distributions as dist
import torch
from torch.nn.functional import pad
from torchtyping import TensorType
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from transformers.tokenization_utils_base import BatchEncoding
from tqdm.auto import tqdm, trange

from berp.datasets.processor import NaturalLanguageStimulusProcessor
from berp.typing import DIMS, is_probability, is_log_probability


L = logging.getLogger(__name__)

# Type variables
B, N_W, N_C, N_F, N_P, V_W = DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_P, DIMS.V_W
T, S = DIMS.T, DIMS.S


class Stimulus(NamedTuple):

    word_lengths: TensorType[N_W, torch.int]
    phoneme_onsets: TensorType[N_W, N_P, torch.float]
    phoneme_onsets_global: TensorType[N_W, N_P, torch.float]
    word_onsets: TensorType[N_W, torch.float]
    word_offsets: TensorType[N_W, torch.float]
    word_surprisals: TensorType[N_W, torch.float]
    p_candidates: TensorType[N_W, N_C, torch.float, is_log_probability]
    candidate_phonemes: TensorType[N_W, N_C, N_P, torch.long]


def align_to_sample_rate(times, sample_rate):
    """
    Adjust the time series so that each event happens aligned to the left edge of
    a sample, assuming the given sample rate.
    """
    return torch.round(times * sample_rate) / sample_rate


class StimulusGenerator(object):

    def __init__(self,
                 phon_delay_range: Tuple[float, float] = (0.08, 0.2),
                 word_delay_range: Tuple[float, float] = (0.1, 0.4)):
        self.phon_delay_range = phon_delay_range
        self.word_delay_range = word_delay_range

        self.first_onset = 1.0  # TODO magic to make epoching not break

    def __call__(self, *args, **kwargs) -> Stimulus:
        raise NotImplementedError()

    def sample_stream(self, word_lengths: TensorType[N_W, int],
                      max_num_phonemes: int,
                      align_sample_rate: Optional[int] = None,
                      ) -> Tuple[TensorType[N_W, N_P, float],
                                 TensorType[N_W, N_P, float],
                                 TensorType[N_W, float],
                                 TensorType[N_W, float]]:
        num_words = len(word_lengths)

        phoneme_durations = rand_unif(*self.phon_delay_range, num_words, max_num_phonemes)
        phoneme_durations[torch.arange(max_num_phonemes) >= word_lengths.unsqueeze(1)] = 0.

        word_delays = rand_unif(*self.word_delay_range, num_words)
        word_delays[0] = 0.

        if align_sample_rate is not None:
            phoneme_durations = align_to_sample_rate(phoneme_durations, align_sample_rate)
            word_delays = align_to_sample_rate(word_delays, align_sample_rate)

        # Remaining variables are deterministic derivatives on the above.

        phoneme_onsets = torch.cat([
            torch.zeros(num_words, 1),
            phoneme_durations.cumsum(dim=1)[:, :-1]
        ], dim=1)
        phoneme_offsets = phoneme_durations.cumsum(dim=1)

        assert (phoneme_offsets[:, :-1] <= phoneme_onsets[:, 1:]).all().item()

        word_durations = phoneme_offsets[:, -1] - phoneme_onsets[:, 0]
        word_onsets = (torch.cat([torch.tensor([self.first_onset]),
                                  word_durations[:-1]])
                                + word_delays).cumsum(0)
        word_offsets = word_onsets + word_durations

        assert (word_offsets[:-1] < word_onsets[1:]).all().item()
        
        # Make phoneme_onsets global (not relative to word onset).
        phoneme_onsets_global = phoneme_onsets + word_onsets.view(-1, 1)

        return phoneme_onsets, phoneme_onsets_global, word_onsets, word_offsets


def rand_unif(low, high, *shape) -> torch.Tensor:
    return torch.rand(*shape) * (high - low) + low


class RandomStimulusGenerator(StimulusGenerator):

    PAD_PHONEME = "_"

    def __init__(self,
                 num_words: int = 100,
                 num_candidates: int = 10,
                 num_phonemes: int = 5,
                 phoneme_voc_size: int = 18,
                 word_surprisal_params: Tuple[float, float] = (1., 0.5),
                 **kwargs):
        super().__init__(**kwargs)

        self.num_words = num_words
        self.num_candidates = num_candidates
        self.num_phonemes = num_phonemes
        self.word_surprisal_params = word_surprisal_params

        # Generate phoneme set
        self.phonemes = np.array(list("abcdefghijklmnopqrstuvwxyz"[:phoneme_voc_size - 1] + "_"))
        self.phoneme2idx = {p: idx for idx, p in enumerate(self.phonemes)}

    def __call__(self, **stream_kwargs) -> Stimulus:
        word_lengths = 1 + dist.Binomial(self.num_phonemes - 1, 0.5) \
            .sample((self.num_words, self.num_candidates)).long()  # type: ignore
        gt_word_lengths = word_lengths[:, 0]

        candidate_phonemes = torch.randint(0, len(self.phonemes) - 2,
                                          (self.num_words,
                                           self.num_candidates,
                                           self.num_phonemes))
        # Use padding token when word length exceeded.
        # TODO can have candidates with different lengths
        pad_idx = self.phoneme2idx[self.PAD_PHONEME]
        pad_mask = (torch.arange(self.num_phonemes) >= word_lengths[:, :, None])
        candidate_phonemes[pad_mask] = pad_idx

        phoneme_onsets, phoneme_onsets_global, word_onsets, word_offsets = \
            self.sample_stream(gt_word_lengths, self.num_phonemes, **stream_kwargs)

        word_surprisals: torch.Tensor = dist.LogNormal(*self.word_surprisal_params) \
            .sample((self.num_words,))  # type: ignore

        # Calculate p_candidates using surprisal; allocate remainder randomly
        p_gt_word = (-word_surprisals).exp()
        remainder = 1 - p_gt_word
        p_candidates = (remainder / (self.num_candidates - 1)).view(-1, 1) \
            * torch.ones(self.num_words, self.num_candidates - 1)
        p_candidates = torch.cat([p_gt_word.view(-1, 1), p_candidates], dim=1) \
            .log()

        return Stimulus(gt_word_lengths, phoneme_onsets, phoneme_onsets_global,
                        word_onsets, word_offsets,
                        word_surprisals, p_candidates, candidate_phonemes)
    
    @property
    def pad_phoneme_id(self):
        return self.phoneme2idx[self.PAD_PHONEME]


class NaturalLanguageStimulusGenerator(StimulusGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.processor = NaturalLanguageStimulusProcessor(*args, **kwargs)

    def __call__(self, tokens: List[str],
                 word_features: Dict[int, torch.Tensor],
                 word_to_token: Optional[Dict[int, List[int]]] = None,
                 ground_truth_phonemes: Optional[Dict[int, List[str]]] = None) -> Stimulus:

        if word_to_token is None:
            # Assume words are the same as tokens
            assert len(word_features) == len(tokens)
            word_to_token = {idx: [idx] for idx in word_features}
    
        nl_stim = self.processor(tokens, word_to_token, word_features, ground_truth_phonemes)

        max_num_phonemes = nl_stim.candidate_phonemes.shape[2]
        phoneme_onsets, phoneme_onsets_global, word_onsets, word_offsets = \
            self.sample_stream(nl_stim.word_lengths, max_num_phonemes)

        return Stimulus(
            nl_stim.word_lengths, phoneme_onsets, phoneme_onsets_global,
            word_onsets, word_offsets,
            nl_stim.word_surprisals, nl_stim.p_candidates, nl_stim.candidate_phonemes
        )