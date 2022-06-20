from typing import NamedTuple, Tuple

import numpy as np
import pyro.distributions as dist
import torch
from torchtyping import TensorType

from berp.typing import DIMS


# Type variables
B, N_W, N_C, N_F, N_P, V_W = DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_P, DIMS.V_W
T, S = DIMS.T, DIMS.S


class Stimulus(NamedTuple):

    word_lengths: TensorType[N_W, int]
    phoneme_onsets: TensorType[N_W, N_P, float]
    phoneme_onsets_global: TensorType[N_W, N_P, float]
    word_onsets: TensorType[N_W, float]
    word_surprisals: TensorType[N_W, float]
    p_word: TensorType[N_W, N_C, float]
    candidate_phonemes: TensorType[N_W, N_C, N_P, int]


class StimulusGenerator(object):

    def __call__(self, *args, **kwargs) -> Stimulus:
        raise NotImplementedError()


def rand_unif(low, high, *shape) -> torch.Tensor:
    return torch.rand(*shape) * (high - low) + low


class RandomStimulusGenerator(StimulusGenerator):

    def __init__(self,
                 num_words: int = 100,
                 num_candidates: int = 10,
                 num_phonemes: int = 5,
                 phoneme_voc_size: int = 18,
                 phon_delay_range: Tuple[float, float] = (0.04, 0.1),
                 word_delay_range: Tuple[float, float] = (0.01, 0.1),
                 word_surprisal_params: Tuple[float, float] = (1., 0.5),):
        self.num_words = num_words
        self.num_candidates = num_candidates
        self.num_phonemes = num_phonemes
        self.phon_delay_range = phon_delay_range
        self.word_delay_range = word_delay_range
        self.word_surprisal_params = word_surprisal_params

        # Generate phoneme set
        self.phonemes = np.array(list("abcdefghijklmnopqrstuvwxyz"[:phoneme_voc_size - 1] + "_"))
        self.phoneme2idx = {p: idx for idx, p in enumerate(self.phonemes)}

        self.first_onset = 1.0  # TODO magic

    def __call__(self) -> Stimulus:
        word_lengths = 1 + dist.Binomial(self.num_phonemes - 1, 0.5) \
            .sample((self.num_words,)).long()  # type: ignore

        candidate_phonemes = torch.randint(0, len(self.phonemes) - 2,
                                          (self.num_words,
                                           self.num_candidates,
                                           self.num_phonemes))
        # Use padding token when word length exceeded.
        # TODO can have candidates with different lengths
        pad_idx = self.phoneme2idx["_"]
        pad_mask = (torch.arange(self.num_phonemes) >= word_lengths[:, None])[:, :, None] \
            .transpose(1, 2).tile((1, self.num_candidates, 1))
        candidate_phonemes[pad_mask] = pad_idx

        phoneme_onsets = rand_unif(*self.phon_delay_range, self.num_words, self.num_phonemes)
        phoneme_onsets[:, 0] = 0.
        phoneme_onsets = phoneme_onsets.cumsum(1)
        word_delays = rand_unif(*self.word_delay_range, self.num_words)
        word_onsets = (torch.cat([torch.tensor([self.first_onset]),
                                self.first_onset + phoneme_onsets[1:, -1]])
                                + word_delays).cumsum(0)
        # Make phoneme_onsets global (not relative to word onset).
        phoneme_onsets_global = phoneme_onsets + word_onsets.view(-1, 1)

        word_surprisals: torch.Tensor = dist.LogNormal(*self.word_surprisal_params) \
            .sample((self.num_words,))  # type: ignore

        # Calculate p_word using surprisal; allocate remainder randomly
        p_gt_word = (-word_surprisals).exp()
        remainder = 1 - p_gt_word
        p_candidates = (remainder / (self.num_candidates - 1)).view(-1, 1) \
            * torch.ones(self.num_words, self.num_candidates - 1)
        p_word = torch.cat([p_gt_word.view(-1, 1), p_candidates], dim=1) \
            .log()

        return Stimulus(word_lengths, phoneme_onsets, phoneme_onsets_global,
                        word_onsets, word_surprisals, p_word, candidate_phonemes)