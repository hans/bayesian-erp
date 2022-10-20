from dataclasses import dataclass
from functools import cached_property
import logging
from typing import List

import numpy as np
import torch
from torchtyping import TensorType  # mypy: ignore

from berp.typing import DIMS, is_probability, is_log_probability, is_positive

L = logging.getLogger(__name__)


# Type variables
B, N_W, N_C, N_F, N_F_T, N_P, V_W = \
    DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_F_T, DIMS.N_P, DIMS.V_W
T, S = DIMS.T, DIMS.S


# Type aliases
Phoneme = str


class Vocabulary(object):

    def __init__(self):
        self.tok2idx = {}
        self.idx2tok = []

    def __len__(self):
        return len(self.idx2tok)
    
    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.idx2tok[key]
        elif isinstance(key, str):
            return self.tok2idx[key]
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def __contains__(self, key):
        if isinstance(key, int):
            return key < len(self.idx2tok)
        elif isinstance(key, str):
            return key in self.tok2idx
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def add(self, token: str):
        if token not in self.tok2idx:
            self.tok2idx[token] = len(self.idx2tok)
            self.idx2tok.append(token)

        return self.tok2idx[token]


@dataclass
class NaturalLanguageStimulus:
    """
    Describes the linguistic stimulus used in a trial, along with probabilistic model
    outputs of counterfactual stimulus data. This data is shareable across subjects
    who perceived the same data. Separately, BerpDataset represents the particular
    presentation of a stimulus to a subject, along with recorded responses.
    """

    name: str

    phonemes: List[Phoneme]
    """
    Phoneme vocabulary.
    """

    pad_phoneme_id: int
    """
    Index of padding phoneme in phoneme vocabulary.
    """

    word_ids: TensorType[N_W, torch.long]
    """
    For each row in the dataset, the ID of the corresponding word in the
    source corpus.
    """

    word_lengths: TensorType[N_W, int]
    """
    Length of each ground-truth word in the dataset (in number of phonemes).
    """

    word_features: TensorType[N_W, N_F, float]
    """
    Arbitrary word-level features.
    """

    p_candidates: TensorType[N_W, N_C, torch.float, is_log_probability]
    """
    Prior predictive distribution over words at each timestep. Each
    row is a proper log-e-probability distribution.
    """

    candidate_ids: TensorType[N_W, N_C, torch.long]
    """
    For each row in the dataset, the IDs of the top `N_C` candidates in the
    candidate vocabulary.
    """

    candidate_vocabulary: Vocabulary
    """
    Vocabulary of candidate words referred to by `candidate_ids`.
    """

    @property
    def max_n_phonemes(self):
        return max(self.word_lengths)

    @property
    def word_surprisals(self) -> TensorType[N_W, torch.float, is_positive]:
        """
        Get surprisals of ground-truth words (in bits; log-2).
        """
        return -self.p_candidates[:, 0] / np.log(2)

    @cached_property
    def candidate_phonemes(self) -> TensorType[N_W, N_C, N_P, torch.long]:
        """
        For each candidate in each prior predictive, the corresponding
        phoneme sequence. Sequences are padded with `pad_phoneme_id`.
        """
        # Compute phoneme sequences for all candidates.
        candidate_phoneme_voc = torch.zeros(
            (len(self.candidate_vocabulary), self.max_n_phonemes),
            dtype=torch.long)
        candidate_phoneme_voc.fill_(self.pad_phoneme_id)

        phon2idx = {p: i for i, p in enumerate(self.phonemes)}
        for i, candidate in enumerate(self.candidate_vocabulary):
            phoneme_seq = torch.tensor([phon2idx[phon] for phon in candidate])
            candidate_phoneme_voc[i, :len(phoneme_seq)] = phoneme_seq

        reindexed = torch.index_select(candidate_phoneme_voc, 0,
                                       self.candidate_ids.flatten())
        reindexed = reindexed.reshape(
            *self.candidate_ids.shape, self.max_n_phonemes)
        return reindexed

    def get_candidate_strs(self, word_idx, top_k=None) -> List[str]:
        """
        Get string representations for the candidates of the given word.
        """
        return [self.candidate_vocabulary[i.item()] for i in
                self.candidate_ids[word_idx, :top_k]]

from berp.datasets.processor import NaturalLanguageStimulusProcessor
from berp.datasets.base import BerpDataset, NestedBerpDataset

__all__ = [
    "NaturalLanguageStimulusProcessor",
    "BerpDataset", "NestedBerpDataset",
]