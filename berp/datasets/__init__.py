from dataclasses import dataclass
from functools import cached_property
import logging
from typing import List, Iterator, Tuple

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
        self.idx2tok: List[Tuple[Phoneme, ...]] = []

    def __eq__(self, other):
        return isinstance(other, Vocabulary) and \
            self.idx2tok == other.idx2tok

    def __len__(self):
        return len(self.idx2tok)

    def __iter__(self) -> Iterator[Tuple[int, Tuple[Phoneme, ...]]]:
        return enumerate(self.idx2tok)
    
    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.idx2tok[key]
        elif isinstance(key, tuple):
            return self.tok2idx[key]
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def __contains__(self, key):
        if isinstance(key, int):
            return key < len(self.idx2tok)
        elif isinstance(key, tuple):
            return key in self.tok2idx
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def add(self, token: Tuple[Phoneme, ...]):
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

    word_feature_names: List[str]

    phoneme_features: List[TensorType[..., "num_phoneme_features", float]]
    """
    Arbitrary phoneme-level features. The `i`th tensor in this list is of
    length `word_lengths[i]`.
    """

    phoneme_feature_names: List[str]

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

    def __eq__(self, other):
        if not isinstance(other, NaturalLanguageStimulus):
            return False

        return (
            self.name == other.name
            and self.phonemes == other.phonemes
            and self.pad_phoneme_id == other.pad_phoneme_id
            and torch.all(self.word_ids == other.word_ids)
            and torch.all(self.word_lengths == other.word_lengths)
            and torch.allclose(self.word_features, other.word_features)
            and self.word_feature_names == other.word_feature_names
            and len(self.phoneme_features) == len(other.phoneme_features)
            and all(
                torch.allclose(f1, f2)
                for f1, f2 in zip(self.phoneme_features, other.phoneme_features)
            )
            and self.phoneme_feature_names == other.phoneme_feature_names
            and torch.allclose(self.p_candidates, other.p_candidates)
            and torch.all(self.candidate_ids == other.candidate_ids)
            and self.candidate_vocabulary == other.candidate_vocabulary
        )

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
        max_phonemes = self.max_n_phonemes
        candidate_phoneme_voc = torch.zeros(
            (len(self.candidate_vocabulary), max_phonemes),
            dtype=torch.long)
        candidate_phoneme_voc.fill_(self.pad_phoneme_id)

        phon2idx = {p: i for i, p in enumerate(self.phonemes)}
        for idx, candidate in self.candidate_vocabulary:
            phoneme_seq = torch.tensor([phon2idx[phon] for phon in candidate])
            candidate_phoneme_voc[idx, :len(phoneme_seq)] = phoneme_seq[:max_phonemes]

        reindexed = torch.index_select(candidate_phoneme_voc, 0,
                                       self.candidate_ids.flatten())
        reindexed = reindexed.reshape(
            *self.candidate_ids.shape, self.max_n_phonemes)
        return reindexed

    def get_candidate_strs(self, word_idx, top_k=None) -> List[Tuple[Phoneme, ...]]:
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