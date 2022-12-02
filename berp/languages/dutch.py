"""
Defines data and utilities for Dutch processing.
"""

from collections import Counter, defaultdict
from functools import cache
import logging
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


L = logging.getLogger(__name__)

Phoneme = str


# NB # in CGN denotes cough, sneeze, etc.
#
# This mapping was manually designed using three sources:
# - CELEX documentation (https://catalog.ldc.upenn.edu/docs/LDC96L14/dug_let.ps p. 3-26)
#   which shows mapping between IPA, SAMPA, CELEX, etc. phonetic annotations, with example
#   words
# - CGN documentation on broad phonetic transcriptions
#   (https://lands.let.ru.nl/cgn/doc_Dutch/topics/version_1.0/annot/phonetics/fon_prot.pdf)
#   which includes sample words, but not IPA
# - The CELEX database itself, which we used to cross-reference the words in the CGN documentation
#   with their CELEX pronunciations
celex_cgn_mapping = {
    "&:": "2",
    "@": "@",
    "A": "A",
    "AU": "A+",
    "E": "E",
    "E:": "E:",
    "EI": "E+",
    "G": "G",
    "I": "I",
    "N": "N",
    "O": "O",
    "O:": "O:",
    "S": "S",
    "U": "Y",
    "UI": "Y+",
    "a:": "a",
    "b": "b",
    "d": "d",
    "e:": "e",
    "f": "f",
    "g": "g",
    "h": "h",
    "i:": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o:": "o",
    "p": "p",
    "r": "r",
    "s": "s",
    "t": "t",
    "u:": "u",
    "v": "v",
    "w": "w",
    "x": "x",
    "y:": "y",
    "z": "z",
    "Z": "Z",
}

# This mapping is maximally faithful to the CELEX representation. The
# function which uses this mapping then makes some neutralizations as
# necessary to map onto the Smits IPA vocabulary.
celex_ipa_mapping = {
    "&:": "ø",
    "@": "ə",
    "A": "ɑ",
    "AU": "ɑu",
    "E": "ɛ",
    "E:": "ɛː",
    "EI": "ɛi",
    "G": "ɣ",
    "I": "ɪ",
    "N": "ŋ",
    "O": "ɔ",
    "O:": "ɔː",
    "S": "ʃ",
    # is centralized -u- in CELEX doc but other sources seem to prefer ʏ
    "U": "ʏ",
    "U:": "ʏː",
    "UI": "œy",
    "a:": "aː",
    "b": "b",
    "d": "d",
    "e:": "eː",
    "f": "f",
    "g": "g",
    "h": "h",
    "i::": "iː",
    "i:": "iː",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o:": "oː",
    "p": "p",
    "r": "r",
    "s": "s",
    "t": "t",
    "u:": "uː",
    "v": "v",
    "w": "ʋ",
    "x": "x",
    "y:": "yː",
    "z": "z",
    "Z": "ʒ",
}

cgn_ipa_mapping = {
    cgn: celex_ipa_mapping[celex]
    for celex, cgn in celex_cgn_mapping.items()
}

celex_chars = set(celex_cgn_mapping.keys())
cgn_chars = set(celex_cgn_mapping.values()) | {"#"}
ipa_chars = set(celex_ipa_mapping.values())


def convert_to_smits_ipa(ipa_phoneme):
    """
    Collapse IPA distinctions as done by
    Smits et al in their phoneme confusion study.
    """
    ret = ipa_phoneme

    # Remove length distinctions.
    ret = ret.replace("ː", "")

    # Exclude voiced velar fricative because "many Dutch
    # speakers--including the speaker for the experiment--neutralize the
    # distinction, maintaining only /x/"
    # See Appendix A #1
    if ret == "ɣ": return "x"

    # Neutralize upper and lower mid vowels
    if ret == "ø": return "œ"

    # Etc. broader transcriptions
    if ret == "ʋ": return "w"

    return ret


smits_ipa_chars = set(convert_to_smits_ipa(phon) for phon in ipa_chars)

punct_only_re = re.compile(r"^[.?!:'\"]+$")

CELEX_PHONEME_RE = re.compile("|".join(sorted(celex_ipa_mapping.keys(), key=len, reverse=True)))
def convert_celex_to_ipa(celex_word, phoneme_processor=None) -> List[Phoneme]:
    phonemes = CELEX_PHONEME_RE.findall(celex_word)
    phonemes = [celex_ipa_mapping[phon] for phon in phonemes]
    # TODO check
    if phoneme_processor is not None:
        phonemes = [phoneme_processor(phon) for phon in phonemes]
    return phonemes


# Sequentially matches all phonemes
SMITS_IPA_PHONEME_RE = re.compile("|".join(sorted(smits_ipa_chars, key=len, reverse=True)))

class CelexPhonemizer:
    
    def __init__(self, celex_path,
                 override_words: Optional[Dict[str, List[Phoneme]]] = None
                 ):
        override_words = override_words or {}

        # Load CELEX pronunciation database. Keep only the most frequent 
        # pronunciation for a word.
        phonemizer_df = pd.read_csv(
            celex_path, sep="\\", header=None, usecols=[1, 2, 6],
            names=["word", "inl_freq", "celex_syl"]).dropna()
        phonemizer_df["word"] = phonemizer_df.word.str.lower()
        phonemizer_df = phonemizer_df \
            .sort_values("inl_freq", ascending=False) \
            .drop_duplicates(subset="word").set_index("word")
        phonemizer_df["celex"] = phonemizer_df.celex_syl \
            .str.replace(r"[\[\]\s]", "", regex=True) \
            .str.replace("::", ":", regex=False)

        # Pre-convert to IPA, and reduce to Smits representation
        # Compute a unigram phoneme frequency distribution at the same time
        # Compute cohort maps at the same time
        ipa_values, phoneme_freqs, cohorts = [], Counter(), defaultdict(set)
        for word, row in tqdm(phonemizer_df.iterrows(),
                              desc="Preprocessing pronunciation dictionary",
                              total=len(phonemizer_df)):
            ipa_word = override_words.get(word, None)
            if ipa_word is None:
                ipa_word = convert_celex_to_ipa(row.celex, phoneme_processor=convert_to_smits_ipa)
            ipa_word_str = "".join(ipa_word)
            ipa_values.append(ipa_word_str)

            phoneme_freqs.update(ipa_word)
            for prefix in range(len(ipa_word)):
                cohorts["".join(ipa_word[:prefix])].add(ipa_word_str)
        phonemizer_df["ipa"] = ipa_values
        self._phoneme_freqs = pd.Series(phoneme_freqs)
        self._phoneme_freqs /= self._phoneme_freqs.sum()
        self._cohorts = {prefix: list(cohort) for prefix, cohort in cohorts.items()}
        
        self._df = phonemizer_df.drop(columns=["celex", "celex_syl"])
        self._df_by_ipa = self._df.reset_index().set_index("ipa")
        
        self._celex_chars = set([char for celex in phonemizer_df.celex.tolist() for char in celex])
        
        self.missing_counter = Counter()
        
    @cache
    def __call__(self, string) -> List[Phoneme]:
        if punct_only_re.match(string):
            return [""]

        try:
            return SMITS_IPA_PHONEME_RE.findall(self._df.loc[string].ipa)
        except KeyError:
            self.missing_counter[string] += 1
            # if self.missing_counter[string] == 1:
            #     L.warning(f"Candidate word {string} is not in CELEX.")

            # Dumb -- just return the subset of characters that are in IPA code
            return [char for char in string if char in ipa_chars]

    def cohort_distribution(self, ipa_prefix, pad_phoneme="_"):
        """
        Compute a cohort word distribution compatible with the given prefix.
        """
        cohort_words = self._cohorts.get(ipa_prefix, [])
        if not cohort_words:
            return None
        cohort_df = self._df_by_ipa.loc[cohort_words].reset_index()
        
        cohort_df["next"] = cohort_df.ipa.str[len(ipa_prefix):] \
            .str.findall(SMITS_IPA_PHONEME_RE).str[0] \
            .fillna(pad_phoneme)
        # add-1 smoothed probability over words
        cohort_df["p"] = (cohort_df.inl_freq + 1) / (cohort_df.inl_freq + 1).sum()
        return cohort_df.drop(columns=["inl_freq"])

    @cache
    def cohort_phoneme_distribution(self, ipa_prefix, pad_phoneme="_"):
        df = self.cohort_distribution(ipa_prefix, pad_phoneme=pad_phoneme)
        if df is None:
            # Back off to unigram distribution
            return self._phoneme_freqs

        ps = df.groupby("next").p.sum()
        # backoff smooth with phoneme unigram distribution.
        gamma = 0.1
        ps = (1 - gamma) * ps
        ps = ps.add(gamma * self._phoneme_freqs, fill_value=0)
        ps /= ps.sum()
        return ps

    def phoneme_surprisal_entropy(self, ipa_prefix, phoneme) -> Tuple[float, float]:
        """
        Compute the entropy over the predictive phoneme distribution conditioned
        on the prefix, and the surprisal of the given phoneme.
        """
        dist = self.cohort_phoneme_distribution(ipa_prefix)
        surprisals = -np.log2(dist)
        return surprisals.loc[phoneme], (dist * surprisals).sum()

    def word_phoneme_info(self, ipa_word) -> List[Tuple[float, float]]:
        """
        Compute phoneme surprisal and entropy values for each phoneme of the
        given IPA word.

        Surprisal here is the negative log conditional of phoneme p_i

            -log_2 p(p_i | p_1,...p_{i-1})

        and entropy is the expected surprisal of the POSTERIOR distribution after
        consuming phoneme p_i

            H(word | p_1, ... p_i)

        NB that they are not computed on the same distribution. We do this to
        match the Gillis implementation.
        """
        surprisals, entropies = [], []
        phonemes = SMITS_IPA_PHONEME_RE.findall(ipa_word)
        for prefix_length in range(len(phonemes) + 1):
            prefix = "".join(phonemes[:prefix_length])

            if prefix_length == len(phonemes):
                # Only compute posterior entropy
                surprisal = 0
                dist_prev = self.cohort_phoneme_distribution(prefix)
                entropy = (dist_prev * -np.log2(dist_prev)).sum()
            else:
                # Compute surprisal and entropy
                surprisal, entropy = self.phoneme_surprisal_entropy(prefix, phonemes[prefix_length])

            surprisals.append(surprisal)
            entropies.append(entropy)

        return list(zip(surprisals[:-1], entropies[1:]))