"""
Defines data and utilities for Dutch processing.
"""

from collections import Counter
from functools import cache
import logging
import re
from typing import List

import pandas as pd


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


def convert_celex_to_ipa(celex_word, phoneme_processor=None) -> List[Phoneme]:
    # Greedily consume phonemes
    celex_keys = sorted(celex_ipa_mapping.keys(), key=lambda code: -len(code))
    ret = []
    orig = celex_word
    i = 0
    while celex_word:
        for key in celex_keys:
            if celex_word.startswith(key):
                phon_i = celex_ipa_mapping[key]
                if phoneme_processor is not None:
                    phon_i = phoneme_processor(phon_i)

                ret.append(phon_i)
                celex_word = celex_word[len(key):]
                break
        else:
            raise KeyError(f"{orig} -> {celex_word}")
            
        i += 1
        if i == 10:
            break
            
    return ret


class CelexPhonemizer:
    
    def __init__(self, celex_path):
        # Load CELEX pronunciation database. Keep only the most frequent 
        # pronunciation for a word.
        phonemizer_df = pd.read_csv(
            celex_path, sep="\\", header=None, usecols=[1, 2, 6],
            names=["word", "inl_freq", "celex_syl"]).dropna()
        phonemizer_df["word"] = phonemizer_df.word.str.lower()
        phonemizer_df = phonemizer_df \
            .sort_values("inl_freq", ascending=False) \
            .drop_duplicates(subset="word").set_index("word")
        phonemizer_df["celex"] = phonemizer_df.celex_syl.str.replace(r"[\[\]]", "", regex=True)
        
        self._df = phonemizer_df
        
        self._celex_chars = set([char for celex in phonemizer_df.celex.tolist() for char in celex])
        
        self.missing_counter = Counter()
    
    @cache
    def __call__(self, string):
        if punct_only_re.match(string):
            return ""

        try:
            celex_form = self._df.loc[string].celex
        except KeyError:
            self.missing_counter[string] += 1
            # if self.missing_counter[string] == 1:
            #     L.warning(f"Candidate word {string} is not in CELEX.")

            # Dumb -- just return the subset of characters that are in IPA code
            return [char for char in string if char in ipa_chars]
        else:
            ipa_form = convert_celex_to_ipa(celex_form)
            
            # Reduce to Smits IPA representation
            ipa_form = [convert_to_smits_ipa(phon) for phon in ipa_form]
            return ipa_form