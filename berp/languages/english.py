"""
Defines data and utilities for English processing.
"""

from typing import List, Dict

from collections import Counter
from functools import cache
import re

# Maps CMU dict pronunciation elements to IPA as used in Heilbron 2022 annotations.
cmu_ipa_mapping = {
    "AO": "\u0254",
    "AO0": "\u0254",
    "AO1": "\u0254",
    "AO2": "\u0254",
    "AA": "\u0251",
    "AA0": "\u0251",
    "AA1": "\u0251",
    "AA2": "\u0251",
    "IY": "i",
    "IY0": "i",
    "IY1": "i",
    "IY2": "i",
    "UW": "u",
    "UW0": "u",
    "UW1": "u",
    "UW2": "u",
    "EH": "\u025b",
    "EH0": "\u025b",
    "EH1": "\u025b",
    "EH2": "\u025b",
    "IH": "\u026a",
    "IH0": "\u026a",
    "IH1": "\u026a",
    "IH2": "\u026a",
    "UH": "\u028a",
    "UH0": "\u028a",
    "UH1": "\u028a",
    "UH2": "\u028a",
    "AH": "\u028c",
    "AH0": "\u0259",
    "AH1": "\u028c",
    "AH2": "\u028c",
    "AE": "\u00e6",
    "AE0": "\u00e6",
    "AE1": "\u00e6",
    "AE2": "\u00e6",
    "AX": "\u0259",
    "AX0": "\u0259",
    "AX1": "\u0259",
    "AX2": "\u0259",
    "EY": "\u025b\u026a",
    "EY0": "\u025b\u026a",
    "EY1": "\u025b\u026a",
    "EY2": "\u025b\u026a",
    "AY": "a\u026a",
    "AY0": "a\u026a",
    "AY1": "a\u026a",
    "AY2": "a\u026a",
    "OW": "o\u028a",
    "OW0": "o\u028a",
    "OW1": "o\u028a",
    "OW2": "o\u028a",
    "AW": "a\u028a",
    "AW0": "a\u028a",
    "AW1": "a\u028a",
    "AW2": "a\u028a",
    "OY": "\u0254\u026a",
    "OY0": "\u0254\u026a",
    "OY1": "\u0254\u026a",
    "OY2": "\u0254\u026a",
    "P": "p",
    "B": "b",
    "T": "t",
    "D": "d",
    "K": "k",
    "G": "g",
    "CH": "t\u0283",
    "JH": "d\u0292",
    "F": "f",
    "V": "v",
    "TH": "\u03b8",
    "DH": "\u00f0",
    "S": "s",
    "Z": "z",
    "SH": "\u0283",
    "ZH": "\u0292",
    "HH": "h",
    "M": "m",
    "N": "n",
    "NG": "\u014b",
    "L": "l",
    "R": "\u0279",
    "ER": "\u025a",
    "ER0": "\u025a",
    "ER1": "\u025a",
    "ER2": "\u025a",
    "AXR": "\u0259r",
    "AXR0": "\u0259r",
    "AXR1": "\u0259r",
    "AXR2": "\u0259r",
    "W": "w",
    "Y": "j"
}

cmudict_overrides = {
    # I just don't agree with the default, sorry
    "WAS": "w ʌ z",
    "WIND": "w ɪ n d",  # default is  waInd ???
    "PERMANENT": "p ɚ m ʌ n ʌ n t",
    "WAITED": "w ɛɪ t ɪ d",
    "USUALLY": "j u ʒ ʌ l i",
    "OCCASIONAL": "ʌ k ɛɪ ʒ ʌ n ʌ l",
    "PROJECTING": "p ɹ ʌ dʒ ɛ k t ɪ ŋ",
    "PROJECTED": "p ɹ ʌ dʒ ɛ k t ʌ d",

    # Missing pronunciation, but needed for Old Man and the Sea.
    "SALAO": "s æ l ɑ u",
    "FURLED": "f ɚ l d",
    "CREASED": "k ɹ i s t",
    "EROSIONS": "ɪ ɹ oʊ ʒ ʌ n z",
    "FISHLESS": "f ɪ ʃ l ʌ s",
    "BUDSHIELDS": "b ʌ d ʃ i l d z",
    "FIBERED": "f aɪ b ɚ d",
    "PERICO": "p v ɹ ɪ k oʊ",
    "BAREFOOTED": "b ɛ ɹ f ʊ t ʌ d",
    "HATUEY": "h æ t j ʌ i",
    "SISLER'S": "s ɪ s l ɚ z",
    "JOTA": "h oʊ t ʌ",
    "VA": "v ɑ",
    "OAKUM": "oʊ k ʌ m",
    "HARBOURS": "h ɑ ɹ b ɚ z",
    "ROADSTEADS": "ɹ oʊ d s t ɛ d z",
    "URINATED": "j ɚ ʌ n ɛɪ t ɪ d",
    "MANOLIN": "m æ n ʌ l ɪ n",
    "PEBBLED": "p ɛ b ʌ l d",
    "MOTORBOATS": "m oʊ t ɚ b oʊ t z",
    "ALBACORES": "æ l b ʌ k ɔ ɹ z",
    "INEFFECTUALLY": "ɪ n ʌ f ɛ k tʃ u ʌ l i",
    "WELTS": "w ɛ l t z",
    "FALSEST": "f ɑ l s ʌ s t",
    "CARAPACED": "k ɛ ɹ ʌ p ɛɪ s t",
    "GRIPPES": "g ɹ ɪ p s",
    "TUNA'S": "t u n ʌ z",
    "UNINTELLIGENT": "ʌ n ɪ n t ɛ l ʌ dʒ ʌ n t",
    "SARDINE'S": "s ɑ ɹ d i n z",
    "BITT": "b ɪ t",
    "PHOSPHORESCENT": "f ɑ s f ʌ ɹ ɛ s ʌ n t",
    "GAFFED": "g æ f t",
    "TREACHERIES": "t ɹ ɛ tʃ ɚ i s",
    "GUNWALE": "g ʌ n w ɛɪ l",
    "CARDEL": "k ɑ ɹ d ʌ l", # TODO typo in annotation? should be "cordel" right?
    "BROADBILL": "b ɹ ɔ d b ɪ l",
    "COAGULATED": "k oʊ æ g j ʌ l ɛɪ t ʌ d",
    "WINDLESS": "w ɪ n d l ʌ s",
    "LONGITUDINALLY": "l ɑ n dʒ ʌ t u d ʌ n ʌ l i",
}

ipa_chars = set(cmu_ipa_mapping.values())


class CMUPhonemizer:

    ipa_stress_length_re = re.compile(r"[ˈˌːˑ]")
    """Match unwanted stress/length markers"""

    dict_line_re = re.compile(r"(\w+)\t([^,]+)(?:,.*)?$")
    """Pattern to extract information from each dict line"""

    ipa_replacements = {
        r"ɝ": "ɚ",
    }
    """
    Modifications to make to the IPA annotation in the CMUdict IPA version
    in order to align with Heilbron& annotation.
    """

    # Sort diphthongs first so that re greedily picks these when matching
    ipa_phonemes_re = re.compile(
        "|".join(sorted(set(cmu_ipa_mapping.values()), key=len, reverse=True))
    )

    def __init__(self, mapping_path):
        self.mapping: Dict[str, List[str]] = {}

        # Parse into a sequence of sounds (possibly incorporating
        # IPA dipthongs).
        with open(mapping_path, 'r') as f:
            for line in f:
                matches = self.dict_line_re.findall(line)
                if matches:
                    word, pronunciation = matches[0]
                    pronunciation = self.ipa_stress_length_re.sub("", pronunciation.strip())
                    for pattern, replacement in self.ipa_replacements.items():
                        pronunciation = re.sub(pattern, replacement, pronunciation)
                    self.mapping[word.lower()] = tuple(self.ipa_phonemes_re.findall(pronunciation))

        self.missing_counter = Counter()

    @cache
    def __call__(self, string) -> List[str]:
        try:
            return self.mapping[string.lower()]
        except KeyError:
            self.missing_counter[string] += 1

            # HACK
            return self.ipa_phonemes_re.findall(string)