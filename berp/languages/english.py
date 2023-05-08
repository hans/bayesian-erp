"""
Defines data and utilities for English processing.
"""

from typing import List, Dict, Tuple

from collections import Counter
from functools import cache
import re

import pandas as pd
from tqdm.auto import tqdm


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
    "Y": "j",
}

cmudict_overrides = {
    # I just don't agree with the default, sorry
    "WAS": "ˈ w ʌ z",
    "WIND": "ˈ w ɪ n d",  # default is  waInd ???
    "PERMANENT": "ˈ p ɚ . m ʌ ˌ n ʌ n t",
    "WAITED": "ˈ w ɛɪ . t ɪ d",
    "USUALLY": "ˈ j u . ʒ ʌ . l i",
    "OCCASIONAL": ". ʌ ˈ k ɛɪ . ʒ ʌ . n ʌ l",
    "PROJECTING": ". p ɹ ʌ ˈ dʒ ɛ k . t ɪ ŋ",
    "PROJECTED": ". p ɹ ʌ ˈ dʒ ɛ k . t ʌ d",

    # Genuine variation, but align with Heilbron. Or sometimes pick our preferred
    # Def want to prefer to have the same number of phonemes as Heilbron, so that
    # we can align with existing TextGrid annotations
    "WITHOUT": ". w ɪ ˈ ð aʊ t",
    "AN": "ˈ ʌ n",
    "BEEN": "ˈ b ɪ n",
    "THAT": "ˈ ð æ t",
    "TO": "ˈ t u",
    "ALWAYS": "ˈ ɔ l ˌ w ɛɪ z",
    "CARRY": "ˈ k æ . ɹ i",
    "EITHER": "ˈ aɪ . ð ɚ",
    "OR": ". ɚ",
    "FLOUR": "ˈ f l aʊ ɹ",
    "HANDS": "ˈ h æ n z",
    "AS": ". ɛ z",
    "WITH": "ˈ w ɪ θ",
    "EVERY": "ˈ ɛ v . ɹ i",
    "FOR": ". f ɚ",
    "BECAUSE": ". b ɪ ˈ k ʌ z",

    # Missing pronunciation, but needed for Old Man and the Sea.
    "SALAO": ". s æ ˈ l ɑ u",
    "FURLED": "ˈ f ɚ l d",
    "CREASED": "ˈ k ɹ i s t",
    "EROSIONS": ". ɪ ˈ ɹ oʊ . ʒ ʌ n z",
    "FISHLESS": "ˈ f ɪ ʃ . l ʌ s",
    "BUDSHIELDS": "ˈ b ʌ d . ʃ i l d z",
    "FIBERED": "ˈ f aɪ . b ɚ d",
    "PERICO": "ˈ p v . ɹ ɪ . k oʊ",
    "BAREFOOTED": "ˈ b ɛ ɹ ˌ f ʊ t . ʌ d",
    "HATUEY": ". h æ ˈ t j ʌ i",
    "SISLER'S": "ˈ s ɪ s . l ɚ z",
    "JOTA": "ˈ h oʊ . t ʌ",
    "VA": "ˈ v ɑ",
    "OAKUM": "ˈ oʊ k . ʌ m",
    "HARBOURS": "ˈ h ɑ ɹ . b ɚ z",
    "ROADSTEADS": "ˈ ɹ oʊ d . s t ɛ d z",
    "URINATED": "ˈ j ɚ . ʌ . n ɛɪ . t ɪ d",
    "MANOLIN": "ˈ m æ . n ʌ . l ɪ n",
    "PEBBLED": "ˈ p ɛ . b ʌ l d",
    "MOTORBOATS": "ˈ m oʊ . t ɚ ˌ b oʊ t z",
    "ALBACORES": "ˈ æ l . b ʌ ˌ k ɔ ɹ z",
    "INEFFECTUALLY": "ˈ ɪ n ʌ ˌ f ɛ k . tʃ u ʌ . l i",
    "WELTS": "ˈ w ɛ l t z",
    "FALSEST": "ˈ f ɑ l . s ʌ s t",
    "CARAPACED": "ˈ k ɛ . ɹ ʌ . p ɛɪ s t",
    "GRIPPES": "ˈ g ɹ ɪ p s",
    "TUNA'S": "ˈ t u . n ʌ z",
    "UNINTELLIGENT": ". ʌ n ˌ ɪ n ˈ t ɛ . l ʌ . dʒ ʌ n t",
    "SARDINE'S": ". s ɑ ɹ ˈ d i n z",
    "BITT": "ˈ b ɪ t",
    "PHOSPHORESCENT": "ˈ f ɑ s ˌ f ʌ . ɹ ɛ . s ʌ n t",
    "GAFFED": "ˈ g æ f t",
    "TREACHERIES": "ˈ t ɹ ɛ . tʃ ɚ ˌ i s",
    "GUNWALE": "ˈ g ʌ n ˌ w ɛɪ l",
    "CARDEL": "ˈ k ɑ ɹ ˌ d ʌ l", # TODO typo in annotation? should be "cordel" right?
    "BROADBILL": "ˈ b ɹ ɔ d . b ɪ l",
    "COAGULATED": ". k oʊ ˈ æ . g j ʌ ˌ l ɛɪ . t ʌ d",
    "WINDLESS": "ˈ w ɪ n d . l ʌ s",
    "LONGITUDINALLY": "ˌ l ɑ n . dʒ ʌ ˈ t u . d ʌ . n ʌ . l i",
}

ipa_chars = set(cmu_ipa_mapping.values())


class Phonemizer:

    # Sort diphthongs first so that re greedily picks these when matching
    ipa_phonemes = set(cmu_ipa_mapping.values())
    ipa_phonemes_pat = "|".join(sorted(ipa_phonemes, key=len, reverse=True))
    ipa_phonemes_re = re.compile(ipa_phonemes_pat)
    valid_ipa_re = re.compile(r"^(?:" + ipa_phonemes_pat + r")+$")

    def __init__(self, mapping_df: pd.DataFrame):
        self.mapping: Dict[str, Tuple[str, ...]] = {}

        # Parse into a sequence of sounds (possibly incorporating
        # IPA dipthongs).
        # Take the first pronunciation idx for each word.
        mapping_df = mapping_df.sort_values(["word", "pronunciation_idx"])
        for word, pronunciation_rows in tqdm(mapping_df.groupby("word")):
            pronunciation = tuple(pronunciation_rows.pronunciation.iloc[0].split(" "))
            for phon in pronunciation:
                assert phon in self.ipa_phonemes, phon
            self.mapping[word.lower()] = pronunciation

        self.missing_counter = Counter()

    @cache
    def __call__(self, string) -> Tuple[str, ...]:
        try:
            return self.mapping[string.lower()]
        except KeyError:
            self.missing_counter[string] += 1

            # HACK
            return tuple(self.ipa_phonemes_re.findall(string))