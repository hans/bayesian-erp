"""
Defines data and utilities for English processing.
"""

from typing import List, Dict, Tuple
from typing_extensions import TypeAlias

from collections import Counter
from functools import cache
import re

import pandas as pd
from tqdm.auto import tqdm

Phoneme: TypeAlias = str


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
syllable_stress_chars = {".": 0, "ˈ": 1, "ˌ": 2}


class Phonemizer:

    # Sort diphthongs first so that re greedily picks these when matching
    ipa_phonemes = set(cmu_ipa_mapping.values())
    ipa_phonemes_pat = "|".join(sorted(ipa_phonemes, key=len, reverse=True))
    ipa_phonemes_re = re.compile(ipa_phonemes_pat)
    valid_ipa_re = re.compile(r"^(?:" + ipa_phonemes_pat + r")+$")

    def __init__(self, mapping_df: pd.DataFrame):
        self.mapping: Dict[str, Tuple[str, ...]] = {}
        # Mapping between word forms and their IPA pronunciations

        self.syllables: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        # Mapping between word forms and list of syllable descriptors,
        # each a phoneme index (syllable onset phoneme) and a syllable
        # type (0 = unstressed, 1 = primary stress, 2 = secondary stress)

        # Parse into a sequence of sounds (possibly incorporating
        # IPA dipthongs).
        # Take the first pronunciation idx for each word.
        mapping_df = mapping_df.sort_values(["word", "pronunciation_idx"])
        for word, pronunciation_rows in tqdm(mapping_df.groupby("word")):
            pronunciation = tuple(pronunciation_rows.pronunciation_syllable.iloc[0].split(" "))

            i = 0
            syllables = []
            pronunciation_chars = []
            for phon in pronunciation:
                if phon in syllable_stress_chars:
                    syllables.append((i, syllable_stress_chars[phon]))
                else:
                    assert phon in self.ipa_phonemes, phon
                    pronunciation_chars.append(phon)
                    i += 1

            self.mapping[word.lower()] = tuple(pronunciation_chars)
            self.syllables[word.lower()] = tuple(syllables)

        self.missing_counter = Counter()

    @cache
    def __call__(self, string) -> Tuple[str, ...]:
        try:
            return self.mapping[string.lower()]
        except KeyError:
            self.missing_counter[string] += 1

            # HACK
            return tuple(self.ipa_phonemes_re.findall(string))


from nltk.tokenize.api import TokenizerI
from nltk.util import ngrams
from string import punctuation
import warnings

class IPASyllableTokenizer(TokenizerI):
    # Based on NLTK SyllableTokenizer

    english_ipa_sonority_hierarchy = [
        ['aɪ', 'aʊ', 'i', 'oʊ', 'u', 'æ', 'ɑ', 'ɔ', 'ɔɪ', 'ɚ', 'ɛ', 'ɛɪ', 'ɪ', 'ʊ', 'ʌ',], # vowels
        ['j', 'l', 'm', 'n', 'w', 'ŋ', 'ɹ'], # nasals and approximants
        ['f', 'h', 's', 'v', 'z', 'ð', 'ʃ', 'ʒ', 'θ'], # fricatives
        ['b', 'd', 'dʒ', 'g', 'k', 'p', 't', 'tʃ'], # stops and affricates
    ]

    def __init__(self, sonority_hierarchy=None):
        """
        :param lang: Language parameter, default is English, 'en'
        :type lang: str
        :param sonority_hierarchy: Sonority hierarchy according to the
                                   Sonority Sequencing Principle.
        :type sonority_hierarchy: list(str)
        """
        sonority_hierarchy = sonority_hierarchy or self.english_ipa_sonority_hierarchy

        self.vowels = sonority_hierarchy[0]
        self.phoneme_map = {}
        for i, level in enumerate(sonority_hierarchy):
            for c in level:
                sonority_level = len(sonority_hierarchy) - i
                self.phoneme_map[c] = sonority_level
                self.phoneme_map[c.upper()] = sonority_level


    def assign_values(self, token):
        """
        Assigns each phoneme its value from the sonority hierarchy.
        Note: Sentence/text has to be tokenized first.

        :param token: Single word or token
        :type token: str
        :return: List of tuples, first element is character/phoneme and
                 second is the soronity value.
        :rtype: list(tuple(str, int))
        """
        syllables_values = []
        for c in token:
            try:
                syllables_values.append((c, self.phoneme_map[c]))
            except KeyError:
                if c not in "0123456789" and c not in punctuation:
                    warnings.warn(
                        "Character not defined in sonority_hierarchy,"
                        " assigning as vowel: '{}'".format(c)
                    )
                    syllables_values.append((c, max(self.phoneme_map.values())))
                    if c not in self.vowels:
                        self.vowels += c
                else:  # If it's a punctuation or numbers, assign -1.
                    syllables_values.append((c, -1))
        return syllables_values


    def validate_syllables(self, syllable_list):
        """
        Ensures each syllable has at least one vowel.
        If the following syllable doesn't have vowel, add it to the current one.

        :param syllable_list: Single word or token broken up into syllables.
        :type syllable_list: list(str)
        :return: Single word or token broken up into syllables
                 (with added syllables if necessary)
        :rtype: list(str)
        """
        valid_syllables = []
        front = []
        vowel_pattern = re.compile("|".join(self.vowels))
        for i, syllable in enumerate(syllable_list):
            syllable_str = "".join(syllable)
            if syllable_str in punctuation:
                valid_syllables.append(syllable)
                continue
            if not vowel_pattern.search(syllable_str):
                if len(valid_syllables) == 0:
                    front += syllable
                else:
                    valid_syllables = valid_syllables[:-1] + [
                        valid_syllables[-1] + syllable
                    ]
            else:
                if len(valid_syllables) == 0:
                    valid_syllables.append(front + syllable)
                else:
                    valid_syllables.append(syllable)

        return [tuple(phoneme_seq) for phoneme_seq in valid_syllables]


    def tokenize(self, token) -> List[Tuple[Phoneme, ...]]:
        """
        Apply the SSP to return a list of syllables.
        Note: Sentence/text has to be tokenized first.

        :param token: Single word or token
        :type token: str
        :return syllable_list: Single word or token broken up into syllables.
        """
        # assign values from hierarchy
        syllables_values = self.assign_values(token)

        # if only one vowel return word
        if sum(token.count(x) for x in self.vowels) <= 1:
            return [tuple(token)]

        syllable_list = []
        syllable = [syllables_values[0][0]]  # start syllable with first phoneme
        for trigram in ngrams(syllables_values, n=3):
            phonemes, values = zip(*trigram)
            # Sonority of previous, focal and following phoneme
            prev_value, focal_value, next_value = values
            # Focal phoneme.
            focal_phoneme = phonemes[1]

            # These cases trigger syllable break.
            if focal_value == -1:  # If it's a punctuation, just break.
                syllable_list.append(syllable)
                syllable_list.append(focal_phoneme)
                syllable = []
            elif prev_value >= focal_value == next_value:
                syllable.append(focal_phoneme)
                syllable_list.append(syllable)
                syllable = []

            elif prev_value > focal_value < next_value:
                syllable_list.append(syllable)
                syllable = []
                syllable.append(focal_phoneme)

            # no syllable break
            else:
                syllable.append(focal_phoneme)

        syllable.append(syllables_values[-1][0])  # append last phoneme
        syllable_list.append(syllable)

        return self.validate_syllables(syllable_list)