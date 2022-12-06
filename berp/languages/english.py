"""
Defines data and utilities for English processing.
"""

from collections import Counter
from functools import cache
import re

# Maps CMU dict pronunciation elements to IPA.
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
    "EY": "e\u026a",
    "EY0": "e\u026a",
    "EY1": "e\u026a",
    "EY2": "e\u026a",
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
    "R": "r",  # NB condense to /r/, not \u0279
    "ER": "\u025d",
    "ER0": "\u025d",
    "ER1": "\u025d",
    "ER2": "\u025d",
    "AXR": "\u0259r",
    "AXR0": "\u0259r",
    "AXR1": "\u0259r",
    "AXR2": "\u0259r",
    "W": "w",
    "Y": "j"
}

ipa_chars = set(cmu_ipa_mapping.values())


class CMUPhonemizer:

    ipa_stress_length_re = re.compile(r"[ˈˌːˑ]")
    """Match unwanted stress/length markers"""

    dict_line_re = re.compile(r"(\w+)\t([^,]+)(?:,.*)?$")
    """Pattern to extract information from each dict line"""

    # Sort diphthongs first so that re greedily picks these when matching
    ipa_phonemes_re = re.compile(
        "|".join(sorted(set(cmu_ipa_mapping.values()), key=len, reverse=True))
    )

    def __init__(self, mapping_path):
        self.mapping = {}

        # Parse into a sequence of sounds (possibly incorporating
        # IPA dipthongs).
        with open(mapping_path, 'r') as f:
            for line in f:
                matches = self.dict_line_re.findall(line)
                if matches:
                    word, pronunciation = matches[0]
                    self.mapping[word.lower()] = \
                        self.ipa_stress_length_re.sub("", pronunciation.strip())

        self.missing_counter = Counter()

    @cache
    def __call__(self, string):
        try:
            return self.mapping[string.lower()]
        except KeyError:
            self.missing_counter[string] += 1

            # TODO
            return string