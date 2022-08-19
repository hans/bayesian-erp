"""
This module defines the logic for converting a raw force-aligned transcript
dataset into a dataset ready for reindexing regression.
"""


from berp.datasets.processor.base import NaturalLanguageStimulusProcessor, \
    NaturalLanguageStimulus


__all__ = [
    "NaturalLanguageStimulusProcessor",
    "NaturalLanguageStimulus",
]