import pytest
import torch

from berp.generators.stimulus import StimulusGenerator


@pytest.fixture
def abstract_generator():
    return StimulusGenerator()


def _basic_draw_and_assert(abstract_generator, word_lengths, max_num_phonemes, **kwargs):
    n_words = len(word_lengths)
    max_num_phonemes = 6

    phoneme_onsets, phoneme_onsets_global, word_onsets, word_offsets = \
        abstract_generator.sample_stream(word_lengths, max_num_phonemes, **kwargs)
    assert phoneme_onsets.shape == (n_words, max_num_phonemes)
    assert phoneme_onsets_global.shape == (n_words, max_num_phonemes)
    assert word_onsets.shape == (n_words,)
    assert word_offsets.shape == (n_words,)

    # Test that the first onset is at the correct time
    assert word_onsets[0] == abstract_generator.first_onset
    assert phoneme_onsets_global[0, 0] == abstract_generator.first_onset

    # Test that the word onsets are in the correct order
    assert torch.all(word_onsets[1:] >= word_onsets[:-1])

    # Test that the phoneme onsets are in the correct order
    assert torch.all(phoneme_onsets_global[:, 1:] >= phoneme_onsets_global[:, :-1])

    return phoneme_onsets, phoneme_onsets_global, word_onsets, word_offsets


def test_sample_stream(abstract_generator):
    _basic_draw_and_assert(
        abstract_generator,
        word_lengths=torch.randint(1, 5, (500,)),
        max_num_phonemes=6)


def test_sample_stream_aligned(abstract_generator):
    sample_rate = 48
    phoneme_onsets, phoneme_onsets_global, word_onsets, word_offsets = \
        _basic_draw_and_assert(
            abstract_generator,
            word_lengths=torch.randint(1, 5, (500,)),
            max_num_phonemes=6,
            align_sample_rate=sample_rate)

    def assert_round(xs, inv_div):
        torch.testing.assert_allclose(xs, torch.round(xs * inv_div) / inv_div)

    assert_round(phoneme_onsets, sample_rate)
    assert_round(phoneme_onsets_global, sample_rate)
    assert_round(word_onsets, sample_rate)
    assert_round(word_offsets, sample_rate)