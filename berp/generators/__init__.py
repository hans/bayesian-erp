from typing import Tuple, NamedTuple, Optional, List

from torchtyping import TensorType

from berp.models import reindexing_regression as rr
from berp.typing import DIMS, is_log_probability

# Type variables
B, N_W, N_C, N_F, N_P, V_W = DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_P, DIMS.V_W
T, S = DIMS.T, DIMS.S


class RRDriver(object):
    """
    Defines a driver class for executing the reindexing regression model
    on a particular generated dataset."""

    pass


class RRDataset(NamedTuple):
    """
    Defines an epoched dataset for reindexing regression, generated with the given
    ground-truth parameters ``params``.

    Each element on the batch axis corresponds to a single word within a single
    item.

    All tensors are padded on the N_P axis on the right to the maximum word length.
    """

    params: rr.ModelParameters
    sample_rate: int
    epoch_window: Tuple[float, float]

    phonemes: List[str]
    """
    Phoneme vocabulary.
    """

    p_word: TensorType[B, N_C, is_log_probability]
    """
    Predictive distribution over expected candidate words at each time step,
    derived from a language model.
    """

    word_lengths: TensorType[B, int]
    """
    Length of ground-truth words in phonemes. Can be used to unpack padded
    ``N_P`` axes.
    """

    candidate_phonemes: TensorType[B, N_C, N_P, int]
    """
    Phoneme ID sequence for each word and alternate candidate set.
    """

    word_onsets: TensorType[B, float]
    """
    Onset of each word in seconds, relative to the start of the sequence.
    """

    phoneme_onsets: TensorType[B, N_P, float]
    """
    Onset of each phoneme within each word in seconds, relative to the start of
    the corresponding word.
    """

    recognition_points: TensorType[B, int]
    """
    Ground-truth recognition points (phoneme indices) for each word.
    Useful for debugging.
    """

    recognition_onsets: TensorType[B, float]
    """
    Ground-truth recognition onset (seconds, relative to word onset) for each
    word. Useful for debugging.
    """

    X_epoch: TensorType[B, N_F, float]
    """
    Epoch features.
    """

    Y_epoch: TensorType[B, T, S, float]
    """
    Epoched response data.
    """

    Y: Optional[TensorType[..., S, float]] = None
    """
    Original response data stream.
    """