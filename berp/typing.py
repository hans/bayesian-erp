from typing import NamedTuple
import torch
from torch.distributions import constraints
from torchtyping import TensorType, TensorDetail, patch_typeguard

patch_typeguard()


class DIMS:
    """
    Defines standard names for tensor dimensions used across model types and
    scripts.
    """

    B = batch_size = "batch_size"

    N_W = n_words = "n_words"
    """
    Number of words in a given item.
    """

    N_C = n_candidates = "n_candidates"
    """
    The number of live candidate words represented in a given context.
    """

    N_F = n_features = "n_features"
    """
    Number of features (IVs) used to predict response values (DVs).
    """

    N_P = n_phonemes = "n_phonemes"
    """
    Maximum length of observed word, in phonemes
    """

    V_W = v_words = "v_words"
    """
    Size of word vocabulary
    """

    V_P = v_phonemes = "v_phonemes"
    """
    Size of phoneme vocabulary
    """

    # EEG dimensions
    T = n_times = "n_times"
    """
    Number of EEG samples
    """

    S = n_sensors = "n_sensors"
    """
    Number of EEG sensors
    """


class ProbabilityDetail(TensorDetail):
    def check(self, t: torch.Tensor) -> bool:
        return bool(constraints.unit_interval.check(t).all())

    def __repr__(self) -> str:
        return "ProbabilityDetail"

    @classmethod
    def tensor_repr(cls, t: torch.Tensor) -> str:
        return repr(t)

class ProperProbabilityDetail(ProbabilityDetail):
    def __init__(self, dim):
        self.dim = dim

    def _val(self, t: torch.Tensor) -> torch.Tensor:
        return t.sum(dim=self.dim)

    def check(self, t: torch.Tensor) -> bool:
        if not super().check(t):
            return False

        if not torch.allclose(self._val(t), torch.tensor(1.)):
            return False
        return True

    def __repr__(self) -> str:
        return f"ProperProbabilityDetail({self.dim})"

    @classmethod
    def tensor_repr(cls, t: torch.Tensor) -> str:
        return f"ProperProbabilityDetail({t})"

class LogProbabilityDetail(TensorDetail):
    def check(self, t: torch.Tensor) -> bool:
        return bool((t <= 0).all())

    def __repr__(self) -> str:
        return "LogProbabilityDetail"

    @classmethod
    def tensor_repr(cls, t: torch.Tensor) -> str:
        return repr(t)

class ProperLogProbabilityDetail(LogProbabilityDetail):
    def __init__(self, dim):
        self.dim = dim

    def _val(self, t: torch.Tensor) -> torch.Tensor:
        return t.logsumexp(dim=self.dim)

    def check(self, t: torch.Tensor) -> bool:
        if not super().check(t):
            return False

        if not torch.allclose(self._val(t), torch.tensor(0.)):
            return False
        return True

    def __repr__(self) -> str:
        return f"ProperLogProbabilityDetail({self.dim})"

    @classmethod
    def tensor_repr(cls, t: torch.Tensor) -> str:
        return f"ProperLogProbabilityDetail({t})"

class PositiveDetail(TensorDetail):
    def check(self, t: torch.Tensor) -> bool:
        return bool(constraints.positive.check(t).all())

    def __repr__(self) -> str:
        return "PositiveDetail"

    @classmethod
    def tensor_repr(cls, t: torch.Tensor) -> str:
        return repr(t)

is_probability = ProbabilityDetail()
is_log_probability = LogProbabilityDetail()
is_positive = PositiveDetail()

Probability = TensorType[float, is_probability]
LogProbability = TensorType[float, is_log_probability]
