from typing import NamedTuple
from typing_extensions import TypeAlias
import torch
from torch.distributions import constraints
from torchtyping import TensorType, TensorDetail, patch_typeguard

patch_typeguard()


Phoneme: TypeAlias = str


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

    N_F = n_variable_features = "n_variable_features"
    """
    Number of variable-onset features (V-IVs) used to predict response values (DVs).
    """

    N_F_T = n_time_series_features = "n_time_series_features"
    """
    Number of time-series features (sampled at same rate as EEG signal) used to predict
    response values (DVs).
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


class FloatingDetail(TensorDetail):
    def check(self, t: torch.Tensor) -> bool:
        return torch.is_floating_point(t)

    def __repr__(self) -> str:
        return "FloatingDetail"
    
    @classmethod
    def tensor_repr(cls, t: torch.Tensor) -> str:
        return repr(t)

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


class ConstraintDetail(TensorDetail):
    """
    Abstract type detail which defers to torch distribution constraints
    """

    def __init__(self, constraint):
        self.constraint = constraint

    def check(self, t: torch.Tensor) -> bool:
        return bool(self.constraint.check(t).all())

    def __repr__(self) -> str:
        return f"ConstraintDetail({self.constraint})"

    @classmethod
    def tensor_repr(cls, t: torch.Tensor) -> str:
        return repr(t)


floating = FloatingDetail()
is_probability = ProbabilityDetail()
is_log_probability = LogProbabilityDetail()
is_positive = ConstraintDetail(constraints.positive)
is_nonnegative = ConstraintDetail(constraints.nonnegative)

Probability = TensorType[float, is_probability]
LogProbability = TensorType[float, is_log_probability]
