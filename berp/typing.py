from typing_extensions import TypeAlias

from jaxtyping import AbstractDtype, Float
import torch


T: TypeAlias = torch.Tensor
FloatScalar: TypeAlias = Float[T, ""]


class Probability(AbstractDtype):
    dtypes = ["float16", "float32", "float64"]

class ProperProbability(Probability):
    pass

class LogProbability(AbstractDtype):
    dtypes = ["float16", "float32", "float64"]

class ProperLogProbability(LogProbability):
    pass

class PositiveFloat(AbstractDtype):
    dtypes = ["float16", "float32", "float64"]

class NonNegativeFloat(AbstractDtype):
    dtypes = ["float16", "float32", "float64"]