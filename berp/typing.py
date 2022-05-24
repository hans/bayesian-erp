import torch
from torch.distributions import constraints
from torchtyping import TensorType, TensorDetail


class ProbabilityDetail(TensorDetail):
    def check(self, t: torch.Tensor) -> bool:
        return constraints.unit_interval.check(t)

    def __repr__(self) -> str:
        return "ProbabilityDetail"

    @classmethod
    def tensor_repr(cls, t: torch.Tensor) -> str:
        return repr(t)

class LogProbabilityDetail(TensorDetail):
    def check(self, t: torch.Tensor) -> bool:
        return t <= 0

    def __repr__(self) -> str:
        return "LogProbabilityDetail"

    @classmethod
    def tensor_repr(cls, t: torch.Tensor) -> str:
        return repr(t)

is_probability = ProbabilityDetail()
is_log_probability = LogProbabilityDetail()

Probability = TensorType[float, is_probability]
LogProbability = TensorType[float, is_log_probability]
