import torch
from torch.distributions import constraints
from torchtyping import TensorType, TensorDetail, patch_typeguard

patch_typeguard()


class ProbabilityDetail(TensorDetail):
    def check(self, t: torch.Tensor) -> bool:
        return constraints.unit_interval.check(t).all()

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

        if not torch.allclose(self._val(), torch.tensor(1.)):
            return False

    def __repr__(self) -> str:
        return f"ProperProbabilityDetail({self.dim})"

    def tensor_repr(self, t: torch.Tensor) -> str:
        return f"ProperProbabilityDetail({self._val(t)})"

class LogProbabilityDetail(TensorDetail):
    def check(self, t: torch.Tensor) -> bool:
        return (t <= 0).all()

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

    def __repr__(self) -> str:
        return f"ProperLogProbabilityDetail({self.dim})"

    def tensor_repr(self, t: torch.Tensor) -> str:
        return f"ProperLogProbabilityDetail({self._val(t)})"

is_probability = ProbabilityDetail()
is_log_probability = LogProbabilityDetail()

Probability = TensorType[float, is_probability]
LogProbability = TensorType[float, is_log_probability]
