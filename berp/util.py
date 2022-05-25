from typing import Union

import torch
from torchtyping import TensorType


def sample_to_time(sample_idx: Union[int, TensorType[torch.long]],
                   sample_rate: Union[int, TensorType[int]],
                   t_zero: Union[float, TensorType[float]] = 0
                   ) -> Union[float, TensorType[float]]:
    """
    Convert sample index representation to time representation (in seconds).
    """
    return t_zero + sample_idx / sample_rate


def time_to_sample(time: Union[float, TensorType[float]],
                   sample_rate: Union[int, TensorType[int]],
                   t_zero: Union[float, TensorType[float]] = 0
                   ) -> Union[int, TensorType[torch.long]]:
    """
    Convert time representation (in seconds) to sample index representation.
    """
    # TODO meaningful difference between floor/ceil here?
    # Probably just consistency that matters.
    return torch.floor((time - t_zero) * sample_rate).long()
