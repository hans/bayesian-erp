from typing import Union, List, Tuple

import torch
from torchtyping import TensorType
from typeguard import typechecked

from berp.typing import DIMS

TT = TensorType


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


# # TODO untested
# @typechecked
# def pad_and_mask(batch: List[TT],
#                  lengths: Union[List[int], TT[int]],
#                  padding_value=0.
#                  ) -> Tuple[TT[DIMS.B, "times", ...],
#                             TT[DIMS.B, "times", bool]]:
#     padded_batch = torch.nn.utils.rnn.pad_sequence(
#         batch, batch_first=True, padding_value=padding_value)
#
#     mask = torch.arange(padded_batch.shape[1]).view(1, -1) \
#         .tile((padded_batch.shape[0], 1))
#     mask = mask >= torch.tensor(lengths)
#     ic(mask)
#
#     return padded_batch, mask
