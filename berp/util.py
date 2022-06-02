from typing import Union, List, Tuple

import torch
from torchtyping import TensorType
from typeguard import typechecked

from berp.typing import DIMS

TT = TensorType


def sample_to_time(sample_idx: TT[torch.long],
                   sample_rate: int,
                   t_zero: float = 0
                   ) -> TT[float]:
    """
    Convert sample index representation to time representation (in seconds).
    """
    return t_zero + sample_idx / sample_rate


def time_to_sample(time: TT[float],
                   sample_rate: int,
                   t_zero: float = 0
                   ) -> TT[torch.long]:
    """
    Convert time representation (in seconds) to sample index representation.
    """
    # TODO meaningful difference between floor/ceil here?
    # Probably just consistency that matters.
    return torch.floor((time - t_zero) * sample_rate).long()


@typechecked
def variable_position_slice(x: torch.Tensor, idxs: torch.LongTensor,
                            slice_width: torch.LongTensor, padding_value=0.
                            ) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Extract fixed-width column slices from `x` with variable position by row,
    specified by `idxs`. Slices which are too close to the right edge of `x`
    to have `slice_width` items will be padded with `padding_value` and
    marked in the returned `mask`.

    Args:
        x: B * T * ...
        idxs: B

    Returns:
        sliced: B * slice_width
        mask: B * slice_width, cell ij is True iff corresponding cell ij of
            `sliced` is a valid member of `x` (and not extending past the
            right edge of `x`)
    """

    # Generate index range for each row.
    # TODO is there a better way to do this with real slice objects?
    slice_idxs = torch.arange(slice_width).tile((x.shape[0], 1)) \
        + idxs.unsqueeze(1)
    mask = slice_idxs < x.shape[1]
    # For invalid cells, just retrieve the first item.
    slice_idxs[~mask] = 0

    if x.ndim > 2:
        # Tile slice indices across remaining dimensions.
        viewer = (...,) + (None,) * (x.ndim - 2)
        slice_idxs = slice_idxs[viewer].tile((1, 1) + x.shape[2:])

    sliced = torch.gather(x, 1, slice_idxs)

    return sliced, mask


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
