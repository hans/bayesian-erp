import logging
from typing import Union, List, Tuple

import numpy as np
import scipy.signal
import torch
from torch.nn.parallel.comm import gather
from torchtyping import TensorType
from typeguard import typechecked

L = logging.getLogger(__name__)


def sample_to_time(sample_idx: torch.LongTensor,
                   sample_rate: int,
                   t_zero: float = 0
                   ) -> TensorType[float]:
    """
    Convert sample index representation to time representation (in seconds).
    """
    return t_zero + sample_idx / sample_rate


def time_to_sample(time: TensorType[float],
                   sample_rate: int,
                   t_zero: float = 0
                   ) -> TensorType[torch.long]:
    """
    Convert time representation (in seconds) to sample index representation.
    """
    return torch.round((time - t_zero) * sample_rate).long()


def gaussian_window(center: float, width: float,\
                    start: float = 0,
                    end: float = 1,
                    sample_rate=128):
    """Gaussian window :class:`NDVar`
    Parameters
    ----------
    center : scalar
        Center of the window (normalized to the closest sample on ``time``).
    width : scalar
        Standard deviation of the window.
    time : UTS
        Time dimension.
    Returns
    -------
    gaussian : NDVar
        Gaussian window on ``time``.
    """

    n_samples = int((end - start) * sample_rate) + 1
    times, step_size = np.linspace(start, end, n_samples, retstep=True, endpoint=True)
    width_i = int(round(width / step_size))
    n_times = len(times)
    center_i = (center - start) // step_size

    slice_start, slice_stop = None, None
    if center_i >= n_times / 2:
        slice_start = None
        slice_stop = n_times
        window_width = 2 * center_i + 1
    else:
        slice_start = -n_times
        slice_stop = None
        window_width = 2 * (n_times - center_i) - 1
    window_data = scipy.signal.windows.gaussian(window_width, width_i)
    window_data = window_data[slice_start: slice_stop]
    return times, window_data


def cat(tensors, dim=0, **kwargs):
    """
    Concatenate tensors along the given dimension, sensitive to their possibly
    different device locatins.
    """
    if not any(t.is_cuda for t in tensors):
        return torch.cat(tensors, dim=dim, **kwargs)
    else:
        return gather(tensors, dim=dim, **kwargs)