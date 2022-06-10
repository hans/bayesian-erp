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