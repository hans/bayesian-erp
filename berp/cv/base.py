from copy import deepcopy
from typing import *
from uuid import uuid4

from hydra_plugins.hydra_optuna_sweeper.config import DistributionType
import numpy as np
from optuna.distributions import (
    BaseDistribution,
    CategoricalChoiceType,
    CategoricalDistribution,
    DiscreteUniformDistribution,
    IntLogUniformDistribution,
    IntUniformDistribution,
    LogUniformDistribution,
    UniformDistribution,
)

from berp.config.cv import DistributionConfig


def make_parameter_distribution(**kwargs) -> BaseDistribution:
    assert kwargs.get("shape") is None, \
        "Multivariate parameters need to be prepared with make_parameter_distributions (note S!)"
    from pprint import pprint; pprint(kwargs)
    if isinstance(kwargs["type"], str):
        kwargs["type"] = DistributionType[kwargs["type"]]
    param = DistributionConfig(**kwargs)
    if param.type == DistributionType.categorical:
        assert param.choices is not None
        return CategoricalDistribution(param.choices)
    if param.type == DistributionType.int:
        assert param.low is not None
        assert param.high is not None
        if param.log:
            return IntLogUniformDistribution(int(param.low), int(param.high))
        step = int(param.step) if param.step is not None else 1
        return IntUniformDistribution(int(param.low), int(param.high), step=step)
    if param.type == DistributionType.float:
        assert param.low is not None
        assert param.high is not None
        if param.log:
            return LogUniformDistribution(param.low, param.high)
        if param.step is not None:
            return DiscreteUniformDistribution(param.low, param.high, param.step)
        return UniformDistribution(param.low, param.high)
    raise NotImplementedError(f"{param.type} is not supported by Optuna sweeper.")


def make_parameter_distributions(name: Optional[str] = None, shape=None, **cfg
                                 ) -> Dict[str, BaseDistribution]:
    """
    Generate parameter distribution(s) for consumption by Optuna.
    Supports multivariate parameters.
    """
    if name is None:
        name = uuid4().hex

    if shape is not None:
        sub_cfg = deepcopy(cfg)
        sub_cfg["shape"] = None

        return {
            f"V{name}/{'_'.join(map(str, idx))}": make_parameter_distribution(sub_cfg)
            for idx in np.ndindex(*shape)
        }

        # # We will replace the name but respect sklearn's hierarchical repr.
        # name_parts = name.rsplit("__", 1)
        # name_prefix = name_parts[0] + "__" if len(name_parts) > 1 else ""
        # target_name = name_parts[1]

        # return {
        #     f"{name_prefix}V{target_name}/{'_'.join(map(str, idx))}": make_parameter_distribution(sub_cfg)
        #     for idx in np.ndindex(*cfg.shape)
        # }

    return {name: make_parameter_distribution(**cfg)}


class EarlyStopException(Exception):
    """
    Raised when a fit decides to early stop, and signals no more fitting on this fold
    is necessary.
    """
    pass