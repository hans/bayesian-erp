from dataclasses import dataclass
from typing import Dict

from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_optuna_sweeper.config import RandomSamplerConfig, DistributionConfig


@dataclass
class TrainTestConfig:
    series_hold_pct: float
    """
    Hold out data from this percentage of time series.
    """

    data_hold_pct: float
    """
    For selected time series, hold out this percentage of samples.
    """


@dataclass
class CVConfig:
    n_outer_folds: int
    n_inner_folds: int

    # Parameters for inner-loop cross-validation.
    param_sampler: RandomSamplerConfig
    params: Dict[str, DistributionConfig]


cs = ConfigStore.instance()
cs.store(group="train_test", name="base_train_test", node=TrainTestConfig)
cs.store(group="cv", name="base_cv", node=CVConfig)