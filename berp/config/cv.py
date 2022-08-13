from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


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
    pass


cs = ConfigStore.instance()
cs.store(group="train_test", name="base_train_test", node=TrainTestConfig)
cs.store(group="cv", name="base_cv", node=CVConfig)