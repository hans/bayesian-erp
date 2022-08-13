from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


GROUP = "model"

@dataclass
class ModelConfig:
    pass


@dataclass
class TRFModelConfig(ModelConfig):
    tmin: float
    tmax: float
    sfreq: float
    alpha: float

    warm_start: bool = True
    fit_intercept: bool = True
    type: str = "trf"


cs = ConfigStore.instance()
cs.store(group=GROUP, name="base_trf", node=TRFModelConfig)