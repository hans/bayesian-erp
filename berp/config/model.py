from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


GROUP = "model"

@dataclass
class ModelConfig:
    standardize_X: bool
    standardize_Y: bool


@dataclass
class TRFModelConfig(ModelConfig):
    tmin: float
    tmax: float
    sfreq: float
    alpha: float

    warm_start: bool = True
    fit_intercept: bool = True
    type: str = "trf"

    _target_: str = "berp.models.trf.TemporalReceptiveField"


@dataclass
class BerpTRFEMModelConfig(ModelConfig):
    tmin: float
    tmax: float
    sfreq: float
    alpha: float

    warm_start: bool = True
    fit_intercept: bool = True
    type: str = "trf_em"

    _target_: str = "berp.models.trf_em.BerpTRFEMEstimator"


cs = ConfigStore.instance()
cs.store(group=GROUP, name="base_trf", node=TRFModelConfig)
cs.store(group=GROUP, name="base_trf_em", node=BerpTRFEMModelConfig)