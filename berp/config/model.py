from dataclasses import dataclass
from typing import *

from hydra.core.config_store import ConfigStore

from berp.config.cv import DistributionConfig
from berp.config.solver import SolverConfig


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

    standardize_X: bool
    standardize_Y: bool

    warm_start: bool = True
    fit_intercept: bool = True
    type: str = "trf"

    optim: SolverConfig = "${solver}"

    _target_: str = "berp.models.trf.BerpTRF"


@dataclass
class BerpTRFEMModelConfig(ModelConfig):
    trf_config: TRFModelConfig

    latent_params: Dict[str, DistributionConfig]

    warm_start: bool = True
    fit_intercept: bool = True
    type: str = "trf_em"

    _target_: str = "berp.models.trf_em.BerpTRFEM"


cs = ConfigStore.instance()
cs.store(group=GROUP, name="base_trf", node=TRFModelConfig)
cs.store(group=GROUP, name="base_trf_em", node=BerpTRFEMModelConfig)