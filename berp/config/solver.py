from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


GROUP = "solver"


@dataclass
class SolverConfig:
    pass


@dataclass
class SGDSolverConfig(SolverConfig):
    learning_rate: float
    n_iter: int


@dataclass
class AdamSolverConfig(SGDSolverConfig):
    beta_1: float
    beta_2: float
    epsilon: float

    type: str = "adam"


@dataclass
class SVDSolverConfig(SolverConfig):
    type: str = "svd"


cs = ConfigStore.instance()
cs.store(group=GROUP, name="base_sgd", node=SGDSolverConfig)
cs.store(group=GROUP, name="base_adam", node=AdamSolverConfig)
cs.store(group=GROUP, name="base_svd", node=SVDSolverConfig)