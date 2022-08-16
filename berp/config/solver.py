from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


GROUP = "solver"


@dataclass
class SolverConfig:
    pass


@dataclass
class SGDSolverConfig(SolverConfig):
    learning_rate: float = 0.01
    n_epochs: int = 1
    batch_size: int = 512


@dataclass
class AdamSolverConfig(SGDSolverConfig):
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8

    type: str = "adam"
    _target_: str = "berp.models.trf.AdamSolver"


@dataclass
class SVDSolverConfig(SolverConfig):
    type: str = "svd"


cs = ConfigStore.instance()
cs.store(group=GROUP, name="base_sgd", node=SGDSolverConfig)
cs.store(group=GROUP, name="base_adam", node=AdamSolverConfig)
cs.store(group=GROUP, name="base_svd", node=SVDSolverConfig)