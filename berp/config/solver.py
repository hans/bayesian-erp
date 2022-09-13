from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore


GROUP = "solver"


@dataclass
class SolverConfig:
    pass


@dataclass
class SGDSolverConfig(SolverConfig):
    learning_rate: float = 0.01
    n_batches: int = 1
    batch_size: int = 512
    early_stopping: Optional[int] = 5


@dataclass
class AdamSolverConfig(SGDSolverConfig):
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8

    type: str = "adam"
    _target_: str = "berp.solvers.AdamSolver"


@dataclass
class SVDSolverConfig(SolverConfig):
    type: str = "svd"


cs = ConfigStore.instance()
cs.store(group=GROUP, name="base_sgd", node=SGDSolverConfig)
cs.store(group=GROUP, name="base_adam", node=AdamSolverConfig)
cs.store(group=GROUP, name="base_svd", node=SVDSolverConfig)