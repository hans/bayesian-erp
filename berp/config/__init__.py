from dataclasses import dataclass
from typing import List

from hydra.core.config_store import ConfigStore

from berp.config.cv import TrainTestConfig, CVConfig
from berp.config.model import ModelConfig
from berp.config.solver import SolverConfig


@dataclass
class Config:
    model: ModelConfig
    solver: SolverConfig
    datasets: List[str]

    train_test: TrainTestConfig
    cv: CVConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)