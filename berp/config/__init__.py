from dataclasses import dataclass
from typing import List

from hydra.core.config_store import ConfigStore

from berp.config.model import ModelConfig
from berp.config.solver import SolverConfig


@dataclass
class Config:
    model: ModelConfig
    solver: SolverConfig
    datasets: List[str]


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)