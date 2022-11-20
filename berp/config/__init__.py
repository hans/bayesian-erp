from dataclasses import dataclass
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from berp.config.cv import CVConfig
from berp.config.dataset import DatasetConfig
from berp.config.model import ModelConfig, FeatureConfig
from berp.config.solver import SolverConfig
from berp.config.viz import VizConfig


# Add interpolators for configs
OmegaConf.register_new_resolver("eval", eval)


@dataclass
class Config:
    model: ModelConfig
    features: FeatureConfig

    solver: SolverConfig
    dataset: DatasetConfig

    cv: CVConfig

    viz: VizConfig

    baseline_model_path: Optional[str] = None


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


__all__ = [
    "Config",
    "CVConfig",
    "DatasetConfig",
    "ModelConfig",
    "SolverConfig",
    "VizConfig",
]