from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore


GROUP = "viz"


@dataclass
class TensorboardConfig:

    log_dir: str = "."
    """
    Directory to store tensorboard logs. Relative to Hydra output path.
    """

    flush_secs: int = 60

    _target_: str = "berp.tensorboard.Tensorboard.instance"



@dataclass
class VizConfig:

    feature_patterns: List[str] = field(default_factory=lambda: [".*"])
    """
    List of regex patterns to match against feature names. Only features matching
    at least one of these patterns will be plotted.
    """

    tensorboard: TensorboardConfig = field(default_factory=lambda: TensorboardConfig())


cs = ConfigStore.instance()
cs.store(group=GROUP, name="base_viz", node=VizConfig)
cs.store(group=f"{GROUP}/tensorboard", name="base_tensorboard", node=TensorboardConfig)