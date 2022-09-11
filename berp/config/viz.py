from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore


GROUP = "viz"

@dataclass
class VizConfig:

    feature_patterns: List[str] = field(default_factory=lambda: [".*"])
    """
    List of regex patterns to match against feature names. Only features matching
    at least one of these patterns will be plotted.
    """


cs = ConfigStore.instance()
cs.store(group=GROUP, name="base_viz", node=VizConfig)