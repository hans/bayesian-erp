from dataclasses import dataclass
from typing import *

from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_optuna_sweeper.config import TPESamplerConfig, DistributionType


@dataclass
class DistributionConfig:

    # Type of distribution. "int", "float" or "categorical"
    type: DistributionType

    # Choices of categorical distribution
    # List element type should be Union[str, int, float, bool]
    choices: Optional[List[Any]] = None

    # Lower bound of int or float distribution
    low: Optional[float] = None

    # Upper bound of int or float distribution
    high: Optional[float] = None

    # If True, space is converted to the log domain
    # Valid for int or float distribution
    log: bool = False

    # Discritization step
    # Valid for int or float distribution
    step: Optional[float] = None

    # Shape for vectorized parameters
    shape: Optional[Tuple[int, ...]] = None

    _target_: str = "berp.cv.make_parameter_distributions"


@dataclass
class CVConfig:
    n_outer_folds: int
    n_inner_folds: int

    # Parameters for inner-loop cross-validation.
    param_sampler: TPESamplerConfig
    params: Dict[str, DistributionConfig]

    max_iter: int = 50
    n_trials: int = 20


cs = ConfigStore.instance()
cs.store(group="train_test", name="base_train_test", node=TrainTestConfig)
cs.store(group="cv", name="base_cv", node=CVConfig)