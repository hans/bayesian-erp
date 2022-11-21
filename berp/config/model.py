from dataclasses import dataclass, field
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
    n_outputs: int
    alpha: float

    warm_start: bool = True
    fit_intercept: bool = True
    type: str = "trf"

    optim: SolverConfig = "${solver}"  # type: ignore
    # TODO any clean way to do structured config interpolation without type errors?

    _target_: str = "berp.models.trf.TemporalReceptiveField"

@dataclass
class TRFPipelineConfig(ModelConfig):
    trf: TRFModelConfig

    _target_: str = "berp.models.trf_em.BasicTRF"


@dataclass
class BerpTRFModelConfig(ModelConfig):
    """
    Abstract model config structure shared between subtypes. Subtypes
    vary in specific estimation technique.
    """

    trf: TRFModelConfig

    scatter_point: float = 0.0
    prior_scatter_index: int = 0
    prior_scatter_point: float = 0.0

    variable_trf_zero_left: int = 0
    """
    For variable-onset features, enforce that this many samples starting
    from the left egde of the TRF are zeroed out. This effectively narrows 
    width of the TRF window for these features.
    """
    variable_trf_zero_right: int = 0
    """
    For variable-onset features, enforce that this many samples starting
    from the right egde of the TRF are zeroed out. This effectively narrows 
    width of the TRF window for these features.
    """

    confusion_path: Optional[str] = None
    """
    Path to a confusion matrix. Must be compatible with the phoneme
    vocabulary declared in the dataset.
    """

    pretrained_pipeline_paths: Optional[List[str]] = None
    """
    Optional path to a pretrained pipeline to use to initialize this model.
    (Depending on the type of pipeline this will have different effects.)
    For a vanilla pipeline, the encoder coefficents and optimal regularization
    parameters will be used as initialization and fixed hyperparameters,
    respectively.
    """


@dataclass
class BerpTRFFixedModelConfig(BerpTRFModelConfig):

    threshold: float = 0.5

    type: str = "trf-berp-fixed"
    _target_: str = "berp.models.trf_em.BerpTRFFixed"


@dataclass
class BerpCannonTRFModelConfig(BerpTRFModelConfig):

    threshold: float = 0.5
    n_quantiles: int = 3

    type: str = "trf-berp-cannon"
    _target_: str = "berp.models.trf_em.BerpTRFCannon"


@dataclass
class BerpTRFEMModelConfig(BerpTRFModelConfig):

    latent_params: Dict[str, DistributionConfig] = field(default_factory=dict)

    n_iter: int = 1
    """
    Maximum number of EM iterations to run.
    """

    early_stopping: Optional[int] = 1
    """
    Number of EM iterations to tolerate no improvement in validation
    loss before stopping. If `None`, do not early stop.
    """

    warm_start: bool = True
    fit_intercept: bool = True
    type: str = "trf-em"

    _target_: str = "berp.models.trf_em.BerpTRFEM"


@dataclass
class FeatureConfig:
    """
    Specifies the dataset features that a pipeline should draw on /
    feed to a model.
    """

    ts_feature_names: List[str]
    """
    Names of the time series features to extract from a dataset. These
    correspond to the first N columns of the model design matrix.
    """

    variable_feature_names: List[str]
    """
    Names of the variable onset features to extract from a dataset. These
    will be used to produce some number of variable-onset predictors,
    depending on the model.
    """


cs = ConfigStore.instance()
cs.store(group=GROUP, name="base_trf", node=TRFModelConfig)
cs.store(group=GROUP, name="base_trf_pipeline", node=TRFPipelineConfig)
cs.store(group=GROUP, name="base_trf_em", node=BerpTRFEMModelConfig)
cs.store(group=GROUP, name="base_trf_berp_fixed", node=BerpTRFFixedModelConfig)
cs.store(group=GROUP, name="base_trf_berp_cannon", node=BerpCannonTRFModelConfig)