from dataclasses import dataclass
from typing import List, Dict, Optional

import mne

from hydra.core.config_store import ConfigStore


GROUP = "dataset"

@dataclass
class DatasetConfig:
    paths: List[str]
    stimulus_paths: Dict[str, str]

    encoder_key_re: str
    """
    A regular expression to apply on dataset names to determine how
    subdatasets should be mapped to encoders. Typically, it makes sense
    to use a regular expression here that picks out the subject ID.
    """

    device: str = "cpu"


@dataclass
class EEGDatasetConfig(DatasetConfig):

    subset_sensors: Optional[List[str]] = None
    """
    Optionally load just a subset of the dataset sensors. Each element
    is a reference to a montage channel name (not an index).
    """

    normalize_X_ts: bool = True
    normalize_X_variable: bool = True
    normalize_Y: bool = True

    special_normalize_variable_intercept: bool = False
    """
    If True, intercept values in `X_variable` are normalized such that,
    after scattering into the design matrix, the values are normally
    distributed. If False, no normalization is done to intercept values.
    """

    _target_: str = "berp.datasets.eeg.load_eeg_dataset"

    @property
    def montage(self) -> mne.channels.DigMontage:
        return mne.channels.make_standard_montage(self.montage_name)


cs = ConfigStore.instance()
cs.store(group=GROUP, name="base_eeg_dataset", node=EEGDatasetConfig)
