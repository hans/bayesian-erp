from dataclasses import dataclass
from typing import List, Optional

import mne

from hydra.core.config_store import ConfigStore


GROUP = "dataset"

@dataclass
class DatasetConfig:
    paths: List[str]


@dataclass
class EEGDatasetConfig(DatasetConfig):
    montage_name: str

    subset_sensors: Optional[List[str]] = None
    """
    Optionally load just a subset of the dataset sensors. Each element
    is a reference to a montage channel name (not an index).
    """

    normalize_X_ts: bool = True
    normalize_X_variable: bool = True
    normalize_Y: bool = True

    drop_X_variable: Optional[List[str]] = None

    _target_: str = "berp.datasets.eeg.load_eeg_dataset"

    @property
    def montage(self) -> mne.channels.DigMontage:
        return mne.channels.make_standard_montage(self.montage_name)


cs = ConfigStore.instance()
cs.store(group=GROUP, name="base_eeg_dataset", node=EEGDatasetConfig)
