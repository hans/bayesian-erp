from dataclasses import dataclass
from typing import List

import mne

from hydra.core.config_store import ConfigStore


GROUP = "dataset"

@dataclass
class DatasetConfig:
    paths: List[str]


@dataclass
class EEGDatasetConfig(DatasetConfig):
    montage_name: str

    @property
    def montage(self) -> mne.channels.DigMontage:
        return mne.channels.make_standard_montage(self.montage_name)


cs = ConfigStore.instance()
cs.store(group=GROUP, name="base_eeg_dataset", node=EEGDatasetConfig)