"""
Defines data splitters for the Berp dataset.

Splitting data needs to be sensitive to the fact that predictions /
learning signals at the edge of resulting splits are not going to be
usable.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.utils import check_random_state
from torchtyping import TensorType
from typeguard import typechecked

from berp.config import TrainTestConfig
from berp.datasets.base import BerpDataset


@dataclass
class BerpTrainTestSplitter(object):

    def __init__(self, cfg: TrainTestConfig, random_state=None):
        self.cfg = cfg
        self.random_state = random_state

    @typechecked
    def split(self, datasets: List[BerpDataset]) -> Tuple[List[BerpDataset], List[BerpDataset]]:
        """
        Returns:
            (train, test)
        """

        # Shape checks. Everything but batch axis should match across
        # subjects. Batch axis should match within-subject between
        # X and Y.
        for dataset in datasets:
            assert dataset.X_ts.shape[1:] == datasets[0].X_ts.shape[1:]
            assert dataset.X_variable.shape[1:] == datasets[0].X_variable.shape[1:]
            assert dataset.Y.shape[1:] == datasets[0].Y.shape[1:]
            assert dataset.X_ts.shape[0] == dataset.Y.shape[0]

        rng = check_random_state(self.random_state)
        
        # Select subjects to hold out.
        dataset_idxs = np.arange(len(datasets))
        rng.shuffle(dataset_idxs)
        datasets_holdout = dataset_idxs[:int(self.cfg.series_hold_pct * len(dataset_idxs))]

        ret_train_datasets, ret_test_datasets = [], []
        for i, dataset in enumerate(datasets):
            if i in datasets_holdout:
                # Hold out random portion of time series.
                len_i = dataset.X_ts.shape[0]

                slice_point = int(self.cfg.data_hold_pct * len_i)
                if rng.random() < 0.5:
                    # Slice end to test set.
                    slice_train_i = slice(None, slice_point)
                    slice_test_i = slice(slice_point, None)
                else:
                    # Slice start to test set.
                    slice_train_i = slice(slice_point, None)
                    slice_test_i = slice(None, slice_point)

                ret_train_datasets.append(dataset[slice_train_i])
                ret_test_datasets.append(dataset[slice_test_i])
            else:
                ret_train_datasets.append(dataset)

        return ret_train_datasets, ret_test_datasets