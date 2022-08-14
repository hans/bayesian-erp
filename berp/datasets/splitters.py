"""
Defines data splitters for the Berp dataset.

Splitting data needs to be sensitive to the fact that predictions /
learning signals at the edge of resulting splits are not going to be
usable.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterator

import numpy as np
from sklearn.utils import check_random_state
from torchtyping import TensorType
from typeguard import typechecked

from berp.config import TrainTestConfig
from berp.datasets.base import BerpDataset, NestedBerpDataset


# Type alias
intQ = Optional[int]
SplitterRet = Tuple[List[Tuple[int, intQ, intQ]],
                    List[Tuple[int, intQ, intQ]]]
"""Train/test indices returned from splitters"""


class BerpTrainTestSplitter(object):

    def __init__(self, cfg: TrainTestConfig, random_state=None):
        self.cfg = cfg
        self.random_state = random_state

    @typechecked
    def split(self, datasets: NestedBerpDataset) -> SplitterRet:
        """
        Returns:
            (train_idxs, test_idxs)
        """
        # NB we return ranges as int start/end because slices are not hashable
        rng = check_random_state(self.random_state)
        
        # Select subjects to hold out.
        dataset_idxs = np.arange(datasets.n_datasets)
        rng.shuffle(dataset_idxs)
        datasets_holdout = dataset_idxs[:int(self.cfg.series_hold_pct * len(dataset_idxs))]

        train_slices, test_slices = [], []
        for i, dataset in enumerate(datasets.iter_datasets()):
            if i in datasets_holdout:
                # Hold out random portion of time series.
                len_i = dataset.X_ts.shape[0]

                slice_point = int(self.cfg.data_hold_pct * len_i)
                if rng.random() < 0.5:
                    # Slice end to test set.
                    slice_train_i = (None, slice_point)
                    slice_test_i = (slice_point, None)
                else:
                    # Slice start to test set.
                    slice_train_i = (slice_point, None)
                    slice_test_i = (None, slice_point)

                train_slices.append((i, *slice_train_i))
                test_slices.append((i, *slice_test_i))
            else:
                train_slices.append((i, None, None))

        return train_slices, test_slices


class BerpKFold(object):
    """
    K-fold cross-validation over grouped time series datasets.
    """

    def __init__(self, n_splits: int):
        self.n_splits = n_splits

    @typechecked
    def split(self, datasets: NestedBerpDataset, *args
              ) -> Iterator[SplitterRet]:
        """
        Returns:
            (train, test)
        """

        flat_idxs: List[Tuple[int, int]] = datasets.flat_idxs

        def flat_idxs_to_ranges(flat_idxs: np.ndarray) -> List[Tuple[int, int, int]]:
            """
            Convert a list of flat idxs back into a list of slice instructions.
            Assumes input is sorted by flat idx.
            """
            # Split sorted idxs (i, j) at points where i changes.
            idxs_grouped = np.split(flat_idxs, np.where(np.diff(flat_idxs[:, 0]) != 0)[0] + 1)

            ret = []
            for idxs_i in idxs_grouped:
                dataset_idx = int(idxs_i[0, 0])
                ret.append((dataset_idx, int(idxs_i[0, 1]), int(idxs_i[-1, 1] + 1)))

            return ret

        # TODO consider non contiguous folds, this may yield high variance estimates otherwise.
        test_slice_size = int(flat_idxs.shape[0] / self.n_splits)
        for test_slice_start in range(0, flat_idxs.shape[0], test_slice_size):
            test_slice_end = test_slice_start + test_slice_size
            test_idxs = flat_idxs[test_slice_start:test_slice_end]
            train_idxs = np.concatenate([flat_idxs[:test_slice_start], flat_idxs[test_slice_end:]])

            train_slices = flat_idxs_to_ranges(train_idxs)
            test_slices = flat_idxs_to_ranges(test_idxs)

            yield train_slices, test_slices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits