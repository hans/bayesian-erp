"""
Defines utilities for data splitting (for model selection and cross validation).
"""

from typing import Iterator, Tuple

import numpy as np
from sklearn import model_selection
from typeguard import typechecked

from berp.datasets import NestedBerpDataset


class KFold(model_selection.KFold):
    """
    Generate K-fold train/test splits from the given nested dataset
    using a time-series cross validation method.
    
    Each draw from the splitter yields interleaved train/test splits,
    with otherwise maximally contiguous splits. This ensures that test
    data is sampled widely within and across time series in the nested
    dataset.
    
    TODO link to viz notebook for demonstration
    """

    @typechecked
    def split(self, dataset: NestedBerpDataset, y=None, groups=None):
        if y is not None:
            raise ValueError("Didn't expect y to be passed")
        if groups is not None:
            raise ValueError("Didn't expect groups to be passed")
        assert self.shuffle == False, "Only supports non-shuffled k-fold"

        # Scikit-learn K-fold draws contiguous dataset indices.
        # Reorder nested dataset by time, so that contiguous draws will
        # draw the same time slice from different datasets.
        dataset.order_by_time()

        return super().split(dataset)


@typechecked
def kfold(dataset: NestedBerpDataset, splitter_cls=KFold,
          **kfold_kwargs) -> Iterator[Tuple[NestedBerpDataset, NestedBerpDataset]]:
    """
    Generate K-fold train/test splits from the given nested dataset
    using a time-series cross validation method.
    
    Each draw from the generator yields interleaved train/test splits,
    with otherwise maximally contiguous splits. This ensures that test
    data is sampled widely within and across time series in the nested
    dataset.
    
    TODO link to viz notebook for demonstration
    """

    kf = splitter_cls(shuffle=False, **kfold_kwargs)
    for train_idxs, test_idxs in kf.split(dataset):
        yield dataset[train_idxs], dataset[test_idxs]


@typechecked
def train_test_split(dataset: NestedBerpDataset, test_size=0.25
                     ) -> Tuple[NestedBerpDataset, NestedBerpDataset]:
    """
    Split the given nested dataset into train/test sets.
    """
    
    # Reorder nested dataset by time, so that contiguous draws will
    # draw the same time slice from different datasets.
    dataset.order_by_time()

    n_test = int(len(dataset) * test_size)
    n_train = len(dataset) - n_test

    train_idxs = np.arange(n_train)
    test_idxs = np.arange(n_train, n_train + n_test)
    return dataset[train_idxs], dataset[test_idxs]