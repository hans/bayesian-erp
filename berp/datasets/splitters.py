"""
Defines utilities for data splitting (for model selection and cross validation).
"""

from typing import Iterator, Tuple

from sklearn.model_selection import KFold
from typeguard import typechecked

from berp.datasets import NestedBerpDataset


@typechecked
def split_kfold(dataset: NestedBerpDataset, splitter_cls=KFold,
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

    # Scikit-learn K-fold draws contiguous dataset indices.
    # Reorder nested dataset by time, so that contiguous draws will
    # draw the same time slice from different datasets.
    dataset.order_by_time()

    kf = splitter_cls(shuffle=False, **kfold_kwargs)
    for train_idxs, test_idxs in kf.split(dataset):
        yield dataset[train_idxs], dataset[test_idxs]