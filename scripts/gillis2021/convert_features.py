"""
Convert Eelbrain feature representations from Gillis data to
a raw ndarray representation.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, List

import eelbrain
from eelbrain._data_obj import asarray
import numpy as np


def to_ndarray(feature: eelbrain.NDVar, include_times=True) -> np.ndarray:
    """
    Convert stimulus ndvar to n_samples * n_features + 1 ndarray,
    where first column is time axis
    """
    data = feature.get_data().T
    if data.ndim == 1:
        data = data[:, np.newaxis]
    
    if include_times:
        return np.concatenate([asarray(feature.time)[:, np.newaxis], data], axis=1)
    else:
        return data

    
def load_features(stim_dir, story) -> Tuple[List[str], np.ndarray]:
    """
    Returns:
        feature_names: names for the columns of the data array
        data: stimulus data, n_samples * n_features
    """
    
    feature_names, feature_data = [], []
    story_dir = stim_dir / story
    if not story_dir.exists():
        raise ValueError()
    
    for feature_file in story_dir.glob("*.pickle"):
        feature_names.append(feature_file.stem)
        feature_data.append(eelbrain.load.unpickle(feature_file))
        
    min_data_length = min(stim.time.tstop for stim in feature_data)
    feature_data = [feature.sub(time=(None, min_data_length)) for feature in feature_data]
    
    # Convert features to numpy representation
    feature_data_np = [to_ndarray(feature) for feature in feature_data]
    # Sanity check: time axes should all match
    for feature in feature_data_np[1:]:
        np.testing.assert_allclose(feature_data_np[0][:, 0], feature[:, 0])
        
    # Expand feature names. Some feature files have multiple dimensions (e.g.
    # power of multiple frequency bands).
    feature_names_flat = [f"{feature.name}_{i}"
                          for feature, feature_np in zip(feature_data, feature_data_np)
                          for i in range(feature_np.shape[1] - 1)]
    feature_data_flat = np.concatenate([feature[:, 1:] for feature in feature_data_np],
                                       axis=1)
    assert len(feature_names_flat) == feature_data_flat.shape[1]
    
    return feature_names_flat, feature_data_flat


def main(args):
    stories = [x.name for x in args.stim_dir.glob("*") if x.is_dir()]
    
    story_features = {}
    check_feature_names = None
    for story in stories:
        feature_names, story_features[story] = load_features(args.stim_dir, story)
        
        # Check feature consistency
        if check_feature_names is not None:
            assert check_feature_names == feature_names
        check_feature_names = feature_names

    from pprint import pprint
    print("Feature names:")
    pprint(feature_names)

    # By default, drop frequency and surprisal, because these are already accounted
    # for in the variable-onset data.
    drop_features = ["ngram word frequency_0", "ngram surprisal_0"]
    if args.drop_features is not None:
        drop_features += args.drop_features.strip().split(",")

    print("Dropping features: ", ",".join(map(repr, drop_features)))

    feature_mask = np.ones(len(feature_names), dtype=bool)
    for feature in drop_features:
        feature_mask[check_feature_names.index(feature)] = False
    final_feature_names = [name for i, name in enumerate(check_feature_names)
                           if feature_mask[i]]

    for story, features_mat in story_features.items():
        story_features[story] = features_mat[:, feature_mask]

    print("Feature names after dropping:")
    pprint(final_feature_names)
        
    np.savez(args.out_path, feature_names=final_feature_names,
             **story_features)
    
    
if __name__ == "__main__":
    p = ArgumentParser()

    p.add_argument("stim_dir", type=Path)
    p.add_argument("out_path", type=Path)
    p.add_argument("--drop_features", type=str)

    main(p.parse_args())