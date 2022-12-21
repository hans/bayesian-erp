import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
import torch
from tqdm import tqdm

from berp.config import VizConfig
from berp.datasets import BerpDataset, NestedBerpDataset
from berp.models.trf_em import GroupTRFForwardPipeline
from berp.tensorboard import Tensorboard
from berp.viz import trf_to_dataframe, plot_trf_coefficients


L = logging.getLogger(__name__)


def get_recognition_times(est, dataset):
    # HACK
    assert len(est.params) == 1

    datasets = dataset.datasets if isinstance(dataset, NestedBerpDataset) else [dataset]
    rec_points, rec_times = [], []

    for ds in datasets:
        rec_points_i, rec_times_i = est.get_recognition_times(ds, est.params[0])

        # Reindex recognition times relative to word onset.
        rec_times_i -= ds.word_onsets

        rec_points.append(rec_points_i)
        rec_times.append(rec_times_i)

    return torch.concat(rec_points), torch.concat(rec_times)


def checkpoint_model(est, dataset, params_dir, viz_cfg: VizConfig,
                     baseline_model: Optional[GroupTRFForwardPipeline] = None):
    tb = Tensorboard.instance()

    # May have a wrapper around it.
    if hasattr(est, "pipeline"):
        est = est.pipeline
    assert isinstance(est, GroupTRFForwardPipeline)

    # Save the whole pipeline first in pickle format. This includes
    # latent model parameters as well as all encoder weights. It does
    # not include pre-transformed TRF design matrices
    with (params_dir / "pipeline.pkl").open("wb") as f:
        pickle.dump(est, f)

    # Extract and save particular parameters from the ModelParameters grid.
    if hasattr(est, "params"):
        # TODO should just be linked to the ones we know we are searching over
        berp_params_to_extract = ["threshold", "lambda_"]
        berp_params_df = pd.DataFrame(
            [[getattr(param_set, param_name).item() for param_name in berp_params_to_extract]
            for param_set in est.params],
            columns=berp_params_to_extract
        )
        berp_params_df["weight"] = est.param_weights.numpy()
        berp_params_df.to_csv(params_dir / "berp_params.csv", index=False)

    # If relevant, compute recognition points/times.
    if hasattr(est, "get_recognition_times"):
        rec_points, rec_times = get_recognition_times(est, dataset)
        tb.add_histogram("recognition_points", rec_points)
        tb.add_histogram("recognition_times", rec_times)


def reestimate_trf_coefficients(est, dataset, params_dir, splitter, viz_cfg: VizConfig):
    """
    Re-estimate, table-ize, and plot TRF model coefficients over folds
    of the given dataset.
    """

    tb = Tensorboard.instance()

    # May have a wrapper around it.
    if hasattr(est, "pipeline"):
        est = est.pipeline
    assert isinstance(est, GroupTRFForwardPipeline)

    ts_predictor_names, variable_predictor_names = est.encoder_predictor_names
    predictor_names = ts_predictor_names + variable_predictor_names
    coef_dfs = []

    if splitter is None:
        fold_list = [(np.arange(len(dataset.flat_idxs)), None)]
    else:
        fold_list = list(splitter.split(dataset))

    for i, (train, _) in enumerate(tqdm(fold_list, desc="Re-estimating TRF coefficients", unit="fold")):
        est_i = clone(est)
        est_i.fit(dataset[train])

        for key, encoder in est_i.encoders_.items():
            coef_df_i = trf_to_dataframe(encoder, predictor_names=predictor_names)
            coef_df_i["fold"] = i
            coef_df_i["name"] = key
            coef_dfs.append(coef_df_i)

    coef_df = pd.concat(coef_dfs)

    for key, key_coefs in coef_df.groupby("name"):
        key_coefs.to_csv(params_dir / f"encoder_coefs.{key}.csv", index=False)

        fig = plot_trf_coefficients(key_coefs, predictor_names=predictor_names,
                                    predictor_match_patterns=viz_cfg.predictor_patterns)
        fig.savefig(params_dir / f"encoder_coefs.{key}.png")
        tb.add_figure(f"encoder_coefs/{key}", fig)

    return coef_df