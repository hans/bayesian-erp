import logging
import pickle
import re
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
import torch
from tqdm import tqdm

from berp.config import VizConfig
from berp.datasets import BerpDataset, NestedBerpDataset, get_metadata
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


def pipeline_to_dataframe(pipe: GroupTRFForwardPipeline):
    ts_predictor_names, var_predictor_names = pipe.encoder_predictor_names
    predictor_names = ts_predictor_names + var_predictor_names

    trf_df = pd.concat([
        trf_to_dataframe(encoder, predictor_names=predictor_names)
        for encoder in pipe.encoders_.values()
    ], names=["subject"], keys=pipe.encoders_.keys()) \
        .droplevel(1)

    try:
        sensor_names = next(iter(pipe.encoders_.values())).output_names
    except AttributeError: pass
    else:
        trf_df["sensor_name"] = trf_df.sensor.map(dict(enumerate(sensor_names)))

    return trf_df


def aggregate_cannon_coef_df(df: pd.DataFrame, pipe: GroupTRFForwardPipeline) -> pd.DataFrame:
    """
    Combine the coefficients of the given cannon model coefficient dataframe so that
    we have a single value per variable feature+bin. Concretely, combine the global coefficient
    with each of the per-bin coefficients for each variable feature.

    The column `quantile` in the resulting dataframe is a zero-based quantile index.
    """
    _, var_predictor_names = pipe.encoder_predictor_names
    combine_features = tuple(set(re.sub("_(\d+)$", "", predictor) for predictor in var_predictor_names))

    coef_df = df.copy()
    cdf = coef_df[coef_df.predictor_name.str.startswith(combine_features)].reset_index()
    cdf["base_predictor"] = cdf.predictor_name.str.replace(r"_(\d+)$", "", regex=True)
    cdf["quantile"] = cdf.predictor_name.str.extract(r"_(\d+)").astype(int)
    cdf = cdf.set_index(["quantile", "base_predictor",
                         "subject", "lag", "epoch_time", "sensor"])
    cdf = cdf.coef + cdf.loc[0].coef
    cdf = cdf.reset_index()
    cdf = cdf[cdf["quantile"] != 0]
    cdf["quantile"] -= 1
    # Reinstate predictor_name
    cdf["predictor_name"] = cdf.base_predictor + "_" + (cdf["quantile"] + 1).astype(str)

    return cdf


def get_cannon_posterior_df(pipe: GroupTRFForwardPipeline, ds: Dict[Any, BerpDataset]) -> pd.DataFrame:
    """
    Get a word-level long data frame describing word features and cannon posteriors for
    the given pipeline.
    """
    if not hasattr(pipe, "_get_recognition_quantiles"):
        raise ValueError("pipe is not a cannon pipeline")

    recognition_points = {}
    recognition_times = {}
    recognition_quantiles = {}

    for key, dataset in tqdm(ds.items(), unit="dataset"):
        points, times = pipe.get_recognition_times(dataset, pipe.params[0])
        recognition_points[key] = points.numpy()
        recognition_times[key] = (times - dataset.word_onsets).numpy()
        recognition_quantiles[key] = pipe._get_recognition_quantiles(dataset, pipe.params[0]).numpy()

    df = pd.concat([pd.DataFrame({"recognition_quantile": recognition_quantiles[key],
                                  "recognition_point": recognition_points[key],
                                  "recognition_time": recognition_times[key]}).rename_axis("word_idx")
                    for key in ds.keys()],
                   keys=recognition_quantiles.keys(), names=["dataset"])

    # Merge in word metadata.
    metadata_df = pd.concat([get_metadata(ds_i) for ds_i in ds.values()],
                            keys=ds.keys(), names=["dataset"]) 
    df = pd.merge(df, metadata_df, left_index=True, right_index=True)

    return df