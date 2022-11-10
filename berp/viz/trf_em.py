import logging
import pickle
from typing import Optional

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

    # Table-ize and render TRF coefficients.
    ts_feature_names = dataset.ts_feature_names if dataset.ts_feature_names is not None else \
        [str(x) for x in range(dataset.n_ts_features)]
    variable_feature_names = dataset.variable_feature_names if dataset.variable_feature_names is not None else \
        [f"var_{x}" for x in range(dataset.n_variable_features)]
    feature_names = ts_feature_names + variable_feature_names
    for key, encoder in tqdm(est.encoders_.items(), desc="Visualizing encoders"):
        coefs_df = trf_to_dataframe(encoder, feature_names=feature_names)
        coefs_df["name"] = key
        coefs_df.to_csv(params_dir / f"encoder_coefs.{key}.csv", index=False)

        fig = plot_trf_coefficients(
            encoder,
            feature_names=feature_names,
            feature_match_patterns=viz_cfg.feature_patterns)
        fig.savefig(params_dir / f"encoder_coefs.{key}.png")
        tb.add_figure(f"encoder_coefs/{key}", fig)


def trf_em_tb_callback(est, dataset, params_dir, viz_cfg: VizConfig,
                       baseline_model: Optional[GroupTRFForwardPipeline] = None):
    def tb_callback(study, trial):
        tb = Tensorboard.instance()
        tb.global_step += 1
        if study.best_trial.number == trial.number:
            for param, value in trial.params.items():
                tb.add_scalar(f"optuna/{param}", value)
            tb.add_scalar("optuna/test_score", trial.value)

            # Refit and checkpoint model.
            L.info("Refitting and checkpointing model")
            est_i = clone(est)
            est_i.set_params(**trial.params)
            est_i.fit(dataset)
            checkpoint_model(est_i, dataset, params_dir, viz_cfg,
                             baseline_model=baseline_model)
    
    return tb_callback