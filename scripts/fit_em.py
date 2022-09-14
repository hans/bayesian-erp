import json
import logging
from pathlib import Path
import pickle

import hydra
from omegaconf import OmegaConf
import optuna
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm, trange

from berp.config import Config, CVConfig
from berp.cv import OptunaSearchCV, EarlyStopException
from berp.models.trf_em import GroupTRFForwardPipeline, BerpTRFEMEstimator
from berp.tensorboard import Tensorboard
from berp.viz.trf import trf_to_dataframe, plot_trf_coefficients


L = logging.getLogger(__name__)


def make_cv(model, cfg: CVConfig):
    """
    Make cross-validation object.
    """
    param_distributions = {}
    for name, dist_cfg in cfg.params.items():
        param_distributions.update(hydra.utils.call(dist_cfg, name=name))
    
    sampler = hydra.utils.instantiate(cfg.param_sampler)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    
    n_trials = cfg.n_trials if len(cfg.params) > 0 else 1
    return OptunaSearchCV(
        estimator=clone(model),
        study=study,
        enable_pruning=True,
        max_iter=cfg.max_iter, n_trials=n_trials,
        param_distributions=param_distributions,
        error_score="raise",
        cv=KFold(n_splits=cfg.n_inner_folds, shuffle=False),
        refit=True,
        verbose=1,)


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))

    # Set up Tensorboard singleton instance before instantiating data/model classes.
    tb = hydra.utils.call(cfg.viz.tensorboard)

    dataset = hydra.utils.call(cfg.dataset)
    dataset.set_n_splits(4)

    model = hydra.utils.call(cfg.model,
                             phonemes=dataset.phonemes,
                             n_outputs=dataset.n_sensors,
                             optim=cfg.solver)
    from pprint import pprint; pprint(model.get_params())

    # Before splitting datasets, prime model pipeline with full data.
    model.prime(dataset)

    data_train, data_test = train_test_split(dataset, test_size=0.25, shuffle=False)

    params_dir = Path("params")
    params_dir.mkdir()

    # TODO outer CV

    if len(cfg.cv.params) > 0:
        L.info(f"Running cross-validation to estimate hyperparameters, with "
               f"{cfg.cv.n_inner_folds} inner folds.")

        cv = make_cv(model, cfg.cv)
        cv.fit(data_train)

        # Save study information for all hparam options.
        cv.study.trials_dataframe().to_csv("trials.csv", index=False)

        np.savez(params_dir / "hparams.npz", **cv.best_params_)
        est = cv.best_estimator_
    else:
        assert isinstance(model, BerpTRFEMEstimator), 'only supported for EM'

        # TODO cv. For now, just fit on the whole train set.
        L.warning("outer CV folds not yet implemented. Training on whole dataset")

        # HACK collapse learning rates
        # lr_cut_factor = 10  # TODO configurable
        # L.info("Cutting encoder learning rates by factor of %d", lr_cut_factor)
        # for enc in model.pipeline.encoders_.values():
        #     enc.optim.set_params(learning_rate=enc.optim.learning_rate / lr_cut_factor)

        # TODO is this a healthy val set?
        data_train, data_val = train_test_split(
            data_train, test_size=0.1, shuffle=False)
        try:
            model.fit(data_train, X_val=data_val, use_tqdm=True)
        except EarlyStopException: pass

        est = model

    # else:
    #     for _ in trange(cfg.cv.max_iter):
    #         try:
    #             model.partial_fit(data_train)
    #         except EarlyStopException:
    #             break
    #     est = model

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
            feature_match_patterns=cfg.viz.feature_patterns)
        fig.savefig(params_dir / f"encoder_coefs.{key}.png")

    # TODO calculate score on test set. or do full CV.

    # # TODO do we need to clear cache?
    # for dataset in tqdm(dataset.datasets, desc="Datasets"):
    #     data_train = NestedBerpDataset([dataset], n_splits=4)

    #     cv = make_cv(model, cfg.cv)
    #     cv.fit(data_train)

    #     trf = cv.best_estimator_.named_steps["trf"]
    #     coefs_df = trf_to_dataframe(trf)
    #     coefs_df["dataset"] = dataset.name
    #     coefs_df.to_csv(f"coefs.{dataset.name.replace('/', '-')}.csv", index=False)

    #########

    # # TODO figure out shuffling. Can shuffle at the subject level ofc but not at the
    # # time series level.

    # # Nested cross-validation. Outer CV loop error on test set;
    # # inner CV loop estimates optimal hyperparameters.
    # outer_cv = KFold(n_splits=cfg.cv.n_outer_folds, shuffle=False)
    # fold_results = []
    # for i_split, (train_fold, test_fold) in enumerate(tqdm(outer_cv.split(data_train))):
    #     inner_cv = make_cv(model, cfg.cv)
    #     fold_results.append(inner_cv.fit(data_train[train_fold]))

    # if cfg.solver.type == "svd":
    #     model.fit(data_train)
    # else:
    #     model.partial_fit(data_train)


if __name__ == "__main__":
    main()