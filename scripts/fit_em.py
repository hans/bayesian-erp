import json
import logging
from pathlib import Path
import pickle
from typing import Callable, List, Optional

import hydra
from omegaconf import OmegaConf
import optuna
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from tqdm.auto import tqdm, trange

from berp.config import Config, CVConfig
from berp.cv import OptunaSearchCV, EarlyStopException
from berp.cv.evaluation import BaselinedScorer
from berp.datasets.splitters import KFold, train_test_split
from berp.models import load_model
from berp.models.trf_em import GroupTRFForwardPipeline, BerpTRFEMEstimator
from berp.viz.trf_em import trf_em_tb_callback, checkpoint_model


L = logging.getLogger(__name__)

# Use root logger for Optuna output.
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()


def make_cv(model, cfg: CVConfig,
            callbacks: Optional[List[Callable]] = None,
            baseline_model: Optional[GroupTRFForwardPipeline] = None):
    """
    Make cross-validation object.
    """
    if callbacks is None:
        callbacks = []

    param_distributions = {}
    for name, dist_cfg in cfg.params.items():
        param_distributions.update(hydra.utils.call(dist_cfg, name=name))
    
    sampler = hydra.utils.instantiate(cfg.param_sampler)
    study = optuna.create_study(sampler=sampler, direction="maximize")

    aggregation_fn = getattr(np, cfg.sensor_aggregation_fn)
    scoring = BaselinedScorer(baseline_model, aggregation_fn=aggregation_fn)
    
    n_trials = cfg.n_trials if len(cfg.params) > 0 else 1
    return OptunaSearchCV(
        estimator=clone(model),
        study=study,
        enable_pruning=False,
        max_iter=cfg.max_iter, n_trials=n_trials,
        param_distributions=param_distributions,
        error_score="raise",
        scoring=scoring,
        cv=KFold(n_splits=cfg.n_inner_folds),
        refit=True,
        verbose=1,
        callbacks=callbacks,)


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))

    # Set up Tensorboard singleton instance before instantiating data/model classes.
    tb = hydra.utils.call(cfg.viz.tensorboard)

    dataset = hydra.utils.call(cfg.dataset)
    dataset.set_n_splits(8)

    model = hydra.utils.call(cfg.model,
                             phonemes=dataset.phonemes,
                             n_outputs=dataset.n_sensors,
                             optim=cfg.solver)
    from pprint import pprint; pprint(model.get_params())

    # Before splitting datasets, prime model pipeline with full data.
    model.prime(dataset)

    baseline_model: Optional[GroupTRFForwardPipeline] = None
    if cfg.baseline_model_path is not None:
        baseline_model = load_model(cfg.baseline_model_path)
        baseline_model.prime(dataset)

    # DEV: use a much smaller training set for dev cycle efficiency
    # test_size = 0.75
    # L.warning("Using a teeny training set for dev purposes")
    test_size = .25
    data_train, data_test = train_test_split(dataset, test_size=test_size)

    params_dir = Path("params")
    params_dir.mkdir()

    # TODO outer CV

    if len(cfg.cv.params) > 0:
        L.info(f"Running cross-validation to estimate hyperparameters, with "
               f"{cfg.cv.n_inner_folds} inner folds.")

        viz_splitter = KFold(n_splits=cfg.cv.n_inner_folds)
        tb_callback = trf_em_tb_callback(
            model, data_train, params_dir, viz_splitter, cfg.viz,
            baseline_model=baseline_model)
        cv = make_cv(
            model, cfg.cv,
            baseline_model=baseline_model,
            callbacks=[tb_callback])
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
        lr_cut_factor = 100  # TODO configurable
        alpha_reset = None
        L.info("Cutting encoder learning rates by factor of %d", lr_cut_factor)
        if alpha_reset is not None:
            L.info("Setting alpha to %d", alpha_reset)
        for enc in model.pipeline.encoders_.values():
            enc.optim.set_params(learning_rate=enc.optim.learning_rate / lr_cut_factor)
            if alpha_reset is not None:
                enc.set_params(alpha=alpha_reset)

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

    checkpoint_model(est, dataset, params_dir, cfg.viz)

    tb.close()

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