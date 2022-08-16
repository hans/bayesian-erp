from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import List, Dict, Any

import hydra
from hydra_plugins.hydra_optuna_sweeper._impl import create_optuna_distribution_from_config
import numpy as np
from omegaconf import DictConfig, OmegaConf
import optuna
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split
import torch
from tqdm.auto import tqdm

from berp.config import Config, CVConfig
from berp.datasets import NestedBerpDataset
from berp.models import BerpTRFExpectationMaximization, BerpTRF
from berp.util import PartialPipeline


MODELS = {
    "em-trf": BerpTRFExpectationMaximization,
    "trf": BerpTRF,
}


def score(estimator: PartialPipeline, X, Y):
    # TODO: valid samples
    # TODO Should be implemented in score probably
    _, Y_gt = estimator.pre_transform(X, Y)
    Y_pred = estimator.predict(X)
    mse = ((Y_pred - Y_gt) ** 2).sum(dim=1).mean()
    return -mse


def make_cv(model, cfg: CVConfig):
    """
    Make cross-validation object.
    """
    # TODO if we want to customize sampler, we have to set up a "study"
    param_sampler = hydra.utils.instantiate(cfg.param_sampler)
    param_distributions = {
        k: create_optuna_distribution_from_config(v)
        for k, v in cfg.params.items()
    }
    
    return optuna.integration.OptunaSearchCV(
        estimator=clone(model),
        # param_sampler=param_sampler,
        # n_trials=cfg.n_trials,
        # n_jobs=cfg.n_jobs,
        n_jobs=1,
        enable_pruning=True,
        max_iter=1,
        param_distributions=param_distributions,
        scoring=score,
        cv=KFold(n_splits=cfg.n_inner_folds, shuffle=False),
        refit=True,
        verbose=1,)


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))

    dataset = hydra.utils.call(cfg.dataset)
    dataset.set_n_splits(4)

    model = MODELS[cfg.model.type](cfg.model)

    # TODO use cfg
    # DEV: tiny training set
    data_train, data_test = train_test_split(dataset, test_size=0.25, shuffle=False)
    # Re-merge into nested datasets for further CV fun.
    data_train = NestedBerpDataset(data_train)
    data_test = NestedBerpDataset(data_test)

    # TODO figure out shuffling. Can shuffle at the subject level ofc but not at the
    # time series level.

    # Nested cross-validation. Outer CV loop error on test set;
    # inner CV loop estimates optimal hyperparameters.
    outer_cv = KFold(n_splits=cfg.cv.n_outer_folds, shuffle=False)
    fold_results = []
    for i_split, (train_fold, test_fold) in enumerate(tqdm(outer_cv.split(data_train))):
        inner_cv = make_cv(model, cfg.cv)
        fold_results.append(inner_cv.fit(data_train[train_fold]))

    if cfg.solver.type == "svd":
        model.fit(data_train)
    else:
        model.partial_fit(data_train)


if __name__ == "__main__":
    # p = ArgumentParser()

    # p.add_argument("-m", "--model", default="em-trf", choices=["em-trf", "baseline"])
    # p.add_argument("--solver", default="adam", choices=["adam", "sgd", "svd"])
    # p.add_argument("datasets", nargs="+", type=Path)

    # main(p.parse_args())
    main()