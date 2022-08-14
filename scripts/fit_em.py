from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import List, Dict, Any

import hydra
from hydra_plugins.hydra_optuna_sweeper._impl import create_optuna_distribution_from_config
from omegaconf import DictConfig, OmegaConf
import optuna
from sklearn.base import clone
from sklearn.model_selection import KFold

from berp.config import Config, CVConfig
from berp.datasets import BerpTrainTestSplitter, BerpKFold, NestedBerpDataset
from berp.models import BerpTRFExpectationMaximization, BerpTRF


MODELS = {
    "em-trf": BerpTRFExpectationMaximization,
    "trf": BerpTRF,
}


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
    
    def scoring(*args, **kwargs):
        print("SCORING", args, kwargs)
        return 0.0
    
    return optuna.integration.OptunaSearchCV(
        estimator=clone(model),
        # param_sampler=param_sampler,
        # n_trials=cfg.n_trials,
        # n_jobs=cfg.n_jobs,
        n_jobs=1,
        param_distributions=param_distributions,
        scoring=scoring,
        cv=BerpKFold(n_splits=cfg.n_inner_folds),
        refit=True,
        verbose=1,)


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))

    datasets = []
    for dataset in cfg.datasets:
        with open(dataset, "rb") as f:
            datasets.append(pickle.load(f).ensure_torch())
    dataset = NestedBerpDataset(datasets)

    model = MODELS[cfg.model.type](cfg.model)

    splitter = BerpTrainTestSplitter(cfg.train_test)
    train_idxs, test_idxs = splitter.split(dataset)
    data_train = dataset[train_idxs]

    # Nested cross-validation. Outer CV loop error on test set;
    # inner CV loop estimates optimal hyperparameters.
    outer_cv = BerpKFold(n_splits=cfg.cv.n_outer_folds)
    fold_results = []
    for i_split, (train_fold, test_fold) in enumerate(outer_cv.split(data_train)):
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