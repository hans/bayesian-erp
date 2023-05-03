import logging
from pathlib import Path
import sys
from typing import Callable, List, Optional
import yaml

import hydra
import numpy as np
from omegaconf import OmegaConf
import optuna
from sklearn.base import clone, BaseEstimator
import torch
from tqdm.auto import tqdm

from berp.config import Config
from berp.cv import OptunaSearchCV
from berp.cv.evaluation import BaselinedScorer
from berp.datasets import NestedBerpDataset
from berp.datasets.splitters import KFold, train_test_split
from berp.models import load_model
from berp.models.trf_em import GroupTRFForwardPipeline
from berp.tensorboard import Tensorboard
from berp.viz.trf_em import reestimate_trf_coefficients, checkpoint_model


L = logging.getLogger(__name__)


# Install custom YAML representer to avoid representer errors with
# Python objects in safe_dump
yaml.SafeDumper.yaml_representers[None] = lambda self, data: \
    yaml.representer.SafeRepresenter.represent_str(
        self,
        repr(data),
    )
tensor_representer = lambda self, data: \
    yaml.representer.SafeRepresenter.represent_str(self,
        f"tensor of type {data.dtype}, shape {data.shape}")
yaml.SafeDumper.yaml_representers[torch.Tensor] = tensor_representer
yaml.SafeDumper.yaml_representers[np.ndarray] = tensor_representer


class Trainer:

    def __init__(self, cfg: Config, checkpoint_path=None, dataset=None):
        self.cfg = cfg

        # Set up Tensorboard singleton instance before instantiating data/model classes.
        tb = hydra.utils.call(cfg.viz.tensorboard)

        root_dir = Path(checkpoint_path if checkpoint_path is not None else ".")
        self.params_dir = root_dir / Path("params")

        if checkpoint_path is None:
            # New training run
            self.params_dir.mkdir(exist_ok=False)

        # Allow preloading/sharing datasets across instances
        if dataset is None:
            self.dataset: NestedBerpDataset = hydra.utils.call(self.cfg.dataset)
            self.dataset.set_n_splits(8)
        else:
            self.dataset = dataset
        self.prepare_datasets()

        self.prepare_models(checkpoint_path)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, dataset=None, device=None):
        cfg = OmegaConf.load(Path(checkpoint_path) / ".hydra" / "config.yaml")

        if device is not None:
            cfg.dataset.device = device
            cfg.model.device = device

        return cls(cfg, checkpoint_path, dataset=dataset)

    def prepare_datasets(self):
        # DEV: use a much smaller training set for dev cycle efficiency
        # test_size = 0.75
        # L.warning("Using a teeny training set for dev purposes")
        test_size = .25
        self.data_train, self.data_test = train_test_split(self.dataset, test_size=test_size)

    def prepare_models(self, checkpoint_path):
        if checkpoint_path is not None:
            self.model: GroupTRFForwardPipeline = \
                load_model(checkpoint_path, device=self.cfg.model.device)
        else:
            self.model: GroupTRFForwardPipeline = hydra.utils.call(
                self.cfg.model,
                encoder_key_re=self.cfg.dataset.encoder_key_re,
                features=self.cfg.features,
                optim=self.cfg.solver,
                phonemes=self.dataset.phonemes)

        # Dump model parameters to stdout, but avoid dumping all the torch values.
        yaml.safe_dump(self.model.get_params(), sys.stdout)

        # Before splitting datasets, prime model pipeline with full data.
        self.model.prime(self.dataset)

        self.baseline_model: Optional[GroupTRFForwardPipeline] = None
        if self.cfg.baseline_model_path is not None:
            self.baseline_model = load_model(self.cfg.baseline_model_path)
            self.baseline_model.prime(self.dataset)

    def _make_tb_callback(self):
        """
        Prepare a callback function for Optuna search which sends results
        to Tensorboard.
        """
        # DEV: don't do k-fold estimation for viz.
        viz_splitter = None
        def tb_callback(study, trial):
            tb = Tensorboard.instance()
            if tb._disabled:
                return

            tb.global_step += 1
            if study.best_trial.number == trial.number:
                for param, value in trial.params.items():
                    tb.add_scalar(f"optuna/{param}", value)
                tb.add_scalar("optuna/test_score", trial.value)

                # Refit and checkpoint model.
                L.info("Refitting and checkpointing model")
                est_i = clone(self.model)
                est_i.set_params(**trial.params)
                est_i.fit(self.data_train)
                checkpoint_model(
                    est_i, self.data_train, self.params_dir, self.cfg.viz,
                    baseline_model=self.baseline_model)

                # reestimate_trf_coefficients(
                #     est_i, self.data_train, self.params_dir, viz_splitter, self.cfg.viz)

        return tb_callback

    def _make_cv(self, callbacks: Optional[List[Callable]] = None):
        """
        Make cross-validation object.
        """
        if callbacks is None:
            callbacks = []

        cv_cfg = self.cfg.cv
        param_distributions = {}
        for name, dist_cfg in cv_cfg.params.items():
            param_distributions.update(hydra.utils.call(dist_cfg, name=name))

        sampler = hydra.utils.instantiate(cv_cfg.param_sampler)
        study = optuna.create_study(sampler=sampler, direction="maximize")

        aggregation_fn = getattr(np, cv_cfg.sensor_aggregation_fn)
        scoring = BaselinedScorer(self.baseline_model, aggregation_fn=aggregation_fn)

        n_trials = cv_cfg.n_trials if len(cv_cfg.params) > 0 else 1
        return OptunaSearchCV(
            estimator=clone(self.model),
            study=study,
            enable_pruning=False,
            max_iter=cv_cfg.max_iter,
            n_trials=n_trials,
            param_distributions=param_distributions,
            error_score="raise",
            scoring=scoring,
            cv=KFold(n_splits=cv_cfg.n_inner_folds),
            refit=True,
            verbose=1,
            callbacks=callbacks,)

    def prepare_cv(self):
        if hasattr(self, "cv"):
            return

        # TODO outer CV
        if len(self.cfg.cv.params) == 0:
            raise ValueError("Only CV grid search supported currently.")

        self.cv = self._make_cv(callbacks=[self._make_tb_callback()])

    def fit(self):
        self.prepare_cv()
        self.cv.fit(self.data_train)

        # Save study information for all hparam options.
        self.cv.study.trials_dataframe().to_csv("trials.csv", index=False)

        np.savez(self.params_dir / "hparams.npz", **self.cv.best_params_)
        est = self.cv.best_estimator_

        # Save best model.
        checkpoint_model(est, self.data_train, self.params_dir, self.cfg.viz)

    def score(self, dataset=None):
        if dataset is None:
            dataset = self.data_test
        return self.model.score_multidimensional(dataset)


def load_trainers_from_checkpoints(checkpoint_dirs: List[str],
                                   device=None) -> List[Trainer]:
    """
    Load a collection of `Trainers` for model comparison from the
    given checkpoint paths. These models should be compatible for
    evaluation -- that is, they should have been trained on the
    same training data. This will be checked/asserted during load.

    Returns a list of `Trainer` instances. The dataset/stimulus
    data is loaded only once in the first `Trainer` instance, and
    subsequent `Trainer` objects contain references to the same
    dataset object.
    """

    trainers = []

    for model_dir in tqdm(checkpoint_dirs, unit="model"):
        dataset = None if len(trainers) == 0 else trainers[0].dataset
        if dataset is not None:
            # Peek at the config and make sure stimulus+dataset setup is compatible.
            cfg = OmegaConf.load(Path(model_dir) / ".hydra" / "config.yaml")
            cfg.dataset.device = device

            assert cfg.dataset == trainers[0].cfg.dataset
        
        print(f"\n\n===== {model_dir}")
        trainer = Trainer.from_checkpoint(model_dir, device=device, dataset=dataset)
        trainers.append(trainer)

    return trainers