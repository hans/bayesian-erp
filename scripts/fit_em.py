import logging
import hydra
from omegaconf import OmegaConf
import optuna
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from tqdm.auto import tqdm, trange

from berp.config import Config
from berp.trainer import Trainer

L = logging.getLogger(__name__)

# Use root logger for Optuna output.
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()