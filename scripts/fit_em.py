from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import List, Dict, Any

from omegaconf import DictConfig, OmegaConf
import hydra

from berp.config import Config
from berp.datasets import BerpDatasetSplitter
from berp.models import BerpTRFExpectationMaximization, BerpTRF


MODELS = {
    "em-trf": BerpTRFExpectationMaximization,
    "trf": BerpTRF,
}


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))

    datasets = []
    for dataset in cfg.datasets:
        with open(dataset, "rb") as f:
            datasets.append(pickle.load(f))

    model = MODELS[cfg.model.type](cfg.model)

    splitter = BerpDatasetSplitter(cfg.train_test)
    data_train, data_test = splitter.split(datasets)

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