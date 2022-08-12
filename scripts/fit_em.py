from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import List, Dict, Any

from omegaconf import DictConfig, OmegaConf
import hydra

from berp.models import BerpTRFExpectationMaximization, BerpTRF


MODELS = {
    "em-trf": BerpTRFExpectationMaximization,
    "trf": BerpTRF,
}

@dataclass
class FitConfig:
    model_class: str
    model_kwargs: Dict[str, Any]
    solver: str
    datasets: List[str]


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: FitConfig):
    datasets = []
    for dataset in cfg.datasets:
        with open(dataset, "rb") as f:
            datasets.append(pickle.load(f))

    model = MODELS[cfg.model_class](**cfg.model_kwargs)

    # TODO cross-validation
    dataset = datasets[0]

    if cfg.solver == "svd":
        model.fit(dataset)
    else:
        model.partial_fit(dataset)


if __name__ == "__main__":
    # p = ArgumentParser()

    # p.add_argument("-m", "--model", default="em-trf", choices=["em-trf", "baseline"])
    # p.add_argument("--solver", default="adam", choices=["adam", "sgd", "svd"])
    # p.add_argument("datasets", nargs="+", type=Path)

    # main(p.parse_args())
    main()