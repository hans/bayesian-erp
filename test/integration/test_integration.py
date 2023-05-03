"""
TODO: simple startup test. Using a serialized tiny dataset,
load it, construct two nested models, and do a single fit.
"""

from pathlib import Path
import os
import sys
from typing import cast, Tuple
import yaml

from hydra import initialize, compose
from omegaconf import OmegaConf
import pytest
import torch

from berp.config import Config
from berp.tensorboard import Tensorboard
from berp.trainer import Trainer

from .conftest import IntegrationHarness


# Avoid saving lots of spurious tensorboard events files
@pytest.fixture
def disable_tensorboard():
    Tensorboard.disable()

    
def hydra_param(obj):
    return yaml.safe_dump(obj, default_flow_style=True, width=float("inf")).strip()


@pytest.mark.parametrize("device", [None, "cpu", "cuda"])
def test_integration(device, tmp_path,
                     integration_harness: IntegrationHarness,
                     disable_tensorboard):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")

    dataset_paths = [str(path) for path in integration_harness.dataset_paths.values()]
    stimulus_paths = {run: str(path) for run, path in integration_harness.stimulus_paths.items()}

    with initialize(version_base=None, config_path="../../conf"):
        overrides = [
            f"dataset={integration_harness.dataset_spec}",
            f"dataset.paths={hydra_param(dataset_paths)}",
            "dataset.subset_sensors=null",  # we have our own special sensor setup in the harness
            f"+dataset.stimulus_paths={hydra_param(stimulus_paths)}",
            "model=trf-berp-fixed",
            f"model.confusion_path='{integration_harness.confusion_path}'",
            f"features={integration_harness.features_spec}",
            "cv.n_trials=2",
        ]

        if device is not None:
            overrides.extend([
                f"model.device={device}",
                f"dataset.device={device}",
            ])

        cfg: Config = cast(Config, compose(config_name="config.yaml", overrides=overrides))
        os.chdir(tmp_path)
        print(OmegaConf.to_yaml(cfg))

        trainer = Trainer(cfg)

        trainer.model.pre_transform(trainer.dataset.datasets[0])

        trainer.fit()