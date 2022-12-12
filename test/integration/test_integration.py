"""
TODO: simple startup test. Using a serialized tiny dataset,
load it, construct two nested models, and do a single fit.
"""

from pathlib import Path
import os
import sys
from typing import cast, Tuple

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


@pytest.mark.parametrize("device", [None, "cpu", "cuda"])
def test_integration(device, tmp_path,
                     integration_harness: IntegrationHarness,
                     disable_tensorboard):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")

    with initialize(version_base=None, config_path="../../conf"):
        overrides = [
            f"dataset={integration_harness.dataset_spec}",
            f"dataset.paths=['{integration_harness.dataset_path}']",
            f"+dataset.stimulus_paths={{{integration_harness.stimulus_name}:'{integration_harness.stimulus_path}'}}",
            "model=trf-berp-fixed",
            f"model.confusion_path='{integration_harness.confusion_path}'",
            f"features={integration_harness.features_spec}",
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