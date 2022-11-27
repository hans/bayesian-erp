"""
TODO: simple startup test. Using a serialized tiny dataset,
load it, construct two nested models, and do a single fit.
"""

from pathlib import Path
import os
import sys
from typing import cast

from hydra import initialize, compose
from omegaconf import OmegaConf
import pytest
import torch

from berp.config import Config
from berp.tensorboard import Tensorboard
from berp.trainer import Trainer


# Avoid saving lots of spurious tensorboard events files
@pytest.fixture
def disable_tensorboard():
    Tensorboard.disable()


@pytest.mark.parametrize("device", [None, "cpu", "cuda"])
def test_integration(device, tmp_path, disable_tensorboard):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")

    with initialize(version_base=None, config_path="../../conf"):
        harness_path = str(Path(__file__).parent)
        overrides = [
            f"dataset.paths=['{harness_path}/DKZ_1.microaverage.pkl']",
            f"+dataset.stimulus_paths={{DKZ_1:'{harness_path}/DKZ_1.pkl'}}",
            "model=trf-berp-fixed",
            f"model.confusion_path='{harness_path}/confusion_matrix.npz'",
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