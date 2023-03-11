import io
from pathlib import Path
import pickle

from hydra.utils import to_absolute_path
import torch

from berp.models.trf import BerpTRF
from berp.models.trf_em import BerpTRFEMEstimator, GroupTRFForwardPipeline, GroupVanillaTRFForwardPipeline


class MappingUnpickler(pickle.Unpickler):
    def __init__(self, file, map_location='cpu'):
        self.map_location = map_location
        super().__init__(file)

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.map_location)
        else:
            return super().find_class(module, name)


def load_model(model_dir: str, device=None) -> GroupTRFForwardPipeline:
    pipeline_pickle = Path(to_absolute_path(model_dir)) / "params" / "pipeline.pkl"
    with pipeline_pickle.open("rb") as f:
        if device is None:
            pipeline = pickle.load(f)
        else:
            pipeline = MappingUnpickler(f, map_location=device).load()

    return pipeline


__all__ = [
    "BerpTRF",
    "BerpTRFEMEstimator",

    "load_model",
]