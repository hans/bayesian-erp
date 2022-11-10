from pathlib import Path
import pickle

from hydra.utils import to_absolute_path

from berp.models.trf import BerpTRF
from berp.models.trf_em import BerpTRFEMEstimator


def load_model(model_dir: str):
    pipeline_pickle = Path(to_absolute_path(model_dir)) / "params" / "pipeline.pkl"
    with pipeline_pickle.open("rb") as f:
        pipeline = pickle.load(f)
    return pipeline


__all__ = [
    "BerpTRF",
    "BerpTRFEMEstimator",

    "load_model",
]