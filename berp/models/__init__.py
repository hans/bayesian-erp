from pathlib import Path
import pickle

from hydra.utils import to_absolute_path
import torch

from berp.models.trf import BerpTRF
from berp.models.trf_em import BerpTRFEMEstimator, GroupTRFForwardPipeline, GroupVanillaTRFForwardPipeline


def load_model(model_dir: str) -> GroupTRFForwardPipeline:
    pipeline_pickle = Path(to_absolute_path(model_dir)) / "params" / "pipeline.pkl"
    with pipeline_pickle.open("rb") as f:
        pipeline = pickle.load(f)

    # HACK: Add dummy feature in coefs, because the full model includes a rec point feature.
    if isinstance(pipeline, GroupVanillaTRFForwardPipeline):
        for encoder in pipeline.encoders_.values():
            insert_point = 21
            encoder.coef_ = torch.cat(
                [encoder.coef_[:insert_point, :, :],
                 torch.zeros(1, *encoder.coef_.shape[1:]),
                 encoder.coef_[insert_point:, :, :]],
                 dim=0)
            encoder.n_features_ += 1

    return pipeline


__all__ = [
    "BerpTRF",
    "BerpTRFEMEstimator",

    "load_model",
]