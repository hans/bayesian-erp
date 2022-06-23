from typing import Tuple, List, Any

import numpy as np
import pyro.infer
import pyro.poutine as poutine
import torch
from tqdm import tqdm, trange



class Importance(pyro.infer.Importance):
    """
    Simple variant on pyro.infer.Importance that uses tqdm progress bar.
    """

    def _traces(self, *args, **kwargs):
        """
        Generator of weighted samples from the proposal distribution.
        """
        for i in trange(self.num_samples):
            guide_trace = poutine.trace(self.guide).get_trace(*args, **kwargs)  # type: ignore
            model_trace = poutine.trace(
                poutine.replay(self.model, trace=guide_trace)
            ).get_trace(*args, **kwargs)  # type: ignore
            log_weight = model_trace.log_prob_sum() - guide_trace.log_prob_sum()
            yield (model_trace, log_weight)


def evaluate_sliced_tp(tp: pyro.infer.TracePosterior, sample_points: List[int]
                       ) -> List[Tuple[int, torch.Tensor]]:
    if tp.num_chains > 1:
        raise NotImplementedError()

    assert sorted(sample_points) == sample_points
    ret = []
    for sample_point in sample_points[::-1]:
        # Slice TP to the given sample point.
        tp.log_weights = tp.log_weights[:sample_point]
        tp.exec_traces = tp.exec_traces[:sample_point]
        tp.chain_ids = tp.chain_ids[:sample_point]

        ret.append((sample_point, pyro.infer.EmpiricalMarginal(tp).mean))

    return ret


def fit_importance(model, guide, num_samples, *args, **kwargs
                   ) -> Tuple[Importance, List[Tuple[int, torch.Tensor]]]:
    """
    Run importance sampler and return result along with windowed analysis
    of mean estimate with progressively more samples.
    """
    importance = Importance(model, guide, num_samples=num_samples)

    importance.run(*args, **kwargs)

    sample_points = np.linspace(1, importance.num_samples, 10).round().astype(int)
    slice_means = evaluate_sliced_tp(importance, sample_points.tolist())

    return importance, slice_means