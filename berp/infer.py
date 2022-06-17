import pyro.infer
import pyro.poutine as poutine
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