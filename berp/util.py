from typing import Union, List, Tuple

import numpy as np
import scipy.signal
from sklearn.pipeline import Pipeline
import torch
from torchtyping import TensorType
from typeguard import typechecked


def sample_to_time(sample_idx: torch.LongTensor,
                   sample_rate: int,
                   t_zero: float = 0
                   ) -> TensorType[float]:
    """
    Convert sample index representation to time representation (in seconds).
    """
    return t_zero + sample_idx / sample_rate


def time_to_sample(time: TensorType[float],
                   sample_rate: int,
                   t_zero: float = 0
                   ) -> TensorType[torch.long]:
    """
    Convert time representation (in seconds) to sample index representation.
    """
    # TODO meaningful difference between floor/ceil here?
    # Probably just consistency that matters.
    return torch.floor((time - t_zero) * sample_rate).long()


@typechecked
def variable_position_slice(
    x: torch.Tensor, idxs: torch.LongTensor,
    slice_width: int, padding_value=0.
    ) -> Tuple[torch.Tensor, TensorType[bool]]:
    """
    Extract fixed-width column slices from `x` with variable position by row,
    specified by `idxs`. Slices which are too close to the right edge of `x`
    to have `slice_width` items will be padded with `padding_value` and
    marked in the returned `mask`.

    Args:
        x: T * ...
        idxs: B

    Returns:
        sliced: B * slice_width
        mask: B * slice_width, cell ij is True iff corresponding cell ij of
            `sliced` is a valid member of `x` (and not extending past the
            right edge of `x`)
    """

    # Generate index range for each row.
    # TODO is there a better way to do this with real slice objects?
    slice_idxs = torch.arange(slice_width).view(1, -1).tile((idxs.shape[0], 1)) \
        + idxs.unsqueeze(1)
    mask = slice_idxs < x.shape[0]
    # For invalid cells, just retrieve the first item.
    slice_idxs[~mask] = 0

    if x.ndim > 2:
        # Tile slice indices across remaining dimensions.
        viewer = (...,) + (None,) * (x.ndim - 2)
        slice_idxs = slice_idxs[viewer].tile((1, 1) + x.shape[2:])

    sliced = torch.gather(x, 1, slice_idxs)

    return sliced, mask


def gaussian_window(center: float, width: float,\
                    start: float = 0,
                    end: float = 1,
                    sample_rate=128):
    """Gaussian window :class:`NDVar`
    Parameters
    ----------
    center : scalar
        Center of the window (normalized to the closest sample on ``time``).
    width : scalar
        Standard deviation of the window.
    time : UTS
        Time dimension.
    Returns
    -------
    gaussian : NDVar
        Gaussian window on ``time``.
    """

    n_samples = int((end - start) * sample_rate) + 1
    times, step_size = np.linspace(start, end, n_samples, retstep=True, endpoint=True)
    width_i = int(round(width / step_size))
    n_times = len(times)
    center_i = (center - start) // step_size

    slice_start, slice_stop = None, None
    if center_i >= n_times / 2:
        slice_start = None
        slice_stop = n_times
        window_width = 2 * center_i + 1
    else:
        slice_start = -n_times
        slice_stop = None
        window_width = 2 * (n_times - center_i) - 1
    window_data = scipy.signal.windows.gaussian(window_width, width_i)
    window_data = window_data[slice_start: slice_stop]
    return times, window_data


# # TODO untested
# @typechecked
# def pad_and_mask(batch: List[TT],
#                  lengths: Union[List[int], TT[int]],
#                  padding_value=0.
#                  ) -> Tuple[TT[DIMS.B, "times", ...],
#                             TT[DIMS.B, "times", bool]]:
#     padded_batch = torch.nn.utils.rnn.pad_sequence(
#         batch, batch_first=True, padding_value=padding_value)
#
#     mask = torch.arange(padded_batch.shape[1]).view(1, -1) \
#         .tile((padded_batch.shape[0], 1))
#     mask = mask >= torch.tensor(lengths)
#     ic(mask)
#
#     return padded_batch, mask

from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory


def _fit_transform_one(
    transformer, X, y, weight, message_clsname="", message=None, **fit_params
):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, "fit_transform"):
            res, y = transformer.fit_transform(X, y, **fit_params)
        else:
            res, y = transformer.fit(X, y, **fit_params).transform(X, y)

    if weight is None:
        return res, y, transformer
    return res * weight, y, transformer


class PartialPipeline(Pipeline):
    """
    Modified Pipeline implementation which supports

    1. partial fits
    2. transformers which apply to both X and Y simultaneously


    Utility function to generate a `PartialPipeline`
    Arguments:
        steps: a collection of text-transformers
    ```python
    from tokenwiser.pipeline import PartialPipeline
    from tokenwiser.textprep import HyphenTextPrep, Cleaner
    tc = PartialPipeline([('clean', Cleaner()), ('hyp', HyphenTextPrep())])
    data = ["dinosaurhead", "another$$ sentence$$"]
    results = tc.partial_fit(data).transform(data)
    expected = ['di no saur head', 'an other  sen tence']
    assert results == expected
    ```
    """

    # Estimator interface

    def _fit(self, X, y=None, **fit_params_steps):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer
            X, y, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X, y

    def fit(self, X, y=None, **fit_params):
        """Fit the model.

        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, y = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator.

        Fits all the transformers one after the other and transform the
        data. Then uses `fit_transform` on transformed data with the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, y = self._fit(X, y, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt, y
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]
            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(Xt, y, **fit_params_last_step)
            else:
                return last_step.fit(Xt, y, **fit_params_last_step).transform(Xt)

    def partial_fit(self, X, y=None, classes=None, **kwargs):
        """
        Fits the components, but allow for batches.
        """
        for name, step in self.steps:
            if not hasattr(step, "partial_fit"):
                raise ValueError(
                    f"Step {name} is a {step} which does not have `.partial_fit` implemented."
                )
        for name, step in self.steps:
            if hasattr(step, "predict"):
                step.partial_fit(X, y, classes=classes, **kwargs)
            else:
                step.partial_fit(X, y)
                
            if hasattr(step, "transform"):
                # NB breaking the sklearn API a bit here.
                # Why can't transformers just work on Y too?
                ret = step.transform(X)
                if isinstance(ret, tuple):
                    X, y = ret
                else:
                    X = ret
        return self