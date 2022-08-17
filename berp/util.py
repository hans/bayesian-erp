import logging
from typing import Union, List, Tuple

import numpy as np
import scipy.signal
from sklearn.base import clone
from sklearn.pipeline import Pipeline
import torch
from torchtyping import TensorType
from typeguard import typechecked

L = logging.getLogger(__name__)


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

from sklearn.base import BaseEstimator
from sklearn.utils import _print_elapsed_time
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory, check_is_fitted


def _final_estimator_has(attr):
    """Check that final_estimator has `attr`.

    Used together with `avaliable_if` in `Pipeline`."""

    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        getattr(self._final_estimator, attr)
        return True

    return check


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

    def __init__(self, steps, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)

        self._has_warned_about_partial_fit = False

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
        memory = check_memory(self.memory)
        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for i, (name, step) in enumerate(self.steps):
            if not hasattr(step, "partial_fit"):
                if i == len(self.steps) - 1:
                    raise ValueError(f"Final step {name} is a {step} which does not support `.partial_fit`. Stop.")
                elif not self._has_warned_about_partial_fit:
                    L.warn(f"Step {name} is a {step} which does not support `.partial_fit`. Will use `.fit` instead.")
        
        self._has_warned_about_partial_fit = True

        # TODO merge with _fit?
        for i, (name, step) in enumerate(self.steps):
            if not hasattr(step, "partial_fit"):
                # This step is not a partial fit. That means we could plausibly use the
                # cache. Try it.
                X, y, fitted_transformer = fit_transform_one_cached(
                    clone(step),
                    X, y, None,
                    message_clsname="PartialPipeline",
                    message=self._log_message(i),
                )

                self.steps[i] = (name, fitted_transformer)
            else:
                if hasattr(step, "predict"):
                    step.partial_fit(X, y, classes=classes, **kwargs)
                else:
                    step.partial_fit(X, y)
                    
                if hasattr(step, "transform"):
                    # NB breaking the sklearn API a bit here.
                    # Why can't transformers just work on Y too?
                    ret = step.transform(X, y)
                    if isinstance(ret, tuple):
                        X, y = ret
                    else:
                        X = ret

        return self

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.

            .. versionadded:: 0.20

        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the final estimator.
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt, _ = transform.transform(Xt)
        return self.steps[-1][1].predict(Xt, **predict_params)

    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **fit_params):
        """Transform the data, and apply `fit_predict` with the final estimator.

        Call `fit_transform` of each transformer in the pipeline. The
        transformed data are finally passed to the final estimator that calls
        `fit_predict` method. Only valid if the final estimator implements
        `fit_predict`.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : ndarray
            Result of calling `fit_predict` on the final estimator.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt, y = self._fit(X, y, **fit_params_steps)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][1].fit_predict(Xt, y, **fit_params_last_step)
        return y_pred

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        """Transform the data, and apply `predict_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_proba` method. Only valid if the final estimator implements
        `predict_proba`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        **predict_proba_params : dict of string -> object
            Parameters to the `predict_proba` called at the end of all
            transformations in the pipeline.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_proba` on the final estimator.
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt, _ = transform.transform(Xt)
        return self.steps[-1][1].predict_proba(Xt, **predict_proba_params)

    def pre_transform(self, X, y):
        """
        Transform inputs up to and not including the final estimator.
        """
        Xt, yt = X, y
        for _, name, transform in self._iter(with_final=False):
            Xt, yt = transform.transform(Xt, yt)
        return Xt, yt


class XYTransformerMixin:
    """
    Transformer which acts on both X and Y inputs.
    """

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)


class StandardXYScaler(XYTransformerMixin, BaseEstimator):
    
    def __init__(self, *, standardize_X=True, standardize_Y=True,
                 copy=True, with_mean=True, with_std=True):
        self.standardize_X = standardize_X
        self.standardize_Y = standardize_Y

        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "mean_X_"):
            del self.mean_X_
            del self.std_X_
        if hasattr(self, "mean_Y_"):
            del self.mean_Y_
            del self.std_Y_

    def fit(self, X, y=None, sample_weight=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.24
               parameter *sample_weight* support to StandardScaler.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        
        if self.standardize_X:
            self.mean_X_ = X.mean(axis=0)
            self.std_X_ = X.std(axis=0)
        if self.standardize_Y and y is not None:
            self.mean_Y_ = y.mean(axis=0)
            self.std_Y_ = y.std(axis=0)

        return self

    # HACK to make this work with partialpipeline
    def partial_fit(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def transform(self, X, y=None):
        if self.standardize_X:
            if self.with_mean:
                X = X - self.mean_X_
            if self.with_std:
                X = X / self.std_X_
        if self.standardize_Y and y is not None:
            if self.with_mean:
                y = y - self.mean_Y_
            if self.with_std:
                y = y / self.std_Y_

        return X, y