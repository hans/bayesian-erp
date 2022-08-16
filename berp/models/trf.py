from functools import cached_property
import logging
from typing import Tuple, List, Optional, Any, Union

import hydra
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.distributions as dist
from torchtyping import TensorType
from typeguard import typechecked
from tqdm.auto import tqdm, trange

from berp.config.model import TRFModelConfig
from berp.config.solver import SolverConfig
from berp.datasets.base import BerpDataset, NestedBerpDataset
from berp.util import time_to_sample, PartialPipeline

L = logging.getLogger(__name__)


TRFPredictors = TensorType["n_times", "n_features"]
TRFDesignMatrix = TensorType["n_times", "n_features", "n_delays"]
TRFResponse = TensorType["n_times", "n_outputs"]


# TODO clean up and move
class AdamSolver(BaseEstimator):
    """
    Model mixin which supports advanced stochastic gradient descent
    methods.
    """

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 1,
                 batch_size: int = 512, **kwargs):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self._optim_parameters = None

    @cached_property
    def _optim(self):
        return torch.optim.Adam(self._optim_parameters, lr=self.learning_rate)

    def __call__(self, estimator, X, y=None, **fit_params):
        if self._optim_parameters is None:
            self._optim_parameters = estimator._optim_parameters

        for i in trange(self.n_epochs, leave=False):
            losses = []
            for batch_offset in torch.arange(0, X.shape[0], self.batch_size):
                batch_X = X[batch_offset:batch_offset + self.batch_size]
                batch_y = y[batch_offset:batch_offset + self.batch_size] if y is not None else None

                self._optim.zero_grad()
                loss = estimator._loss_fn(batch_X, batch_y)
                loss.backward()
                self._optim.step()

                losses.append(loss.item())

        return self


class TemporalReceptiveField(BaseEstimator):

    def __init__(self, tmin, tmax, sfreq,
                 optim, fit_intercept=False,
                 warm_start=True, alpha=1, **kwargs):
        self.sfreq = sfreq

        self.tmin = tmin
        self.tmax = tmax
        assert self.tmin < self.tmax

        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.alpha = alpha

        # Prepare optimizer mixin
        self.optim = optim

        self.delays_ = _times_to_delays(self.tmin, self.tmax, self.sfreq)

        if kwargs:
            L.warning(f"Unused arguments: {kwargs}")

    def _init_coef(self):
        self.coef_ = torch.randn(self.n_features_, len(self.delays_),
                                 self.n_outputs_) * 1e-1

    # Provide parameters for SGDEstimatorMixin
    @property
    def _optim_parameters(self):
        return [self.coef_]

    def _check_shapes_types(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        assert X.dtype == Y.dtype
        # May not be available if we haven't been called with fit() yet.
        if hasattr(self, "n_features_"):
            assert X.shape[1] == self.n_features_
            assert X.shape[2] == self.n_delays_
            assert Y.shape[1] == self.n_outputs_
        else:
            _, self.n_features_, self.n_delays_ = X.shape
            self.n_outputs_ = Y.shape[1]
        return (torch.as_tensor(X, dtype=torch.float32), 
                torch.as_tensor(Y, dtype=torch.float32))

    @typechecked
    def fit(self, X: TRFDesignMatrix, Y: TRFResponse
            ) -> "TemporalReceptiveField":
        """
        Fit the TRF encoder with least squares.
        """
        
        X, Y = self._check_shapes_types(X, Y)
        X_est = _reshape_for_est(X)

        # Find ridge regression solution.
        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y 
        lhs = X_est.T @ X_est
        rhs = X_est.T @ Y
        if self.alpha is None:
            self.coef_ = torch.linalg.lstsq(lhs, rhs).solution
        else:
            ridge = self.alpha * torch.eye(lhs.shape[0])
            self.coef_ = torch.linalg.lstsq(lhs + ridge, rhs).solution

        # Reshape resulting coefficients
        self.coef_ = self.coef_.reshape((self.n_features_, self.n_delays_, self.n_outputs_))

        Y_pred = self.predict(X)
        self.residuals_ = Y_pred - Y
        return self

    def _loss_fn(self, X, Y: TRFResponse) -> torch.Tensor:
        Y_pred = X @ self.coef_
        loss = (Y_pred - Y).pow(2).sum(axis=1).mean()
        # Add ridge term.
        loss += self.alpha * torch.norm(self.coef_, p=2)
        return loss

    @typechecked
    def partial_fit(self, X: TRFDesignMatrix, Y: TRFResponse,
                    **kwargs) -> "TemporalReceptiveField":
        """
        Update the TRF encoder weights with gradient descent.
        """
        X, Y = self._check_shapes_types(X, Y)
        X_orig = X

        if not self.warm_start or not hasattr(self, "coef_"):
            self._init_coef()

        # Preprocess X
        X = _reshape_for_est(X)
        # Preprocess coef
        # HACK: reshape coefficients to make sense for SGD
        # Better to just provide a property for reading nicely shaped coefs
        coef_shape = self.coef_.shape
        self.coef_ = self.coef_.view((-1, self.n_outputs_)).requires_grad_()

        self.optim(self, X, Y, **kwargs)

        self.coef_ = self.coef_.detach().view(coef_shape)

        Y_pred = self.predict(X_orig)
        self.residuals_ = Y_pred - Y

        return self

    @property
    def sigma(self):
        if hasattr(self, "residuals_"):
            # Estimate forward model sigma from variance of residuals
            return self.residuals_.std()
        else:
            return torch.tensor(1.)

    @typechecked
    def predict(self, X: TRFDesignMatrix) -> TRFResponse:
        X = _reshape_for_est(X)
        coef = self.coef_.reshape((-1, self.n_outputs_))
        return X @ coef

    @typechecked
    def log_prob(self, X: TRFDesignMatrix, Y: TRFResponse):
        # TODO this is log likelihood, not posterior -- make that clear
        Y_pred = self.predict(X)
        Y_dist = dist.Normal(Y_pred, self.sigma)
        return Y_dist.log_prob(Y)


# class GroupTemporalReceptiveField(BaseEstimator):
#     """
#     Temporal receptive field model estimated and scored at
#     the group level, combining multiple independent datasets.
#     """

#     def __init__(self, cfg: TRFModelConfig):
#         self.trf = TemporalReceptiveField(cfg)

#     def _scatter(self, datasets: List[BerpDataset]
#                  recognition_points: Optional[List[List[int]]] = None,
#                  ) -> Tuple[TensorType["n_times", "n_features"],
#                             TensorType["n_times", "n_outputs"]]:
#         """
#         Join group-level data into a single array, scattering 
#         variable-onset data according to `recognition_points`.

#         If `recognition_points` is none, all recognition points are
#         assumed to be zero (i.e. at word-onset).
#         """
#         raise NotImplementedError()

#     def fit(self, datasets: List[BerpDataset]):

class XYTransformerMixin:
    """
    Transformer which acts on both X and Y inputs.
    """

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)


class GroupScatterTransform(XYTransformerMixin):
    """
    Simultaneously joins grouped time series data into a single array,
    and scatters variable-onset features onto the time series.

    TODO account for resulting invalid samples at join boundaries.
    """

    def fit(self, *args, **kwargs):
        return self

    def partial_fit(self, *args, **kwargs):
        return self

    def _scatter_single(self, dataset: BerpDataset):
        target_samples = time_to_sample(dataset.word_onsets, dataset.sample_rate)

        X = dataset.X_ts[:]

        X_scattered = torch.zeros((X.shape[0], dataset.X_variable.shape[1]))
        X_scattered[target_samples] = dataset.X_variable

        X = torch.concat((X, X_scattered), dim=1)

        return X, dataset.Y

    @typechecked
    def transform(self, datasets: Union[List[BerpDataset], NestedBerpDataset],
                  *args
                  ) -> Tuple[TRFPredictors, TRFResponse]:
        if isinstance(datasets, NestedBerpDataset):
            # NB ignores splits.
            datasets = datasets.datasets

        X, Y = zip(*[self._scatter_single(dataset) for dataset in datasets])
        return torch.cat(X, dim=0), torch.cat(Y, dim=0)


class TRFDelayer(XYTransformerMixin):
    """
    Prepare design matrix for TRF learning/prediction.
    """

    def __init__(self, tmin: float, tmax: float, sfreq: float, **kwargs):
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq

    def fit(self, *args, **kwargs):
        return self

    def partial_fit(self, *args, **kwargs):
        return self

    @typechecked
    def transform(self, X: TRFPredictors, y=None) -> Tuple[TRFDesignMatrix, Any]:
        # TODO fill_mean
        return _delay_time_series(X, self.tmin, self.tmax, self.sfreq), y


def BerpTRF(cfg: TRFModelConfig, optim_cfg: SolverConfig):
    trf_delayer = TRFDelayer(**cfg)
    optim = hydra.utils.instantiate(optim_cfg)
    trf = hydra.utils.instantiate(cfg, optim=optim)

    # TODO caching
    return PartialPipeline([
        ("naive_scatter", GroupScatterTransform()),
        ("trf_delay", trf_delayer),
        ("trf", trf)])


def _times_to_delays(tmin, tmax, sfreq) -> torch.Tensor:
    """Convert a tmin/tmax in seconds to delays."""
    # Convert seconds to samples
    delays = torch.arange(int(np.round(tmin * sfreq)),
                        int(np.round(tmax * sfreq) + 1))
    return delays


def _delay_time_series(X, tmin, tmax, sfreq, fill_mean=False):
    """Return a time-lagged input time series.
    Parameters
    ----------
    X : array, shape (n_times[, n_epochs], n_features)
        The time series to delay. Must be 2D or 3D.
    tmin : int | float
        The starting lag.
    tmax : int | float
        The ending lag.
        Must be >= tmin.
    sfreq : int | float
        The sampling frequency of the series. Defaults to 1.0.
    fill_mean : bool
        If True, the fill value will be the mean along the time dimension
        of the feature, and each cropped and delayed segment of data
        will be shifted to have the same mean value (ensuring that mean
        subtraction works properly). If False, the fill value will be zero.
    Returns
    -------
    delayed : array, shape(n_times[, n_epochs][, n_features], n_delays)
        The delayed data. It has the same shape as X, with an extra dimension
        appended to the end.
    Examples
    --------
    >>> tmin, tmax = -0.1, 0.2
    >>> sfreq = 10.
    >>> x = np.arange(1, 6)
    >>> x_del = _delay_time_series(x, tmin, tmax, sfreq)
    >>> print(x_del)  # doctest:+SKIP
    [[2. 1. 0. 0.]
    [3. 2. 1. 0.]
    [4. 3. 2. 1.]
    [5. 4. 3. 2.]
    [0. 5. 4. 3.]]
    """
    delays = _times_to_delays(tmin, tmax, sfreq)
    # Iterate through indices and append
    delayed = torch.zeros(X.shape + (len(delays),), dtype=X.dtype)
    if fill_mean:
        mean_value = X.mean(dim=0)
        delayed[:] = mean_value[:, None]
    for ii, ix_delay in enumerate(delays):
        # Create zeros to populate w/ delays
        if ix_delay < 0:
            out = delayed[:ix_delay, ..., ii]
            use_X = X[-ix_delay:]
        elif ix_delay > 0:
            out = delayed[ix_delay:, ..., ii]
            use_X = X[:-ix_delay]
        else:  # == 0
            out = delayed[..., ii]
            use_X = X
        out[:] = use_X
        if fill_mean:
            out[:] += (mean_value - use_X.mean(dim=0))
    return delayed


def _reshape_for_est(X_del):
    """Convert X_del to a sklearn-compatible shape."""
    n_times, n_feats, n_delays = X_del.shape
    X_del = X_del.reshape(n_times, -1)
    return X_del