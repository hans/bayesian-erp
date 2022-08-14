import logging
from typing import Tuple, List, Optional

import hydra
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.distributions as dist
from torchtyping import TensorType
from typeguard import typechecked
from tqdm.auto import tqdm, trange

from berp.config.model import TRFModelConfig
from berp.datasets.base import BerpDataset
from berp.util import time_to_sample, PartialPipeline



TRFPredictors = TensorType["n_times", "n_features"]
TRFDesignMatrix = TensorType["n_times", "n_features", "n_delays"]
TRFResponse = TensorType["n_times", "n_outputs"]

class TemporalReceptiveField(BaseEstimator):

    def __init__(self, tmin, tmax, sfreq, fit_intercept=False,
                 warm_start=True, alpha=1, **kwargs):
        self.sfreq = sfreq

        self.tmin = tmin
        self.tmax = tmax
        assert self.tmin < self.tmax

        self.fit_intercept = fit_intercept
        self.warm_start = warm_start
        self.alpha = alpha

        self.delays_ = _times_to_delays(self.tmin, self.tmax, self.sfreq)

        if kwargs:
            logging.warning(f"Unused arguments: {kwargs}")

    def _init_coef(self):
        self.coef_ = torch.randn(self.n_features_, len(self.delays_),
                                 self.n_outputs_) * 1e-1

    def _check_shapes_types(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        # May not be available if we haven't been called with fit() yet.
        if hasattr(self, "n_features_"):
            assert X.shape[1] == self.n_features_
            assert Y.shape[1] == self.n_outputs_
        return torch.as_tensor(X), torch.as_tensor(Y)

    def fit(self, X: TRFPredictors, Y: TRFResponse
            ) -> "TemporalReceptiveField":
        """
        Fit the TRF encoder with least squares.
        """
        
        self.n_features_ = X.shape[-1]
        self.n_outputs_ = Y.shape[-1]

        # TODO valid_samples_

        # Delay input features.
        X_del: TRFDesignMatrix = \
            _delay_time_series(X, self.tmin, self.tmax, self.sfreq,
                               fill_mean=self.fit_intercept)
        n_times, _, self.n_delays_ = X_del.shape
        X_est = _reshape_for_est(X_del)

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

    def partial_fit(self, X: TRFPredictors, Y: TRFResponse,
                    **kwargs) -> "TemporalReceptiveField":
        """
        Update the TRF encoder weights with gradient descent.
        """

        X, Y = self._check_shapes_types(X, Y)

        self.n_features_ = X.shape[-1]
        self.n_outputs_ = Y.shape[-1]
        if not self.warm_start or not hasattr(self, "coef_"):
            self._init_coef()

        X_orig = X

        # Preprocess X
        X = _delay_time_series(X, self.tmin, self.tmax, self.sfreq,
                               fill_mean=self.fit_intercept)
        n_times, _, self.n_delays_ = X.shape
        X = _reshape_for_est(X)
        # Preprocess coef
        coef = self.coef_.view((-1, self.n_outputs_)).requires_grad_()

        def loss_fn(batch_onset, batch_offset):
            X_b, Y_b = X[batch_onset:batch_offset], Y[batch_onset:batch_offset]
            Y_b_pred = X_b @ coef

            loss = (Y_b_pred - Y_b).pow(2).sum(axis=1).mean()
            # Add ridge term.
            loss += self.alpha * torch.norm(coef, p=2)

            return loss

        # TODO remove magic numbers
        n_epochs = 2
        optimizer = torch.optim.Adam([coef], lr=0.05)
        batch_size = 512
        for i in trange(n_epochs, leave=False):
            losses = []
            for batch_offset in torch.arange(0, X.shape[0], batch_size):
                optimizer.zero_grad()
                loss = loss_fn(batch_offset, batch_offset + batch_size)
                loss.backward()
                losses.append(loss)
                optimizer.step()

        self.coef_ = coef.detach().view((self.n_features_, self.n_delays_, self.n_outputs_))

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

    def predict(self, X: TensorType["n_times", "n_features"]
                ) -> TensorType["n_times", "n_outputs"]:
        X = _delay_time_series(X, self.tmin, self.tmax, self.sfreq,
                               fill_mean=self.fit_intercept)
        X = _reshape_for_est(X)
        coef = self.coef_.reshape((-1, self.n_outputs_))
        return X @ coef

    def log_prob(self, X, Y):
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



class GroupScatterTransform(TransformerMixin):
    """
    Simultaneously joins grouped time series data into a single array,
    and scatters variable-onset features onto the time series.

    TODO account for resulting invalid samples at join boundaries.
    """

    def partial_fit(*args, **kwargs):
        pass

    def _scatter_single(self, dataset: BerpDataset):
        target_samples = time_to_sample(dataset.word_onsets, dataset.sample_rate)

        X = dataset.X_ts[:]

        X_scattered = torch.zeros((X.shape[0], dataset.X_variable.shape[1]))
        X_scattered[target_samples] = dataset.X_variable

        X = torch.concat((X, X_scattered), dim=1)

        return X, dataset.Y

    @typechecked
    def transform(self, datasets: List[BerpDataset]
                  ) -> Tuple[TRFPredictors, TRFResponse]:
        X, Y = zip(*[self._scatter_single(dataset) for dataset in datasets])
        return torch.cat(X, dim=0), torch.cat(Y, dim=0)


def BerpTRF(cfg: TRFModelConfig):
    trf = hydra.utils.instantiate(cfg)

    return PartialPipeline([
        ("naive_scatter", GroupScatterTransform()),
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
    delayed = torch.zeros(X.shape + (len(delays),))
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