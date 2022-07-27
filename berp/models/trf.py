from typing import Tuple
import numpy as np
import torch
import torch.distributions as dist
from torchtyping import TensorType


class TemporalReceptiveField(object):

    def __init__(self, tmin, tmax, sfreq,
                 feature_names,
                 fit_intercept=True,
                 alpha=None):
        self.feature_names = feature_names
        self.sfreq = float(sfreq)

        self.tmin = tmin
        self.tmax = tmax
        assert self.tmin < self.tmax

        self.fit_intercept = fit_intercept
        self.alpha = alpha

    def fit(self, X: TensorType["n_times", "n_features"],
            Y: TensorType["n_times", "n_outputs"],
            ) -> "TemporalReceptiveField":
        
        # Initialize delays.
        self.delays_ = _times_to_delays(self.tmin, self.tmax, self.sfreq)

        # TODO valid_samples_

        # Delay input features.
        X_del: TensorType["n_times", "n_features", "n_delays"] = \
            _delay_time_series(X, self.tmin, self.tmax, self.sfreq,
                               fill_mean=self.fit_intercept)
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

        Y_pred = self.predict(X)
        self.residuals_ = Y_pred - Y
        return self

    @property
    def sigma(self):
        # Estimate forward model sigma from variance of residuals
        return self.residuals_.std()

    def predict(self, X: TensorType["n_times", "n_features"]
                ) -> TensorType["n_times", "n_outputs"]:
        X = _delay_time_series(X, self.tmin, self.tmax, self.sfreq,
                               fill_mean=self.fit_intercept)
        X = _reshape_for_est(X)
        return X @ self.coef_

    def log_prob(self, X, Y):
        Y_pred = self.predict(X)
        Y_dist = dist.Normal(Y_pred, self.sigma)
        return Y_dist.log_prob(Y)


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