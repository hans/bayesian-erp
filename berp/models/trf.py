from functools import cached_property
import logging
from typing import Tuple, List, Optional, Any, Union

import hydra
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ShuffleSplit
import torch
import torch.distributions as dist
from torchtyping import TensorType
from typeguard import typechecked
from tqdm.auto import tqdm, trange

from berp.config.model import TRFModelConfig
from berp.config.solver import SolverConfig
from berp.cv import EarlyStopException
from berp.datasets.base import BerpDataset, NestedBerpDataset
from berp.util import time_to_sample, PartialPipeline, XYTransformerMixin, StandardXYScaler

L = logging.getLogger(__name__)


TRFPredictors = TensorType["n_times", "n_features"]
TRFDesignMatrix = TensorType["n_times", "n_features", "n_delays"]
TRFResponse = TensorType["n_times", "n_outputs"]


# TODO clean up and move
# TODO early stopping
class AdamSolver(BaseEstimator):
    """
    Model mixin which supports advanced stochastic gradient descent
    methods.

    Whenever the dataset or estimator structure changes you should call `.prime()`
    """

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 1,
                 batch_size: int = 512,
                 early_stopping: Optional[int] = 5,
                 validation_fraction: float = 0.1,
                 random_state=None,
                 pbar=False,
                 **kwargs):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.random_state = random_state

        self.pbar = pbar

        self._optim_parameters = None
        self._primed = False
        self._has_early_stopped = False

        if kwargs:
            L.warning("Unused kwargs: %s", kwargs)

    @cached_property
    def _optim(self):
        # TODO make sure this is not carried across history
        return torch.optim.Adam(self._optim_parameters, lr=self.learning_rate)

    def reset(self):
        self._primed = False
        self.validation_mask = None

    def prime(self, estimator, X, y):
        if self._primed:
            assert len(self._optim_parameters) == len(estimator._optim_parameters)
            assert y.shape[0] == self.validation_mask.shape[0]
            return

        self._primed = True
        self._optim_parameters = estimator._optim_parameters
        self._has_early_stopped = False
        self.validation_mask = self._make_validation_split(y)

    def __call__(self, loss_fn, X, y, **fit_params):
        if not self._primed:
            raise RuntimeError("Solver must be primed before calling.")
        if self._has_early_stopped:
            L.info("Early stopped, skipping")
            return

        if self.early_stopping:
            X_train, y_train = X[~self.validation_mask], y[~self.validation_mask]
            X_valid, y_valid = X[self.validation_mask], y[self.validation_mask]
        else:
            X_train, y_train = X, y

        best_val_loss = np.inf
        no_improvement_count = 0
        n_batches = 0
        stop = False
        with trange(self.n_epochs, leave=False, disable=not self.pbar) as pbar:
            for i in pbar:
                losses = []
                postfix = {}
                for batch_offset in torch.arange(0, X_train.shape[0], self.batch_size):
                    batch_X = X_train[batch_offset:batch_offset + self.batch_size]
                    batch_y = y_train[batch_offset:batch_offset + self.batch_size]

                    self._optim.zero_grad()
                    loss = loss_fn(batch_X, batch_y)
                    loss.backward()
                    self._optim.step()

                    losses.append(loss.item())

                    if n_batches % 10 == 0:
                        if self.early_stopping:
                            with torch.no_grad():
                                valid_loss = loss_fn(X_valid, y_valid)
                            
                            postfix["val_loss"] = valid_loss.item()

                            if valid_loss >= best_val_loss:
                                no_improvement_count += 1
                            else:
                                no_improvement_count = 0
                                best_val_loss = valid_loss

                            if no_improvement_count > self.early_stopping:
                                L.debug("Stopping early due to no improvement.")
                                stop = True
                                self._has_early_stopped = True
                                raise EarlyStopException()
                    
                    n_batches += 1

                postfix["loss"] = np.mean(losses)
                pbar.set_postfix(postfix)

                if stop:
                    break

        return self

    def _make_validation_split(self, y):
        """Split the dataset between training set and validation set.
        Parameters
        ----------
        y : ndarray of shape (n_samples, )
            Target values.
        Returns
        -------
        validation_mask : ndarray of shape (n_samples, )
            Equal to True on the validation set, False on the training set.
        """

        # # TODO make this into a unit test--should be called only once per prime
        # L.info("Recalc validation mask")

        n_samples = y.shape[0]
        validation_mask = torch.zeros(n_samples).bool()
        if not self.early_stopping:
            # use the full set for training, with an empty validation set
            return validation_mask

        # TODO valid sample boundaries?
        cv = ShuffleSplit(
            test_size=self.validation_fraction, random_state=self.random_state
        )
        idx_train, idx_val = next(cv.split(np.zeros(shape=(y.shape[0], 1)), y))
        if idx_train.shape[0] == 0 or idx_val.shape[0] == 0:
            raise ValueError(
                "Splitting %d samples into a train set and a validation set "
                "with validation_fraction=%r led to an empty set (%d and %d "
                "samples). Please either change validation_fraction, increase "
                "number of samples, or disable early_stopping."
                % (
                    n_samples,
                    self.validation_fraction,
                    idx_train.shape[0],
                    idx_val.shape[0],
                )
            )

        validation_mask[idx_val] = True
        return validation_mask


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
        elif self.optim._has_early_stopped:
            L.info("Early stopped. skipping")

        # Preprocess X
        X = _reshape_for_est(X)
        # Preprocess coef
        # HACK: reshape coefficients to make sense for SGD
        # Better to just provide a property for reading nicely shaped coefs
        coef_shape = self.coef_.shape
        self.coef_ = self.coef_.view((-1, self.n_outputs_)).requires_grad_()

        # TODO don't need to call this every iteration..
        self.optim.prime(self, X, Y)
        self.optim(self._loss_fn, X, Y, **kwargs)

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

    steps = [
        ("naive_scatter", GroupScatterTransform()),
    ]

    standardize_X, standardize_Y = cfg.standardize_X, cfg.standardize_Y
    if standardize_X or standardize_Y:
        steps.append(("standardize", StandardXYScaler(standardize_X=standardize_X,
                                                      standardize_Y=standardize_Y)))

    steps += [
        ("trf_delay", trf_delayer),
        ("trf", trf),
    ]

    # TODO caching
    from tempfile import mkdtemp
    tmpdir = mkdtemp()
    return PartialPipeline(steps)


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