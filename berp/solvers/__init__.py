from functools import cached_property
import logging
from typing import *

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
import torch
from tqdm.auto import tqdm, trange

from berp.cv import EarlyStopException

L = logging.getLogger(__name__)


class Solver(BaseEstimator):
    pass


# TODO clean up and move
class AdamSolver(Solver):
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
