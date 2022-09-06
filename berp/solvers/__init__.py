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
                 random_state=None,
                 pbar=False,
                 **kwargs):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.early_stopping = early_stopping
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
        return torch.optim.SGD(self._optim_parameters, lr=self.learning_rate)

    def reset(self):
        self._primed = False
        self.validation_mask = None

    def prime(self, estimator, X, y):
        if self._primed:
            assert len(self._optim_parameters) == len(estimator._optim_parameters)
            return

        self._primed = True
        self._optim_parameters = estimator._optim_parameters
        self._has_early_stopped = False

    def __call__(self, loss_fn, X, y,
                 validation_mask: Optional[np.ndarray] = None,
                 **fit_params):
        if self._has_early_stopped:
            L.info("Early stopped, skipping")
            return

        if self.early_stopping:
            if validation_mask is None:
                raise ValueError("validation_mask must be provided if early_stopping is enabled")

            assert X.shape[0] == y.shape[0]
            assert X.shape[0] == validation_mask.shape[0]

            X_train, y_train = X[~validation_mask], y[~validation_mask]
            X_valid, y_valid = X[validation_mask], y[validation_mask]
        else:
            X_train, y_train = X, y
            
        from berp.util import tensor_hash
        print("solver X train", X_train[64])
        print(X_train.nonzero()[:15])
        print("solver Y train", tensor_hash(y_train))

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
                    print("batch size", self.batch_size, batch_offset, X_train.shape, batch_X.shape)
                    print("optim/batch_X", tensor_hash(batch_X))
                    print("optim/batch_y", tensor_hash(batch_y))

                    self._optim.zero_grad()
                    loss = loss_fn(batch_X, batch_y)
                    loss.backward()
                    self._optim.step()

                    losses.append(loss.item())
                    # DEV
                    return self

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