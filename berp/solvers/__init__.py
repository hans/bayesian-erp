from functools import cached_property
import logging
from typing import *

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import check_random_state
import torch
from tqdm.auto import tqdm, trange

from berp.cv import EarlyStopException
from berp.tensorboard import tb_add_scalar, tb_global_step

L = logging.getLogger(__name__)


class Solver(BaseEstimator):
    pass


class SGDSolver(Solver):
    """
    Model mixin which supports advanced stochastic gradient descent
    methods and provides early stopping facilities.

    Whenever the dataset or estimator structure changes you should call `.prime()`
    """

    def __init__(self, learning_rate: float = 0.01,
                 n_batches: int = 8,
                 batch_size: int = 512,
                 early_stopping: Optional[int] = 5,
                 random_state=None,
                 name=None,
                 pbar=False,
                 **kwargs):
        self.learning_rate = learning_rate
        self.n_batches = n_batches
        self.batch_size = batch_size

        self.early_stopping = early_stopping
        self.random_state = check_random_state(random_state)

        self.name = name
        self.pbar = pbar

        self._optim_parameters = None
        self._primed = False

        # prepare trackers for early stopping
        self.reset_early_stopping(reset_loss=True)

        if kwargs:
            L.warning("Unused kwargs: %s", kwargs)

    @cached_property
    def _optim(self):
        # TODO make sure this is not carried across history
        return torch.optim.SGD(self._optim_parameters, lr=self.learning_rate)

    def reset(self):
        self._primed = False
        self.validation_mask = None

    def reset_early_stopping(self, reset_loss=False):
        """
        Reset early stopping tracker. If `reset_loss`, also forget about best val loss.
        """
        self._has_early_stopped = False
        self._no_improvement_count = 0
        if reset_loss:
            self._best_val_loss = np.inf

    def prime(self, estimator, X, y):
        if self._primed:
            assert len(self._optim_parameters) == len(estimator._optim_parameters)
            return

        self._primed = True
        self._optim_parameters = estimator._optim_parameters

        # TODO probably should support early stopping resets here. but the clients
        # currently abusively call prime() every iteration, which would make early
        # stopping break of course. putting this off.

    def _tb_scalar(self, tag, value):
        tag = f"{self.__class__.__name__}/{self.name}/{tag}"
        tb_add_scalar(tag, value)

    def _process_loss(self, loss: Union[torch.Tensor, List[torch.Tensor]],
                      tag_prefix="") -> torch.Tensor:
        if isinstance(loss, (list, tuple)):
            for i, loss_i in enumerate(loss):
                self._tb_scalar(f"{tag_prefix}loss_{i}", loss_i.item())
            loss = sum(loss)
        else:
            self._tb_scalar(f"{tag_prefix}loss", loss.item())

        return loss

    def __call__(self, loss_fn: Callable[[torch.Tensor, torch.Tensor],
                                         List[torch.Tensor]],
                 X, y,
                 validation_mask: Optional[np.ndarray] = None,
                 dataset_tag: Optional[str] = None,
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

        total_num_batches = int(np.ceil(len(X_train) / self.batch_size))
        batch_cursor = 0

        # Prepare tensorboard tag strings
        train_tag_str = f"{dataset_tag}/" if dataset_tag else ""
        valid_tag_str = f"{dataset_tag}/val/" if dataset_tag else "val/"

        # Shuffle
        indices = np.arange(len(X_train))
        self.random_state.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        losses = []
        for i in range(self.n_batches):
            batch_start = batch_cursor * self.batch_size
            batch_end = (batch_cursor + 1) * self.batch_size

            batch_X = X_train[batch_start:batch_end]
            batch_y = y_train[batch_start:batch_end]

            self._optim.zero_grad()
            loss = loss_fn(batch_X, batch_y)
            loss: torch.Tensor = self._process_loss(
                loss, tag_prefix=train_tag_str)
            assert not torch.isnan(loss), "nan loss"

            loss.backward()
            self._optim.step()

            losses.append(loss.item())

            if self.early_stopping and batch_cursor % 10 == 0:
                with torch.no_grad():
                    valid_loss = loss_fn(X_valid, y_valid, include_l2=False)
                valid_loss = self._process_loss(
                    valid_loss, tag_prefix=valid_tag_str)

                if valid_loss >= self._best_val_loss:
                    self._no_improvement_count += 1
                else:
                    self._no_improvement_count = 0
                    self._best_val_loss = valid_loss

                if self._no_improvement_count > self.early_stopping:
                    L.info("Stopping early due to no improvement.")
                    self._has_early_stopped = True
                    raise EarlyStopException()

            batch_cursor = (batch_cursor + 1) % total_num_batches
            tb_global_step()

        return self


# TODO clean up and move
class AdamSolver(SGDSolver):

    @cached_property
    def _optim(self):
        # TODO make sure this is not carried across history
        return torch.optim.Adam(self._optim_parameters, lr=self.learning_rate)