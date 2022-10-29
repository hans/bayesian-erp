from typing import Dict, Any

from torch.utils.tensorboard import SummaryWriter


class Tensorboard:
    """
    Single access point for Tensorboard SummaryWriter which manages global steps.
    """

    _disabled = False

    @classmethod
    def disable(cls):
        cls._disabled = True

    @staticmethod
    def instance(*args, **kwargs):
        if not hasattr(Tensorboard, "_instance"):
            Tensorboard._instance = Tensorboard(*args, **kwargs)
        return Tensorboard._instance

    def __init__(self, log_dir=".", **kwargs):
        if self._disabled:
            return

        # We use default log_dir=. because we've already changed dir into the Hydra
        # output directory.
        self.global_step = 0
        self.summary_writer = SummaryWriter(
            log_dir=log_dir,
            **kwargs
        )

    def add_scalar(self, tag, scalar_value, global_step=None):
        if self._disabled:
            return
        if global_step is None:
            global_step = self.global_step
        self.summary_writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        if self._disabled:
            return
        if global_step is None:
            global_step = self.global_step
        self.summary_writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def add_histogram(self, tag, values, global_step=None):
        if self._disabled:
            return
        if global_step is None:
            global_step = self.global_step
        self.summary_writer.add_histogram(tag, values, global_step)

    def add_figure(self, tag, figure, global_step=None):
        if self._disabled:
            return
        if global_step is None:
            global_step = self.global_step
        self.summary_writer.add_figure(tag, figure, global_step)

    def flush(self):
        if self._disabled:
            return
        self.summary_writer.flush()

    def close(self):
        if self._disabled:
            return
        self.summary_writer.close()


def tb_add_scalar(tag, scalar_value, global_step=None):
    Tensorboard.instance().add_scalar(tag, scalar_value, global_step)


def tb_add_histogram(tag, values, global_step=None):
    Tensorboard.instance().add_histogram(tag, values, global_step)


def tb_add_figure(tag, figure, global_step=None):
    Tensorboard.instance().add_figure(tag, figure, global_step)


def tb_global_step(global_step=None):
    """
    Increment or set global step.
    """
    tb = Tensorboard.instance()
    if global_step is None:
        global_step = tb.global_step + 1
    tb.global_step = global_step

    return global_step