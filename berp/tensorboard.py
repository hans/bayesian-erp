from typing import Dict, Any

from torch.utils.tensorboard import SummaryWriter


class Tensorboard:
    """
    Single access point for Tensorboard SummaryWriter which manages global steps.
    """

    @staticmethod
    def instance(*args, **kwargs):
        if not hasattr(Tensorboard, "_instance"):
            Tensorboard._instance = Tensorboard(*args, **kwargs)
        return Tensorboard._instance

    def __init__(self, log_dir=".", **kwargs):
        # We use default log_dir=. because we've already changed dir into the Hydra
        # output directory.
        self.global_step = 0
        self.summary_writer = SummaryWriter(
            log_dir=log_dir,
            **kwargs
        )

    def add_scalar(self, tag, scalar_value, global_step=None):
        if global_step is None:
            global_step = self.global_step
        self.summary_writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        if global_step is None:
            global_step = self.global_step
        self.summary_writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def add_histogram(self, tag, values, global_step=None):
        if global_step is None:
            global_step = self.global_step
        self.summary_writer.add_histogram(tag, values, global_step)

    def flush(self):
        self.summary_writer.flush()

    def close(self):
        self.summary_writer.close()


def tb_add_scalar(tag, scalar_value, global_step=None):
    Tensorboard.instance().add_scalar(tag, scalar_value, global_step)


def tb_add_histogram(tag, values, global_step=None):
    Tensorboard.instance().add_histogram(tag, values, global_step)