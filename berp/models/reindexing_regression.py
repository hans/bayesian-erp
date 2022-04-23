"""
Defines a Bayesian model which samples different response columns per sample.
"""

import pyro
from pyro import poutine
import pyro.distributions as dist
from pyro.infer import config_enumerate
from pyro.nn import PyroModule, PyroSample
from pyro.ops.indexing import Vindex
import torch
from torch.distributions import constraints


def build_model_guide(bayesian=True):
    def model(X, y):
        coef = pyro.sample("coef", dist.Normal(1., 0.1))

        # prior = pyro.sample("prior", dist.Uniform(0., 1.))
        # geom_ps = pyro.param("geom", dist.Normal(prior.expand(y.shape[0]), 0.1),
        #                      constraint=constraints.unit_interval)
        # geom_ps = prior.expand(y.shape[0])
        samps = pyro.sample("samps", dist.Uniform(0., 1.).expand([y.shape[0]]))
                            # constraint=constraints.interval(0., 1.))
        with pyro.plate("data", X.shape[0]):
            # Compute weights over each Y index.
            eval_pts = torch.arange(y.shape[1]).expand(y.shape).T / (y.shape[1] - 1)
            assert eval_pts.shape == (y.shape[1], y.shape[0]), eval_pts.shape
            weights = dist.Normal(samps, 0.05).log_prob(eval_pts).exp().T
            weights = weights / weights.sum(axis=1, keepdim=True)

            y_target = (y * weights).sum(axis=1)

            y_hat = X * coef

            obs = pyro.sample("obs", dist.Normal(y_hat, 1.),
                              obs=y_target)

    def guide(X, y):
        coef = pyro.sample("coef", dist.Normal(1., 0.1))
        # with pyro.plate("data", X.shape[0]):
        #     index = pyro.sample("index", dist.Categorical(weights))

    def discrete_guide(X, y):
        # Don't let inner guide handle params.
        with poutine.block(hide_types=["param"]):
            guide(X, y)

        prior = pyro.param("prior", dist.Normal(0.5, 0.2),
                           constraint=constraints.interval(0., 1.))
        # geom_ps = pyro.param("geom", dist.Normal(prior.expand(y.shape[0]), 0.1),
        #                      constraint=constraints.unit_interval)
        # geom_ps = prior.expand(y.shape[0])
        samps = pyro.param("samps", dist.Normal(prior.expand(y.shape[0]), 0.2),
                           constraint=constraints.interval(0., 1.))

    if not bayesian:
        model = model#config_enumerate(model)
        full_guide = discrete_guide#config_enumerate(discrete_guide)

    return model, full_guide



        # index = pyro.sample("index", dist.Categorical(torch.ones(y.shape[1])))
#
# class ReindexingRegressionModel(PyroModule):
#     def __init__(self, N, d):
#         super().__init__()
#         print(N, d)
#         self.coef = PyroSample(dist.Normal(0., 2.))
#         self.index = PyroSample(dist.Categorical(torch.ones(d)).expand([N]))
#
#     def forward(self, X, y=None):
#         print("====", X.shape, y.shape if y is not None else y,
#               self.index.shape,
#               self.index.squeeze().unsqueeze(-1).shape)
#         sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
#
#         y_hat = self.coef * X
#         y_target = Vindex(y)[:, self.index]
#         y_target = y[torch.arange(y.shape[0]).reshape(-1, 1),
#                      self.index.squeeze().unsqueeze(-1)]
#         print("\t-> ", y_target.shape, y_target.squeeze(-1).shape)
#
#         with pyro.plate("data", X.shape[0]):
#             obs = pyro.sample("obs", dist.Normal(y_hat.squeeze(-1), sigma), obs=y_target.squeeze(-1))
#
#         return y_hat
