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
        coef = pyro.sample("coef", dist.Normal(1., 1))

        prior = pyro.param("prior", dist.Dirichlet(torch.ones(y.shape[1]).unsqueeze(0)),
                           constraint=constraints.positive)
        with pyro.plate("data", X.shape[0]):
            index_probs = pyro.sample("index_probs", dist.Dirichlet(prior.expand(y.shape)))
            index = pyro.sample("index", dist.Categorical(index_probs))

            y_hat = X * coef
            y_target = Vindex(y)[..., index]

            obs = pyro.sample("obs", dist.Normal(y_hat, 5.),
                              obs=y_target)

    def guide(X, y):
        coef = pyro.sample("coef", dist.Normal(1., 1))
        # with pyro.plate("data", X.shape[0]):
        #     index = pyro.sample("index", dist.Categorical(weights))

    def discrete_guide(X, y):
        # Don't let inner guide handle params.
        with poutine.block(hide_types=["param"]):
            guide(X, y)

        prior = pyro.param("prior", dist.Dirichlet(torch.ones(y.shape[1]).unsqueeze(0)),
                           constraint=constraints.positive)
        with pyro.plate("data", X.shape[0]):
            index_probs = pyro.sample("index_probs", dist.Dirichlet(prior.expand(y.shape)))
            pyro.sample("index", dist.Categorical(index_probs))

    if not bayesian:
        model = config_enumerate(model)
        full_guide = config_enumerate(discrete_guide)

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
