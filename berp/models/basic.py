"""
Defines a simple Bayesian linear regression model.
"""

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


class RegressionModel(PyroModule):
    def __init__(self):
        super().__init__()
        self.coef = PyroSample(dist.Normal(0., 2.))
        self.bias = PyroSample(dist.Normal(0., 2.))

    def forward(self, X, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        y_hat = self.coef * X + self.bias

        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Normal(y_hat.squeeze(-1), sigma), obs=y)

        return y_hat
