import pytest
import torch

from berp.models.trf import TRFPipeline



def test_trf_dummy():
    tmin, tmax, sfreq = 0, 2, 1

    # two events which have an effect on Y at 1 time sample delay
    X = torch.tensor([[1],
                      [0],
                      [1],
                      [0]]).float()
    Y = torch.tensor([[0, 0],
                      [1, -1],
                      [0, 0],
                      [1, -1]]).float()

    trf = TRFPipeline(standardize_X=False, standardize_Y=False,
                      tmin=tmin, tmax=tmax, sfreq=sfreq,
                      n_outputs=2, fit_intercept=False,
                      name="Gerald")
    trf.fit(X, Y)

    expected_coef = torch.tensor([[0, 0], [1, -1], [0, 0]]).float()
    coef = trf.named_steps["trf"].coef_

    torch.allclose(coef, expected_coef)



def test_trf_lookback():
    tmin, tmax, sfreq = -1, 1, 1

    # two events which have an effect on Y at -1 time sample delay
    X = torch.tensor([[0],
                      [1],
                      [0],
                      [0],
                      [1],
                      [0]]).float()
    Y = torch.tensor([[1, -1],
                      [0, 0],
                      [0, 0],
                      [1, -1],
                      [0, 0],
                      [0, 0]]).float()

    trf = TRFPipeline(standardize_X=False, standardize_Y=False,
                      tmin=tmin, tmax=tmax, sfreq=sfreq,
                      n_outputs=2, fit_intercept=False,
                      name="NegativeGerald")
    trf.fit(X, Y)

    expected_coef = torch.tensor([[1, -1], [0, 0], [0, 0]]).float()
    coef = trf.named_steps["trf"].coef_

    torch.allclose(coef, expected_coef)