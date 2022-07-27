import torch

from berp.models.trf import TemporalReceptiveField



def test_trf_dummy():
    # three events which have an effect on Y at 1 time sample delay
    X = torch.tensor([[1],
                      [0],
                      [1],
                      [0]]).float()
    Y = torch.tensor([[0, 0],
                      [1, -1],
                      [0, 0],
                      [1, -1]]).float()

    trf = TemporalReceptiveField(tmin=0, tmax=2, sfreq=1, feature_names=["a"],
                                 fit_intercept=False)
    trf.fit(X, Y)

    expected_coef = torch.tensor([[0, 0], [1, -1], [0, 0]]).float()

    torch.allclose(trf.coef_, expected_coef)