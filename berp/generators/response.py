"""
Functions which simulate N400-like neural responses
"""

from typing import Tuple

import numpy as np
import torch

from berp.util import gaussian_window


def simple_peak(width: float, delay: float, sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
    delay_num_samples = int(delay * sample_rate)
    peak_num_samples = int(width * sample_rate)

    delay_xs = torch.arange(delay_num_samples) / sample_rate
    peak_xs = torch.arange(delay_num_samples, delay_num_samples + peak_num_samples) / sample_rate

    delay_ys = torch.zeros_like(delay_xs)
    peak_ys = torch.ones_like(peak_xs)

    return torch.cat((delay_xs, peak_xs), dim=0), torch.cat((delay_ys, peak_ys), dim=0)


def simple_gaussian(width, delay, sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
    window_std = width / 4
    window_center = delay + width / 2
    xs, ys = gaussian_window(window_center.item(), window_std.item(),
                             sample_rate=sample_rate)
    return torch.tensor(xs), torch.tensor(ys)


def n400_like(surprisal, sample_rate=128, rng=np.random) -> Tuple[torch.Tensor, torch.Tensor]:
    # Stolen from https://github.com/christianbrodbeck/Eelbrain/blob/master/eelbrain/datasets/_sim_eeg.py

    # Generate topography
    n400_topo = -1.0  # * _topo(sensor, 'Cz')
    # Generate timing
    times, n400_timecourse = gaussian_window(0.400, 0.034)
    # Put all the dimensions together to simulate the EEG signal
    signal = surprisal * n400_timecourse * n400_topo

    # add early responses:
    # 130 ms
    _, tc = gaussian_window(0.130, 0.025)
    # topo = _topo(sensor, 'O1') + _topo(sensor, 'O2') - 0.5 * _topo(sensor, 'Cz')
    signal += 0.5 * tc  # * topo
    # 195 ms
    amp = rng.normal(0.4, 0.25)
    _, tc = gaussian_window(0.195, 0.015)
    # topo = 1.2 * _topo(sensor, 'F3') + _topo(sensor, 'F4')
    signal += amp * tc  # * topo
    # 270
    amp = rng.normal(1, 1)
    _, tc = gaussian_window(0.270, 0.050)
    # topo = _topo(sensor, 'O1') + _topo(sensor, 'O2')
    signal += amp * tc  # * topo
    # 280
    amp = rng.normal(-1, 1)
    _, tc = gaussian_window(0.280, 0.030)
    # topo = _topo(sensor, 'Pz')
    signal += amp * tc  # * topo
    # 600
    amp = rng.normal(0.5, 0.1)
    _, tc = gaussian_window(0.590, 0.100)
    # topo = -_topo(sensor, 'Fz')
    signal += amp * tc  # * topo

    # Add noise
    # noise = powerlaw_noise(signal, 1, rng)
    # noise = noise.smooth('sensor', 0.02, 'gaussian')
    # noise *= (signal.std() / noise.std() / snr)
    noise = rng.normal(0.2, 0.05, size=len(signal))
    signal += noise

    # # Data scale
    # signal *= 1e-6

    times = torch.tensor(times)
    signal = torch.tensor(signal)

    return times, signal