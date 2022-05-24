#!/usr/bin/env python
# coding: utf-8

# This notebook synthesizes and saves EEG-like data in a CDR-friendly format.
# 
# We are testing here whether an N400-like response is recoverable by CDR(NN).

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm


def simple_peak(x, scale=5, b=0.05):
    """Function which rapidly peaks and returns to baseline"""
    ret = st.gamma.pdf(x * scale, 2.8, b)
    ret /= ret.max()
    return ret


def rate_irf(x):
    return 0.1 * simple_peak(x, scale=20, b=0)


def sample_item(n_words=20, word_delay_range=(0.3, 1),
                response_window=(0.0, 2),  # time window over which word triggers signal response
                sample_rate=128, n400_surprisal_coef=-1,
                surprisal_mean=2., surprisal_sigma=0.2,
                irf=simple_peak, rate_irf=rate_irf):
    # sample unitary response
    peak_xs = np.linspace(*response_window, num=int(response_window[1] * sample_rate), endpoint=False)
    peak_ys = n400_surprisal_coef * irf(peak_xs)
    
    # sample word rate response
    rate_xs = peak_xs.copy()
    rate_ys = rate_irf(rate_xs)

    # sample stimulus times+surprisals
    stim_delays = np.random.uniform(*word_delay_range, size=n_words)
    stim_onsets = np.cumsum(stim_delays)
    # hack: align times to sample rate to make this easier
    stim_onsets = np.round(stim_onsets * sample_rate) / sample_rate
    surprisals = np.random.lognormal(surprisal_mean, surprisal_sigma, size=n_words)

    max_time = stim_onsets.max() + response_window[1]
    all_times = np.linspace(0, max_time, num=int(np.ceil(max_time * sample_rate)), endpoint=False)
    
    signal = np.zeros_like(all_times)
    for onset, surprisal in zip(stim_onsets, surprisals):
        sample_idx = int(onset * sample_rate)  # guaranteed to be round because of hack.
        signal[sample_idx:sample_idx + len(peak_xs)] += peak_ys * surprisal + rate_ys
        
    # build return X, y dataframes.
    X = pd.DataFrame({"time": stim_onsets, "surprisal": surprisals})
    y = pd.DataFrame({"time": all_times, "signal": signal})
    return X, y


def sample_word(n_phons_p=1 / 4., phon_delay_range=(0.1, 0.35),
                surprisal_mean=0.5, surprisal_sigma=0.4,
                **kwargs):
    """
    Sample a trajectory of phoneme surprisals for an individual word.
    """
    n_phons = np.random.geometric(n_phons_p)
    return sample_item(n_words=n_phons, word_delay_range=phon_delay_range,
                       surprisal_mean=surprisal_mean, surprisal_sigma=surprisal_sigma,
                       **kwargs)


def sample_item_with_phons(n_words=20, word_delay_range=(0.3, 1),
                           response_window=(0.0, 2),  # time window over which word triggers signal response
                           p_word_recognition = 1 / 3.,
                           recognition_irf=simple_peak,
                           irf=simple_peak, rate_irf=rate_irf,
                           sample_rate=128,
                           n400_surprisal_coef=-1,
                           word_surprisal_mean=2., word_surprisal_sigma=0.2,
                           **kwargs):
    """
    Sample an item combining phoneme-level surprisal with a distinct
    word recognition point.
    """
    
    acc_X_word, acc_X_phon, acc_y = [], None, None
    time_acc = 0
    
    # sample unitary response
    peak_xs = np.linspace(*response_window, num=int(response_window[1] * sample_rate), endpoint=False)
    peak_ys = n400_surprisal_coef * irf(peak_xs)
    
    # sample word rate response
    rate_xs = peak_xs.copy()
    rate_ys = rate_irf(rate_xs)
    
    for i in range(n_words):
        X, y = sample_word(response_window=response_window,
                           sample_rate=sample_rate,
                           irf=irf,
                           n400_surprisal_coef=n400_surprisal_coef,
                           **kwargs)
        X["word_idx"] = i
        X.index.name = "phon_idx"
        X = X.reset_index().set_index(["word_idx", "phon_idx"])
        y = y.set_index("time")
        
        # Sample a word recognition point.
        rec_point = min(len(X) - 1, np.random.geometric(p_word_recognition))
        rec_onset = X.iloc[rec_point].time
        rec_surprisal = np.random.lognormal(word_surprisal_mean, word_surprisal_sigma)
        
        # Produce word signal df
        rec_times = peak_xs + rec_onset
        rec_signal = peak_ys * rec_surprisal + rate_ys
        y_word = pd.DataFrame({"time": rec_times, "signal": rec_signal}) \
            .set_index("time")
        
        y = y.add(y_word, fill_value=0.0).reset_index()
        
        final_word_onset = X.time.max()
        X.time += time_acc
        y.time += time_acc
        X_word_row = (time_acc + rec_onset, rec_point, rec_surprisal)
        
        acc_X_word.append(X_word_row)
        acc_X_phon = X if acc_X_phon is None else pd.concat([acc_X_phon, X])

        # acc_y may have overlapping samples. merge-add them
        y = y.set_index("time")
        if acc_y is None:
            acc_y = y
        else:
            acc_y = acc_y.add(y, fill_value=0.0)
        
        delay = np.random.uniform(*word_delay_range)
        # Fit to sample phase
        delay = np.round(delay * sample_rate) / sample_rate
        time_acc += final_word_onset + delay
        
    acc_X_word = pd.DataFrame(acc_X_word, columns=["time", "recognition_point", "surprisal"])
    
    return acc_X_word, acc_X_phon, acc_y.reset_index()


def sample_dataset(size, n_word_range=(10, 25), **item_kwargs):
    ret_X, ret_y = [], []
    item_sizes = np.random.randint(*n_word_range, size=size)
    for size in item_sizes:
        X, y = sample_item(n_words=size, **item_kwargs)
        ret_X.append(X)
        ret_y.append(y)
        
    X = pd.concat(ret_X, names=["item", "word_idx"], keys=np.arange(len(ret_X)))
    y = pd.concat(ret_y, names=["item", "sample_idx"], keys=np.arange(len(ret_y)))
    return X, y


def sample_dataset_with_phons(size, n_word_range=(10, 25), **item_kwargs):
    ret_X_word, ret_X_phon, ret_y = [], [], []
    item_sizes = np.random.randint(*n_word_range, size=size)
    for size in tqdm(item_sizes):
        X_word, X_phon, y = sample_item_with_phons(n_words=size, **item_kwargs)
        ret_X_word.append(X_word)
        ret_X_phon.append(X_phon)
        ret_y.append(y)
        
    X_word = pd.concat(ret_X_word, names=["item", "word_idx"], keys=np.arange(len(ret_X_word)))
    X_phon = pd.concat(ret_X_phon, names=["item", "word_idx", "phon_idx"], keys=np.arange(len(ret_X_phon)))
    y = pd.concat(ret_y, names=["item", "sample_idx"], keys=np.arange(len(ret_y)))
    return X_word, X_phon, y


def dataset_to_epochs(X, y, epoch_window=(-0.1, 0.9), test_window=(0.3, 0.5)):
    assert X.index.names[0] == y.index.names[0]
    
    epoch_data = []
    epoch_left, epoch_right = epoch_window
    test_left, test_right = test_window
    
    for index, x in tqdm(X.iterrows(), total=len(X)):
        y_df = y.loc[index[0]]
        
        epoch_window = y_df[(y_df.time >= x.time + epoch_left) & (y_df.time <= x.time + epoch_right)]
        baseline_window = epoch_window[epoch_window.time <= x.time]
        test_window = epoch_window[(epoch_window.time >= x.time + test_left) & (epoch_window.time <= x.time + test_right)]

        # take means over temporal window
        baseline_window = baseline_window.signal.mean(axis=0)
        test_window = test_window.signal.mean(axis=0)

        if not isinstance(index, tuple):
            index = (index,)
        epoch_data.append(index + (baseline_window, test_window))
        
    epoch_df = pd.DataFrame(epoch_data, columns=X.index.names + ["baseline_N400", "value_N400"]) \
        .set_index(X.index.names)
    return epoch_df


def main(args):
    kwargs = dict(sample_rate=args.sample_rate,
                  n400_surprisal_coef=args.n400_coef)
    
    if args.with_phons:
        X_word, X_phon, y = sample_dataset_with_phons(
            args.n_items,
            word_surprisal_mean=args.surprisal_mean,
            word_surprisal_sigma=args.surprisal_sigma,
            **kwargs)
 
        X_word.to_csv(args.outdir / "X_word.txt", sep=" ")
        X_phon.to_csv(args.outdir / "X_phon.txt", sep=" ")
        y.to_csv(args.outdir / "y.txt", sep=" ")
    else:
        X, y = sample_dataset(args.n_items,
                              surprisal_mean=args.surprisal_mean,
                              surprisal_sigma=args.surprisal_sigma,
                              **kwargs)

        X.to_csv(args.outdir / "X.txt", sep=" ")
        y.to_csv(args.outdir / "y.txt", sep=" ")
    
    
if __name__ == "__main__":
    p = ArgumentParser()
    
    p.add_argument("outdir", type=Path)
    p.add_argument("-n", "--n_items", type=int, default=100)
    p.add_argument("--sample_rate", type=int, default=128)
    
    p.add_argument("--with_phons", default=False, action="store_true")
    
    # mean, sigma params for a log-normal surprisal distribution
    p.add_argument("--surprisal_mean", type=float, default=2)
    p.add_argument("--surprisal_sigma", type=float, default=0.2)
    
    # coefficient linking surprisal value to n400 amplitude
    p.add_argument("--n400_coef", type=float, default=-0.1)
    
    main(p.parse_args())