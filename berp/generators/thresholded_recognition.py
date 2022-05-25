"""
Synthesizes an EEG dataset following the rough generative process of the
thresholded reindexing regression model.
"""

from argparse import ArgumentParser
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm.auto import tqdm


def simple_peak(x, scale=5, b=0.05):
    """Function which rapidly peaks and returns to baseline"""
    ret = st.gamma.pdf(x * scale, 2.8, b)
    ret /= ret.max()
    return ret


def rate_irf(x):
    return 0.1 * simple_peak(x, scale=20, b=0)


def sample_word(word: str, phon_delay_range=(0.1, 0.35),
                response_window=(0.0, 2),
                sample_rate=128,
                n400_surprisal_coef=-1,
                irf=simple_peak, rate_irf=rate_irf):
    """
    For the given word, sample a phoneme onset trajectory and compute
    corresponding phoneme surprisals.
    """
    # sample unitary response
    peak_xs = np.linspace(*response_window, num=int(response_window[1] * sample_rate), endpoint=False)
    peak_ys = n400_surprisal_coef * irf(peak_xs)

    # sample word rate response
    rate_xs = peak_xs.copy()
    rate_ys = rate_irf(rate_xs)

    # sample stimulus times+surprisals
    n_phons = len(word)
    stim_delays = np.random.uniform(*phon_delay_range, size=n_phons)
    stim_onsets = np.cumsum(stim_delays)
    # hack: align times to sample rate to make this easier
    stim_onsets = np.round(stim_onsets * sample_rate) / sample_rate
    # TODO shouldn't be random
    surprisal_mean = 2.
    surprisal_sigma = 0.2
    surprisals = np.random.lognormal(surprisal_mean, surprisal_sigma, size=n_phons)

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


def sample_item(words: List[str],
                word_surprisals: List[float],
                word_delay_range=(0.3, 1),
                response_window=(0.0, 2),  # time window over which word triggers signal response
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

    for i, word in enumerate(words):
        X, y = sample_word(word,
                           response_window=response_window,
                           sample_rate=sample_rate,
                           irf=irf,
                           n400_surprisal_coef=n400_surprisal_coef,
                           **kwargs)
        X["word_idx"] = i
        X.index.name = "phon_idx"
        X = X.reset_index().set_index(["word_idx", "phon_idx"])
        y = y.set_index("time")

        # Sample a word recognition point.
        # TODO should not be random; do thresholded
        p_word_recognition = 0.9
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


def main(args):
    sentences = ["this is a test sentence".split()]
    surprisals = [np.random.lognormal(2, 0.2, size=len(sentences[0]))]

    for sentence, surprisal_seq in zip(sentences, surprisals):
        X_word, X_phon, y = sample_item(sentence, surprisal_seq)
        print(X_word, X_phon, y)


if __name__ == "__main__":
    p = ArgumentParser()

    # TODO

    main(p.parse_args())
