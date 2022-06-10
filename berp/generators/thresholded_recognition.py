"""
Synthesizes an EEG dataset following the rough generative process of the
thresholded reindexing regression model.
"""

from argparse import ArgumentParser
import re
from typing import List, Tuple, NamedTuple, Optional, cast

from icecream import ic
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats as st
from tqdm.auto import tqdm
from typeguard import typechecked

import transformers
import torch
from torch.nn.functional import pad
from torchtyping import TensorType

from berp.models.reindexing_regression import ModelParameters
from berp.typing import is_probability, is_log_probability, \
                        ProperProbabilityDetail, ProperLogProbabilityDetail, \
                        DIMS

# This module is extremely messy in its handling of tensors vs. lists. We used
# tensors originally to get nice easy shape/dimension checking with type hints,
# but it became a hassle to force everything into tensors early (due to variable
# length content) and we abandoned this later. But the whole thing should be
# refactored to be consistent if this is going to be anything but throwaway
# research code.


model_ref = "gpt2"
# model_ref = "hf-internal-testing/tiny-xlm-roberta"
model = transformers.AutoModelForCausalLM.from_pretrained(model_ref, is_decoder=True)  # type: ignore
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_ref)  # type: ignore
tokenizer.pad_token = tokenizer.eos_token

# dumb phoneme=letter
phonemes = list("abcdefghijklmnopqrstuvwxyz_")
phoneme2idx = {p: idx for idx, p in enumerate(phonemes)}
phoneme_confusion = torch.diag(torch.ones(len(phonemes))) + \
    0.1 * torch.rand(len(phonemes), len(phonemes))
phoneme_confusion /= phoneme_confusion.sum(dim=0, keepdim=True)



def gaussian_window(center: float, width: float,\
                    start: float = 0,
                    end: float = 1,
                    sample_rate=128):
    """Gaussian window :class:`NDVar`
    Parameters
    ----------
    center : scalar
        Center of the window (normalized to the closest sample on ``time``).
    width : scalar
        Standard deviation of the window.
    time : UTS
        Time dimension.
    Returns
    -------
    gaussian : NDVar
        Gaussian window on ``time``.
    """

    n_samples = int((end - start) * sample_rate) + 1
    times, step_size = np.linspace(start, end, n_samples, retstep=True, endpoint=True)
    width_i = int(round(width / step_size))
    n_times = len(times)
    center_i = (center - start) // step_size

    slice_start, slice_stop = None, None
    if center_i >= n_times / 2:
        slice_start = None
        slice_stop = n_times
        window_width = 2 * center_i + 1
    else:
        slice_start = -n_times
        slice_stop = None
        window_width = 2 * (n_times - center_i) - 1
    window_data = scipy.signal.windows.gaussian(window_width, width_i)
    window_data = window_data[slice_start: slice_stop]
    return times, window_data


def simulate_erp(event_probability, sample_rate=128, rng=np.random) -> Tuple[np.ndarray, np.ndarray]:
    # Stolen from https://github.com/christianbrodbeck/Eelbrain/blob/master/eelbrain/datasets/_sim_eeg.py

    # Generate topography
    n400_topo = -1.0  # * _topo(sensor, 'Cz')
    # Generate timing
    times, n400_timecourse = gaussian_window(0.400, 0.034)
    # Put all the dimensions together to simulate the EEG signal
    signal = event_probability * n400_timecourse * n400_topo

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

    # Data scale
    signal *= 1e-6

    return times, signal


def simulate_phoneme_sequence(phoneme_surprisals: torch.Tensor,
                              phon_delay_range: Tuple[float, float] = (0.04, 0.1),
                              sample_rate: int = 128,
                              n400_surprisal_coef: float = -0.25,
                              ) -> Tuple[np.ndarray, TensorType, TensorType]:
    """
    Simulate phoneme temporal sequence and resulting ERPs.
    """

    response_window = (0., 1.)

    # sample stimulus times+surprisals
    n_phons = len(phoneme_surprisals)
    stim_delays = np.random.uniform(*phon_delay_range, size=n_phons)
    stim_onsets = np.cumsum(stim_delays)
    # hack: align times to sample rate to make this easier
    stim_onsets = np.round(stim_onsets * sample_rate) / sample_rate

    max_time = stim_onsets.max() + response_window[1] + 1 / sample_rate  # DEV wut
    all_times = torch.tensor(np.linspace(0, max_time, num=int(np.ceil(max_time * sample_rate)), endpoint=False))

    signal = torch.zeros_like(all_times)
    for onset, surprisal in zip(stim_onsets, phoneme_surprisals):
        sample_idx = int(onset * sample_rate)  # guaranteed to be round because of hack.

        _, phoneme_response_nd = gaussian_window(0.300, 0.03, *response_window)
        phoneme_response = torch.tensor(phoneme_response_nd) * n400_surprisal_coef * surprisal
        signal[sample_idx:sample_idx + len(phoneme_response)] += phoneme_response

    signal *= 1e-6

    return stim_onsets, all_times, signal


def clean_word_str(word):
    return re.sub(r"[^a-z]", "", word.lower())


# Type variables
B, N_W, N_C, N_F, N_P, V_W = DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_P, DIMS.V_W


def _tensor_index(t: torch.Tensor, val: torch.Tensor) -> Optional[int]:
    # value is Tensor([]) when not found, which has shape [0];
    # value is Tensor(k) when found, shape []
    # makes no sense, but let's just follow the logic
    matches = (t == val).nonzero().squeeze()
    if matches.shape:
        return None
    else:
        return cast(int, matches.item())


@typechecked
def compute_candidate_phoneme_likelihoods(
    word: str, word_id: torch.LongTensor,
    p_word_prior: TensorType[V_W, is_log_probability],
    n_candidates=10
    ) -> Tuple[TensorType[N_C, int], List[str],
                TensorType[N_C, N_P, torch.int64],
                TensorType[N_C, N_P, is_log_probability]]:

    word = clean_word_str(word)

    # Draw small set of candidate alternate words
    # NB drawing more than `n_candidates` because some will be filtered out.
    # Ideally wouldn't have this magic-number setup.
    candidate_ids = p_word_prior.argsort(dim=0, descending=True)[:n_candidates * 4]

    # If ground truth word is already in candidates, swap into idx 0. Otherwise
    # replace the most probable item at idx 0.
    gt_idx = _tensor_index(candidate_ids, word_id)
    gt_word_candidate_pos = 0
    if gt_idx is not None:
        idxs = torch.arange(len(candidate_ids))
        other_candidate = idxs[gt_word_candidate_pos]
        idxs[gt_word_candidate_pos] = gt_idx
        idxs[gt_idx] = other_candidate
        candidate_ids = candidate_ids[idxs]
    else:
        candidate_ids[gt_word_candidate_pos] = word_id

    # Clean candidate tokens, prep for phoneme processing
    candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_ids)
    candidate_pairs = [(i, clean_word_str(tok)) for i, tok in zip(candidate_ids, candidate_tokens)]

    candidate_pairs = [(i, tok) for i, tok in candidate_pairs if tok][:n_candidates]
    candidate_ids = torch.tensor([i for i, _ in candidate_pairs])
    candidate_tokens = [tok for _, tok in candidate_pairs]

    # NB we pad ONE MORE than the maximum token length, since the final padding
    # element is meaningful for the model -- indicates recognition at/after
    # final phoneme offset.
    max_tok_length = max(len(tok) for tok in candidate_tokens) + 1
    candidate_tokens = [tok + ("_" * (max_tok_length - len(tok)))
                        for tok in candidate_tokens]

    # Now convert to padded phoneme seqs
    candidate_phoneme_seqs = torch.tensor([[phoneme2idx[p] for p in tok]
                                           for tok in candidate_tokens])
    gt_phoneme_seq = candidate_phoneme_seqs[gt_word_candidate_pos]

    candidate_phoneme_likelihoods = \
        phoneme_confusion[candidate_phoneme_seqs, gt_phoneme_seq].log()
    return (candidate_ids, candidate_tokens, candidate_phoneme_seqs,
            candidate_phoneme_likelihoods)


@typechecked
def compute_recognition_point(candidate_ids: TensorType[N_C, torch.long],
                              p_word_prior: TensorType[V_W, is_log_probability],
                              candidate_phoneme_likelihoods: TensorType[N_C, N_P, is_log_probability],
                              threshold: TensorType[is_probability],
                              gt_word_candidate_pos=0
                              ) -> torch.LongTensor:
    # Get likelihood of each phoneme subsequence under GT word
    incremental_phoneme_likelihoods = candidate_phoneme_likelihoods.cumsum(axis=1)

    # Combine with priors (renormalized over candidate space) and normalize.
    p_word_prior = p_word_prior[candidate_ids].exp()
    p_word_prior /= p_word_prior.sum()
    bayes_p_word = (p_word_prior.unsqueeze(-1).log() + incremental_phoneme_likelihoods).exp()
    bayes_p_word /= bayes_p_word.sum(axis=0, keepdim=True)

    # Compute recognition point.
    recognition_point: torch.LongTensor = \
        (bayes_p_word[gt_word_candidate_pos, :] >= threshold).int().argmax()

    return recognition_point


class WordObservation(NamedTuple):
    cleaned_word: str
    candidate_tokens: List[str]
    candidate_ids: List[int]
    candidate_phonemes: TensorType[N_C, N_P, torch.int64]
    phoneme_onsets: List[float]

    recognition_point: int

    X_phon: pd.DataFrame
    y: pd.DataFrame


@typechecked
def sample_word(word: str, word_id: torch.LongTensor,
                p_word_prior: TensorType[V_W, is_log_probability],
                sample_rate=128,
                n_candidates=10,
                recognition_threshold=torch.tensor(0.2),
                **phoneme_seq_kwargs
                ) -> WordObservation:
    """
    For the given word, sample a phoneme onset trajectory and compute
    corresponding phoneme surprisals.
    """
    candidate_ids, candidate_tokens, candidate_phonemes, candidate_phoneme_likelihoods = \
        compute_candidate_phoneme_likelihoods(word, word_id, p_word_prior,
                                              n_candidates=n_candidates)
    # print(word, [tok.rstrip("_") for tok in candidate_tokens])
    recognition_point = compute_recognition_point(candidate_ids,
                                                  p_word_prior,
                                                  candidate_phoneme_likelihoods,
                                                  threshold=recognition_threshold)

    # Phoneme surprisal (forward model)
    gt_word_candidate_pos = 0
    cleaned_word = candidate_tokens[gt_word_candidate_pos].rstrip("_")
    n_phons = len(cleaned_word)
    phoneme_surprisals = p_word_prior[word_id] + \
        candidate_phoneme_likelihoods[gt_word_candidate_pos, :]
    phoneme_surprisals = -phoneme_surprisals / np.log(2)
    phoneme_surprisals = phoneme_surprisals[:n_phons]

    stim_onsets, times, signal = simulate_phoneme_sequence(phoneme_surprisals, **phoneme_seq_kwargs)

    # build return X, y dataframes.
    X = pd.DataFrame({"time": stim_onsets, "phoneme": list(cleaned_word),
                      "surprisal": phoneme_surprisals})
    y = pd.DataFrame({"time": times, "signal": signal})

    return WordObservation(
        cleaned_word=cleaned_word,
        candidate_tokens=candidate_tokens,
        candidate_ids=candidate_ids.tolist(),
        candidate_phonemes=candidate_phonemes,
        phoneme_onsets=stim_onsets.tolist(),

        recognition_point=int(recognition_point),

        X_phon=X,
        y=y,
    )


class ItemObservation(NamedTuple):
    X_word: pd.DataFrame
    X_phon: pd.DataFrame
    y: pd.DataFrame

    candidate_ids: TensorType[N_W, N_C, torch.int64]
    candidate_tokens: List[List[str]]
    # DEV N_P will vary in between calls .. oops
    candidate_phonemes: TensorType[N_W, N_C, N_P, torch.int64]
    word_lengths: List[int]
    phoneme_onsets: TensorType[N_W, N_P, float]
    p_word: TensorType[N_W, N_C, is_log_probability]


def sample_item(sentence: str,
                word_delay_range=(0.1, 0.25),
                sample_rate=128,
                n400_surprisal_coef=-0.1,
                recognition_threshold=torch.tensor(0.2),
                **kwargs
                ) -> ItemObservation:
    """
    Sample an item combining phoneme-level surprisal with a distinct
    word recognition point.
    """
    sentence = re.sub(r"[^a-z\s]", "", sentence.lower()).strip()
    print(sentence)

    acc_X_word, acc_X_phon, acc_y = [], pd.DataFrame(), pd.DataFrame()
    acc_candidate_ids, acc_candidate_tokens, acc_candidate_phonemes = [], [], []
    acc_phoneme_onsets, acc_p_word = [], []

    time_acc = 0

    # Compute word-level predictive distributions
    tokenized = tokenizer(sentence, return_tensors="pt",
                          padding=True, truncation=True,
                          max_length=model.config.max_length)
    with torch.no_grad():
        model_outputs = model(tokenized["input_ids"])[0].log_softmax(dim=2)
    model_outputs = model_outputs.squeeze(0)
    # Draw tokens and corresponding predictive distributions
    gt_tokens = tokenized.tokens(0)[1:]
    gt_token_ids = tokenized["input_ids"][0, 1:]
    predictive_dists = model_outputs[torch.arange(gt_token_ids.shape[0])]

    for i, (word, word_id, p_word_prior) in enumerate(zip(gt_tokens,
                                                          gt_token_ids,
                                                          predictive_dists)):
        word_obs = sample_word(
            word, word_id, p_word_prior,
            sample_rate=sample_rate,
            n400_surprisal_coef=n400_surprisal_coef,
            recognition_threshold=recognition_threshold,
            **kwargs)

        X = word_obs.X_phon
        X["token_idx"] = i
        X.index.name = "phon_idx"
        X = X.reset_index().set_index(["token_idx", "phon_idx"])
        y = word_obs.y
        y = y.set_index("time")

        if word_obs.recognition_point >= len(word_obs.cleaned_word):
            # HACK just place at made-up word offset
            rec_onset = X.iloc[-1].time + 3 / sample_rate
        else:
            rec_onset = X.iloc[word_obs.recognition_point].time
        rec_surprisal = float(-p_word_prior[word_id] / np.log(2))

        # Produce word signal df
        rec_times, rec_signal = simulate_erp(rec_surprisal,
                                             sample_rate=sample_rate)
        rec_times += rec_onset
        # rec_times = peak_xs + rec_onset
        # rec_signal = peak_ys * rec_surprisal + rate_ys
        y_word = pd.DataFrame({"time": rec_times, "signal": rec_signal}) \
            .set_index("time")

        y = y.add(y_word, fill_value=0.0).reset_index()

        final_word_onset = X.time.max()
        X.time += time_acc
        y.time += time_acc
        X_word_row = (word_obs.cleaned_word, time_acc + rec_onset,
                      word_obs.recognition_point, rec_surprisal)

        acc_X_word.append(X_word_row)
        acc_X_phon = X if acc_X_phon is None else pd.concat([acc_X_phon, X])

        # acc_y may have overlapping samples. merge-add them
        y = y.set_index("time")
        if acc_y is None:
            acc_y = y
        else:
            acc_y = acc_y.add(y, fill_value=0.0)

        acc_candidate_ids.append(word_obs.candidate_ids)
        acc_candidate_tokens.append(word_obs.candidate_tokens)
        acc_candidate_phonemes.append(word_obs.candidate_phonemes.unbind())

        acc_phoneme_onsets.append(word_obs.phoneme_onsets)

        # store renormalized prior over top-k candidate words
        p_word_prior = p_word_prior[word_obs.candidate_ids].exp()
        p_word_prior /= p_word_prior.sum()
        acc_p_word.append(p_word_prior.log())

        delay = np.random.uniform(*word_delay_range)
        # Fit to sample phase
        delay = np.round(delay * sample_rate) / sample_rate
        time_acc += final_word_onset + delay

    acc_X_word = pd.DataFrame(acc_X_word, columns=["token", "time", "recognition_point", "surprisal"]) \
        .rename_axis("token_idx")

    # Pad candidate phoneme sequences.
    max_length = max(len(candidate_phonemes[0]) for candidate_phonemes
                     in acc_candidate_phonemes)
    ret_candidate_phonemes = torch.stack([
        pad(torch.stack(candidate_phonemes),
            (0, max_length - len(candidate_phonemes[0])),
            value=phoneme2idx["_"])
        for candidate_phonemes in acc_candidate_phonemes
    ])

    word_lengths = [len(onsets) for onsets in acc_phoneme_onsets]

    ret_phoneme_onsets = torch.nn.utils.rnn.pad_sequence(
        list(map(torch.tensor, acc_phoneme_onsets)), batch_first=True,
        padding_value=0.)

    return ItemObservation(
        acc_X_word, acc_X_phon, acc_y.reset_index(),
        torch.tensor(acc_candidate_ids),
        acc_candidate_tokens, ret_candidate_phonemes,
        word_lengths, ret_phoneme_onsets,
        torch.stack(acc_p_word))


class RRDataset(NamedTuple):
    params: ModelParameters

    X_word: pd.DataFrame
    X_phon: pd.DataFrame
    y: pd.DataFrame

    candidate_ids: List[TensorType[N_W, N_C, torch.int64]]
    candidate_tokens: List[List[List[str]]]
    candidate_phonemes: List[TensorType[N_W, N_C, N_P, torch.int64]]
    word_lengths: List[List[int]]
    phoneme_onsets: List[TensorType[N_W, N_P, float]]
    p_word: List[TensorType[N_W, N_C, is_log_probability]]

    sample_rate: int


def sample_dataset(sentences: List[str],
                   params: Optional[ModelParameters] = None,
                   sample_rate=128,
                   **item_kwargs) -> RRDataset:
    if params is None:
        params = ModelParameters(
            lambda_=torch.tensor(1.),
            confusion=phoneme_confusion,
            threshold=torch.tensor(0.7),

            # TODO we don't actually control generative process to determine
            # these response parameters. Best guesses at ideal model.
            a=torch.tensor(0.4),
            b=torch.tensor(0.2),
            coef=torch.tensor([1., -1.])
        )

    ret_X_word, ret_X_phon, ret_y = [], [], []
    ret_candidate_tokens, ret_candidate_ids, ret_candidate_phonemes = [], [], []
    ret_word_lengths, ret_phoneme_onsets, ret_p_word = [], [], []
    for sentence in tqdm(sentences):
        item = sample_item(sentence,
                           n400_surprisal_coef=params.coef[1],
                           recognition_threshold=params.threshold,
                           sample_rate=sample_rate,
                           **item_kwargs)
        ret_X_word.append(item.X_word)
        ret_X_phon.append(item.X_phon)
        ret_y.append(item.y)

        ret_candidate_tokens.append(item.candidate_tokens)
        ret_candidate_ids.append(item.candidate_ids)
        ret_candidate_phonemes.append(item.candidate_phonemes)
        ret_word_lengths.append(item.word_lengths)
        ret_phoneme_onsets.append(item.phoneme_onsets)
        ret_p_word.append(item.p_word)

    X_word = pd.concat(ret_X_word, names=["item", "token_idx"], keys=np.arange(len(ret_X_word)))
    X_phon = pd.concat(ret_X_phon, names=["item", "token_idx", "phon_idx"], keys=np.arange(len(ret_X_phon)))
    y = pd.concat(ret_y, names=["item", "sample_idx"], keys=np.arange(len(ret_y)))
    return RRDataset(
        params,
        X_word, X_phon, y,
        ret_candidate_ids, ret_candidate_tokens,
        ret_candidate_phonemes, ret_word_lengths, ret_phoneme_onsets,
        ret_p_word,
        sample_rate=sample_rate)


def dataset_to_epochs(X, y, epoch_window=(-0.1, 0.9)):
    """
    Resample dataset as epochs. Unlike other `dataset_to_epochs` impls in this
    codebase, this simply returns the epoch data without averaging.
    """

    assert X.index.names[0] == y.index.names[0]

    epoch_data = {}
    epoch_left, epoch_right = epoch_window
    for index, x in tqdm(X.iterrows(), total=len(X)):
        y_df = y.loc[index[0]]

        epoch_window = y_df[(y_df.time >= x.time + epoch_left) & (y_df.time <= x.time + epoch_right)] \
            .copy()
        epoch_window["epoch_time"] = epoch_window.time - x.time
        epoch_data[index] = epoch_window

    epoch_df = pd.concat(epoch_data, names=tuple(X.index.names) + tuple(y.index.names[1:]))
    return epoch_df


def main(args):
    sentences = ["this is a test sentence"]

    for sentence in sentences:
        obs = sample_item(sentence)
        print(obs)


if __name__ == "__main__":
    p = ArgumentParser()

    # TODO

    main(p.parse_args())
