"""
Synthesizes an EEG dataset following the rough generative process of the
thresholded reindexing regression model.
"""

from argparse import ArgumentParser
import re
from typing import List, Tuple, NamedTuple

from icecream import ic
import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm.auto import tqdm
from typeguard import typechecked

import transformers
import torch
from torchtyping import TensorType

from berp.typing import is_probability, is_log_probability, \
                        ProperProbabilityDetail, ProperLogProbabilityDetail


# model_ref = "gpt2"
model_ref = "hf-internal-testing/tiny-xlm-roberta"
model = transformers.AutoModelForCausalLM.from_pretrained(model_ref, is_decoder=True)
model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_ref)
tokenizer.pad_token = tokenizer.eos_token

# dumb phoneme=letter
phonemes = list("abcdefghijklmnopqrstuvwxyz_")
phoneme2idx = {p: idx for idx, p in enumerate(phonemes)}
phoneme_confusion = torch.tensor(np.diag(np.ones(len(phonemes)))) + \
    0.25 * torch.rand(len(phonemes), len(phonemes))
phoneme_confusion /= phoneme_confusion.sum(axis=1, keepdim=True)


def simple_peak(x, scale=5, b=0.05):
    """Function which rapidly peaks and returns to baseline"""
    ret = st.gamma.pdf(x * scale, 2.8, b)
    ret /= ret.max()
    return torch.tensor(ret)


def rate_irf(x):
    return 0.1 * simple_peak(x, scale=20, b=0)


def clean_word_str(word):
    return re.sub(r"[^a-z]", "", word.lower())


# Type variables
TT = TensorType
N_W = "n_words"
N_C = "n_candidates"
N_P = "n_phonemes"
V = "vocab"


@typechecked
def compute_candidate_phoneme_likelihoods(
    word: str, word_id: torch.LongTensor,
    p_word_prior: TT[V, is_log_probability],
    n_candidates=10
    ) -> Tuple[TT[N_C, int], List[str],
                TT[N_C, N_P, torch.int64],
                TT[N_C, N_P, is_log_probability]]:

    word = clean_word_str(word)

    # Draw small set of candidate alternate words
    # NB drawing more than `n_candidates` because some will be filtered out.
    # Ideally wouldn't have this magic-number setup.
    candidate_ids = p_word_prior.argsort(axis=0, descending=True)[:n_candidates * 4]
    candidate_ids[0] = word_id
    # if gt_word_id not in candidate_ids:
    #     candidate_ids[-1] = gt_word_id
    gt_word_candidate_pos = 0  # (candidate_ids == gt_word_id).int().argmax()
    # Clean candidate tokens, prep for phoneme processing
    candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_ids)
    candidate_pairs = [(i, clean_word_str(tok)) for i, tok in zip(candidate_ids, candidate_tokens)]

    candidate_pairs = [(i, tok) for i, tok in candidate_pairs if tok][:n_candidates]
    candidate_ids = torch.tensor([i for i, _ in candidate_pairs])
    candidate_tokens = [tok for _, tok in candidate_pairs]
    max_tok_length = max(len(tok) for tok in candidate_tokens)
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
def compute_recognition_point(candidate_ids: TT[N_C, torch.long],
                              p_word_prior: TT[V, is_log_probability],
                              candidate_phoneme_likelihoods: TT[N_C, N_P, is_log_probability],
                              threshold: TT[is_probability],
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
    recognition_point = (bayes_p_word[gt_word_candidate_pos, :] >= threshold).int().argmax()

    return recognition_point


class WordObservation(NamedTuple):
    cleaned_word: str
    candidate_tokens: List[str]
    candidate_ids: List[int]
    candidate_phonemes: TensorType[N_C, N_P, torch.int64]

    recognition_point: int

    X_phon: pd.DataFrame
    y: pd.DataFrame


@typechecked
def sample_word(word: str, word_id: torch.LongTensor,
                p_word_prior: TT[V, is_log_probability],
                phon_delay_range=(0.1, 0.35),
                response_window=(0.0, 2),
                sample_rate=128,
                n400_surprisal_coef=-1,
                irf=simple_peak, rate_irf=rate_irf,
                n_candidates=10,
                recognition_threshold=torch.tensor(0.2)
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

    # sample unitary response
    peak_xs = torch.tensor(np.linspace(*response_window, num=int(response_window[1] * sample_rate), endpoint=False))
    peak_ys = n400_surprisal_coef * irf(peak_xs)

    # sample word rate response
    rate_xs = peak_xs[:]
    rate_ys = rate_irf(rate_xs)

    # sample stimulus times+surprisals
    stim_delays = np.random.uniform(*phon_delay_range, size=n_phons)
    stim_onsets = np.cumsum(stim_delays)
    # hack: align times to sample rate to make this easier
    stim_onsets = np.round(stim_onsets * sample_rate) / sample_rate

    max_time = stim_onsets.max() + response_window[1]
    all_times = torch.tensor(np.linspace(0, max_time, num=int(np.ceil(max_time * sample_rate)), endpoint=False))

    signal = torch.zeros_like(all_times)
    for onset, surprisal in zip(stim_onsets, phoneme_surprisals):
        sample_idx = int(onset * sample_rate)  # guaranteed to be round because of hack.
        signal[sample_idx:sample_idx + len(peak_xs)] += peak_ys * surprisal + rate_ys

    # build return X, y dataframes.
    X = pd.DataFrame({"time": stim_onsets, "phoneme": list(cleaned_word),
                      "surprisal": phoneme_surprisals})
    y = pd.DataFrame({"time": all_times, "signal": signal})

    return WordObservation(
        cleaned_word=cleaned_word,
        candidate_tokens=candidate_tokens,
        candidate_ids=candidate_ids.tolist(),
        candidate_phonemes=candidate_phonemes,

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
    candidate_phonemes: TensorType[N_W, N_C, N_P, torch.int64]
    p_word: TT[N_W, N_C, is_log_probability]


@typechecked
def sample_item(sentence: str,
                word_delay_range=(0.3, 1),
                response_window=(0.0, 2),  # time window over which word triggers signal response
                recognition_irf=simple_peak,
                irf=simple_peak, rate_irf=rate_irf,
                sample_rate=128,
                n400_surprisal_coef=-1,
                recognition_threshold=torch.tensor(0.2),
                **kwargs
                ) -> ItemObservation:
    """
    Sample an item combining phoneme-level surprisal with a distinct
    word recognition point.
    """
    sentence = re.sub(r"[^a-z\s]", "", sentence.lower()).strip()
    print(sentence)

    acc_X_word, acc_X_phon, acc_y = [], None, None
    acc_candidate_ids, acc_candidate_tokens, acc_candidate_phonemes, acc_p_word = [], [], [], []
    time_acc = 0

    # sample unitary response
    peak_xs = np.linspace(*response_window, num=int(response_window[1] * sample_rate), endpoint=False)
    peak_ys = n400_surprisal_coef * irf(peak_xs)

    # sample word rate response
    rate_xs = peak_xs.copy()
    rate_ys = rate_irf(rate_xs)

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
            response_window=response_window,
            sample_rate=sample_rate,
            irf=irf,
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
        rec_times = peak_xs + rec_onset
        rec_signal = peak_ys * rec_surprisal + rate_ys
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

        # store renormalized prior over top-k candidate words
        p_word_prior = p_word_prior[word_obs.candidate_ids].exp()
        p_word_prior /= p_word_prior.sum()
        acc_p_word.append(p_word_prior.log())

        delay = np.random.uniform(*word_delay_range)
        # Fit to sample phase
        delay = np.round(delay * sample_rate) / sample_rate
        time_acc += final_word_onset + delay

    acc_X_word = pd.DataFrame(acc_X_word, columns=["token", "time", "recognition_point", "surprisal"])
    acc_X_word.index.name = "token_idx"

    acc_candidate_ids = torch.tensor(acc_candidate_ids)

    # Pad candidate phoneme sequences.
    ic(acc_candidate_phonemes)
    acc_candidate_phonemes = torch.nn.utils.rnn.pad_sequence(
        acc_candidate_phonemes, batch_first=True,
        padding_value=phoneme2idx["_"])

    acc_p_word = torch.stack(acc_p_word)
    return ItemObservation(
        acc_X_word, acc_X_phon, acc_y.reset_index(),
        acc_candidate_ids, acc_candidate_tokens, acc_candidate_phonemes,
        acc_p_word)


class RRDataset(NamedTuple):
    X_word: pd.DataFrame
    X_phon: pd.DataFrame
    y: pd.DataFrame

    candidate_ids: List[TensorType[N_W, N_C, torch.int64]]
    candidate_tokens: List[List[str]]
    candidate_phonemes: List[TensorType[N_W, N_C, N_P, torch.int64]]
    p_word: List[TensorType[N_W, N_C, is_log_probability]]


def sample_dataset(sentences: List[str], **item_kwargs) -> RRDataset:
    ret_X_word, ret_X_phon, ret_y = [], [], []
    ret_candidate_tokens, ret_candidate_ids, ret_candidate_phonemes, ret_p_word = [], [], [], []
    for sentence in tqdm(sentences):
        item = sample_item(sentence, **item_kwargs)
        ret_X_word.append(item.X_word)
        ret_X_phon.append(item.X_phon)
        ret_y.append(item.y)

        ret_candidate_tokens.append(item.candidate_tokens)
        ret_candidate_ids.append(item.candidate_ids)
        ret_candidate_phonemes.append(item.candidate_phonemes)
        ret_p_word.append(item.p_word)

    X_word = pd.concat(ret_X_word, names=["item", "token_idx"], keys=np.arange(len(ret_X_word)))
    X_phon = pd.concat(ret_X_phon, names=["item", "token_idx", "phon_idx"], keys=np.arange(len(ret_X_phon)))
    y = pd.concat(ret_y, names=["item", "sample_idx"], keys=np.arange(len(ret_y)))
    return RRDataset(
        X_word, X_phon, y,
        ret_candidate_ids, ret_candidate_tokens, ret_candidate_phonemes, ret_p_word)


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
    sentences = ["this is a test sentence"]

    for sentence in sentences:
        X_word, X_phon, y, p_word = sample_item(sentence)
        print(X_word, X_phon, y)


if __name__ == "__main__":
    p = ArgumentParser()

    # TODO

    main(p.parse_args())
