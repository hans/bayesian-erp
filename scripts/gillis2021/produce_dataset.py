# +
from argparse import ArgumentParser, Namespace
from collections import defaultdict, Counter
import itertools
import logging
from pathlib import Path
import pickle
import re
import sys
sys.path.append(str(Path(".").resolve().parent.parent))

from colorama import Fore, Style
import h5py
import mne
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import pad
from tqdm.auto import tqdm
import transformers
from typeguard import check_return_type
# -

IS_INTERACTIVE = False
try:
    get_ipython()  # type: ignore
except NameError: pass
else: IS_INTERACTIVE = True
IS_INTERACTIVE

EEG_SUFFIX = "_1_256_8_average_4_128"

# %load_ext autoreload
# %autoreload 2

from berp.datasets import BerpDataset, NaturalLanguageStimulus
from berp.datasets import NaturalLanguageStimulusProcessor
from berp.util import time_to_sample

p = ArgumentParser()
p.add_argument("natural_language_stimulus_path", type=Path)
p.add_argument("aligned_words_path", type=Path)
p.add_argument("aligned_phonemes_path", type=Path)
p.add_argument("eeg_path", type=Path)
p.add_argument("stim_path", type=Path)
p.add_argument("-o", "--output_path", type=Path, required=True)

if IS_INTERACTIVE:
    args = Namespace(natural_language_stimulus_path=Path("DKZ_1.pkl"),
                     aligned_words_path=Path("DKZ_1.words.csv"),
                     aligned_phonemes_path=Path("DKZ_1.phonemes.csv"),
                     eeg_path=Path("../../data/gillis2021/eeg/DKZ_1/2019_C2DNN_1_1_256_8_average_4_128.mat"),
                     stim_path=Path("stimuli.npz"),
                     output_path=Path("DKZ_1.2019_C2DNN_1.pkl"),)
else:
    args = p.parse_args()

subject = args.eeg_path.stem[:args.eeg_path.stem.index(EEG_SUFFIX)]
story_name = args.natural_language_stimulus_path.stem

# ## Load and process natural language stimulus and time series features

with args.natural_language_stimulus_path.open("rb") as f:
    story_stim: NaturalLanguageStimulus = pickle.load(f)
assert story_stim.name == story_name
ts_features_dict = np.load(args.stim_path)
ts_feature_names = ts_features_dict["feature_names"].tolist()
time_series_features = ts_features_dict[story_name]

# Variable onset features are simply a variable onset intercept,
# word features and word surprisals.
X_variable = torch.concat(
    [torch.ones_like(story_stim.word_surprisals).unsqueeze(1),
     story_stim.word_features,
     story_stim.word_surprisals.unsqueeze(1)],
    dim=1)
# NB word_frequency comes from stimulus processor setup in previous script
variable_feature_names = ["recognition_onset"] + story_stim.word_feature_names + ["word_surprisal"]
assert X_variable.shape[1] == len(variable_feature_names)

# Load other stimulus time-series features.
X_ts = torch.tensor(time_series_features)

assert sorted(set(variable_feature_names)) == sorted(variable_feature_names), \
    "Variable feature names must be unique"
assert sorted(set(ts_feature_names)) == sorted(ts_feature_names), \
    "Time series feature names must be unique"

# ## Load aligned word/phoneme data

# +
words_df = pd.read_csv(args.aligned_words_path)
phonemes_df = pd.read_csv(args.aligned_phonemes_path)

words_df = words_df[words_df.story == story_name]
phonemes_df = phonemes_df[phonemes_df.story == story_name]

assert len(words_df) > 0
assert len(phonemes_df) > 0


# -

# ## Load and process EEG data

def load_eeg(path, info: mne.Info, trim_n_samples=None):
    """
    Load EEG data from the given path.
    
    Returns:
        an mne.io.RawArray
    """
    
    fid = h5py.File(path, "r")
    
    sample_rate = fid["fs"][0, 0]
    assert sample_rate == info["sfreq"]
    
    raw_eeg = fid[fid["epochs"][0, 0]][:64, :]
    
    if trim_n_samples is not None:
        # Trim to match other data source.
        raw_eeg = raw_eeg[:, :trim_n_samples]
    
    return mne.io.RawArray(raw_eeg, info, verbose=False)


montage = mne.channels.make_standard_montage("biosemi64")
info = mne.create_info(ch_names=montage.ch_names,
                       sfreq=128, ch_types="eeg").set_montage(montage)


# Load EEG and trim to match time series features.
eeg = load_eeg(args.eeg_path, info, trim_n_samples=X_ts.shape[0])

# Retrieve word boundary information.
word_onsets = words_df.groupby("original_idx").start.min().to_dict()
word_onsets = torch.tensor([word_onsets[word_id.item()]
                            for word_id in story_stim.word_ids])
word_offsets = words_df.groupby("original_idx").end.max().to_dict()
word_offsets = torch.tensor([word_offsets[word_id.item()]
                             for word_id in story_stim.word_ids])

# +
# Phoneme onsets.
phoneme_onsets = phonemes_df.groupby("original_idx") \
    .apply(lambda xs: list(xs.start - xs.start.min())).to_dict()
phoneme_onsets = [torch.tensor(phoneme_onsets[word_id.item()])
                  for word_id in story_stim.word_ids]
# -

sfreq = int(info["sfreq"])

# Add phoneme-level features based on aligned phoneme data.
X_ts_phoneme = torch.zeros(len(X_ts), len(story_stim.phoneme_feature_names))
flat_phoneme_onsets, flat_phoneme_features = [], []
for word_onset_i, phoneme_onsets_i, phoneme_features_i in zip(word_onsets, phoneme_onsets, story_stim.phoneme_features):
    flat_phoneme_onsets.append(phoneme_onsets_i + word_onset_i)
    flat_phoneme_features.append(phoneme_features_i)
flat_phoneme_onsets_samp = time_to_sample(torch.cat(flat_phoneme_onsets), sfreq)
flat_phoneme_features = torch.cat(flat_phoneme_features).float()
X_ts_phoneme[flat_phoneme_onsets_samp] = flat_phoneme_features

X_ts = torch.cat([X_ts, X_ts_phoneme], dim=1)
ts_feature_names += story_stim.phoneme_feature_names
assert X_ts.shape[1] == len(ts_feature_names)

# +

# Compare word onset measures from aligned data with stimulus features.
word_onset_feature_idx = ts_feature_names.index("word onsets_0")
ts_word_onset_samps = torch.where(X_ts[:, word_onset_feature_idx] == 1)[0]
word_onset_samps = time_to_sample(word_onsets, sfreq)
print(f"{Style.BRIGHT}Word onset checks{Style.RESET_ALL}")
print(f"{len(ts_word_onset_samps)} word onsets from time series features")
print(f"{len(word_onset_samps)} word onsets from aligned data")
n_words_missing_in_ts = len(set(word_onset_samps.numpy()) - set(ts_word_onset_samps.numpy()))
print(f"Words in aligned data missing from time series: {n_words_missing_in_ts} "
      f"({n_words_missing_in_ts / len(word_onset_samps):.2%})")
n_words_missing_in_aligned = len(set(ts_word_onset_samps.numpy()) - set(word_onset_samps.numpy()))
print(f"Words in time series missing from aligned data: {n_words_missing_in_aligned} "
      f"({n_words_missing_in_aligned / len(ts_word_onset_samps):.2%})")
print(f"{Fore.GREEN}Updating time series word onset feature to match our word representations.\n{Style.RESET_ALL}")
X_ts[:, word_onset_feature_idx] = 0
X_ts[word_onset_samps, word_onset_feature_idx] = 1

# Compare phoneme onset measures from aligned data with stimulus features.
phoneme_onset_feature_idx = ts_feature_names.index("phoneme onsets_0")
ts_phoneme_onset_samps = torch.where(X_ts[:, phoneme_onset_feature_idx] == 1)[0]
phoneme_onsets_flat = torch.cat([ph_ons_i + word_onset for word_onset, ph_ons_i in zip(word_onsets, phoneme_onsets)])
phoneme_onset_samps = time_to_sample(phoneme_onsets_flat, sfreq)
print(f"{Style.BRIGHT}Phoneme onset checks{Style.RESET_ALL}")
print(f"{len(ts_phoneme_onset_samps)} phoneme onsets from time series features")
print(f"{len(flat_phoneme_onsets_samp)} phoneme onsets from aligned data")
n_phonemes_missing_in_ts = len(set(flat_phoneme_onsets_samp.numpy()) - set(ts_phoneme_onset_samps.numpy()))
print(f"Phonemes in aligned data missing from time series: {n_phonemes_missing_in_ts} "
      f"({n_phonemes_missing_in_ts / len(flat_phoneme_onsets_samp):.2%})")
n_phonemes_missing_in_aligned = len(set(ts_phoneme_onset_samps.numpy()) - set(flat_phoneme_onsets_samp.numpy()))
print(f"Phonemes in time series missing from aligned data: {n_phonemes_missing_in_aligned} "
      f"({n_phonemes_missing_in_aligned / len(ts_phoneme_onset_samps):.2%})")
print(f"{Fore.GREEN}Updating time series phoneme onset feature to match our phoneme representations.\n{Style.RESET_ALL}")
X_ts[:, phoneme_onset_feature_idx] = 0
X_ts[flat_phoneme_onsets_samp, phoneme_onset_feature_idx] = 1

max_num_phonemes = max(len(onsets) for onsets in phoneme_onsets)
# Sanity check: max_num_phonemes as computed from aligned data should
# match that produced earlier by the natural language stimulus processor
assert max_num_phonemes == story_stim.candidate_phonemes.shape[2], \
    "%d %d" % (max_num_phonemes, story_stim.candidate_phonemes.shape[2])
max_num_phonemes = story_stim.candidate_phonemes.shape[2]
phoneme_onsets = torch.stack([
    pad(onsets, (0, max_num_phonemes - len(onsets)), value=0.)
    if len(onsets) < max_num_phonemes
    else onsets[:max_num_phonemes]
    for onsets in phoneme_onsets
])

ret = BerpDataset(
    name=f"{story_name}/{subject}",
    stimulus_name=story_stim.name,
    sample_rate=sfreq,
    
    phonemes=story_stim.phonemes,
    
    word_onsets=word_onsets,
    word_offsets=word_offsets,
    phoneme_onsets=phoneme_onsets,
    
    X_ts=X_ts,
    ts_feature_names=ts_feature_names,
    X_variable=X_variable,
    variable_feature_names=variable_feature_names,
    
    Y=eeg.get_data().T,
    sensor_names=eeg.info.ch_names,
)

with args.output_path.open("wb") as f:
    pickle.dump(ret, f)
