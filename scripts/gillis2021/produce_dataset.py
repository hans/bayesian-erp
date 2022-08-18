# +
from argparse import ArgumentParser, Namespace
from collections import defaultdict, Counter
import itertools
from pathlib import Path
import pickle
import re
import sys
sys.path.append(str(Path(".").resolve().parent.parent))

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
    get_ipython()
except NameError: pass
else: IS_INTERACTIVE = True
IS_INTERACTIVE

# %load_ext autoreload
# %autoreload 2

from berp.datasets import BerpDataset
from berp.datasets import NaturalLanguageStimulusProcessor

p = ArgumentParser()
p.add_argument("tokenized_corpus_dir", type=Path)
p.add_argument("aligned_words_path", type=Path)
p.add_argument("aligned_phonemes_path", type=Path)
p.add_argument("eeg_dir", type=Path)
p.add_argument("stim_path", type=Path)
p.add_argument("-m", "--model", default="GroNLP/gpt2-small-dutch")
p.add_argument("-n", "--n_candidates", type=int, default=10)
p.add_argument("--vocab_path", type=Path, default="../../data/gillis2021/vocab.txt")

if IS_INTERACTIVE:
    args = Namespace(tokenized_corpus_dir=Path("tokenized"),
                     aligned_words_path=Path("aligned_words.csv"),
                     aligned_phonemes_path=Path("aligned_phonemes.csv"),
                     eeg_dir=Path("../../data/gillis2021/eeg"),
                     stim_path=Path("stimuli.npz"),
                     model="GroNLP/gpt2-small-dutch",
                     n_candidates=10,
                     vocab_path=Path("../../data/gillis2021/vocab.txt"))
else:
    args = p.parse_args()

PAD_PHONEME = "_"
EEG_SUFFIX = "_1_256_8_average_4_128"

eeg_paths = {story: list(paths)
             for story, paths in itertools.groupby(args.eeg_dir.glob("*/*.mat"),
                                                   key=lambda path: path.parent.name)}

subjects = [p.name.replace(f"{EEG_SUFFIX}.mat", "")
            for p in next(iter(eeg_paths.values()))]

tokenized = {}
for tokenized_path in args.tokenized_corpus_dir.glob("*.txt"):
    with tokenized_path.open() as f:
        tokenized[tokenized_path.stem] = f.read().strip()

words_df = pd.read_csv(args.aligned_words_path)
phonemes_df = pd.read_csv(args.aligned_phonemes_path)

# ## Prepare frequency data

drop_re = re.compile(r"[^a-zA-Z]")

vocab = args.vocab_path.read_text()
filtered_vocab = Counter()
for line in tqdm(vocab.strip("\n").split("\n")):
    word, freq = line.rsplit("\t", 1)
    
    if drop_re.search(word):
        continue
    filtered_vocab[word.lower()] += int(freq)

# Convert to neg-log2-freq
total_freq = sum(filtered_vocab.values())
filtered_vocab = {word: -np.log2(freq / total_freq)
                  for word, freq in filtered_vocab.items()}

words_df["frequency"] = words_df.text.map(filtered_vocab)
# Put words with missing frequency in the lowest 2 percentile. (Get last bin of quantile cut.)
oov_freq = pd.qcut(words_df.frequency, 50, retbins=True, duplicates="drop")[1][-1]
words_df["frequency"] = words_df.frequency.fillna(oov_freq)


# ----

# ## Process story language data

# +
# TODO casing
# -

# Dumb phonemizer which just drops characters not in the phon vocabulary
# TODO make not dumb. Use a model or build your own model from the corpus.
def phonemizer(string):
    return [phon for phon in string if phon in phonemes]


# +
phonemes = sorted(phonemes_df.text.unique()) + [PAD_PHONEME]

proc = NaturalLanguageStimulusProcessor(phonemes=phonemes, hf_model=args.model,
                                        num_candidates=args.n_candidates,
                                        phonemizer=phonemizer)


# -

def process_story_language(story):
    story_words_df = words_df[words_df.story == story]
    story_phonemes_df = phonemes_df[phonemes_df.story == story]
    
    # Prepare token mask.
    tokens = tokenized[story].split(" ")
    # Find all tokens that are not covered by row(s) of story_words_df.
    mask_token_ids = set(np.arange(len(tokens))) - set(story_words_df.tok_idx)
    token_mask = np.ones(len(tokens), dtype=bool)
    token_mask[list(mask_token_ids)] = False
    
    # Also mask tokens which are not the first for a word.
    secondary_subword = story_words_df.original_idx.diff(1) == 0
    token_mask[story_words_df[secondary_subword].tok_idx.to_numpy()] = False
    
    # Prepare proc metadata input.
    word_to_token = story_words_df \
        .astype({"original_idx": int}) \
        .groupby("original_idx") \
        .apply(lambda x: list(x.tok_idx)).to_dict()
    ground_truth_phonemes = story_phonemes_df[~story_phonemes_df.original_idx.isna()] \
        .astype({"original_idx": int}) \
        .groupby("original_idx").apply(lambda xs: list(xs.text)).to_dict()
    
    # Prepare word-level features.
    word_features = dict(story_words_df.groupby(["original_idx"])
                         .apply(lambda xs: torch.tensor(xs.iloc[0].frequency).unsqueeze(0)))
    
    return proc(tokens, token_mask, word_to_token, word_features, ground_truth_phonemes)


processed_stories = {story: process_story_language(story)
                     for story in tqdm(tokenized)}

# ## Load and process stimulus features

stimulus_features = np.load(args.stim_path)


# ## Load and process EEG data

def load_eeg(path, info: mne.Info,
             trim_n_samples=None):
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


def produce_dataset(story, subject, mne_info: mne.Info,
                    eeg_suffix=EEG_SUFFIX,
                    features=None,
                   ) -> BerpDataset:
    # Prepare predictors.
    story_stim = processed_stories[story]
    # Variable onset features are simply word features and word surprisals.
    X_variable = torch.concat([story_stim.word_features,
                               story_stim.word_surprisals.unsqueeze(1)],
                              dim=1)
    # Load other stimulus time-series features.
    X_ts = torch.tensor(stimulus_features[story])

    # Load EEG and trim.
    eeg_path = args.eeg_dir / story / f"{subject}{eeg_suffix}.mat"
    if not eeg_path.exists():
        raise ValueError(f"Cannot find EEG data at path {eeg_path}")
    eeg = load_eeg(eeg_path, info, trim_n_samples=X_ts.shape[0])
    
    # Retrieve onset information.
    word_onsets = words_df[words_df.story == story] \
        .groupby("original_idx").start.min().to_dict()
    word_onsets = torch.tensor([word_onsets[word_id.item()]
                                for word_id in story_stim.word_ids])
    
    # Phoneme onsets.
    # TODO these can be precomputed
    phoneme_onsets = phonemes_df[phonemes_df.story == story] \
        .groupby("original_idx") \
        .apply(lambda xs: list(xs.start)).to_dict()
    phoneme_onsets = [torch.tensor(phoneme_onsets[word_id.item()])
                      for word_id in story_stim.word_ids]
    # TODO this fails. why? check that the data match? probably an
    # indexing bug somewhere.
    # max_num_phonemes = max(len(onsets) for onsets in phoneme_onsets)
    # assert max_num_phonemes == story_stim.candidate_phonemes.shape[2]
    max_num_phonemes = story_stim.candidate_phonemes.shape[2]
    phoneme_onsets = torch.stack([
        pad(onsets, (0, max_num_phonemes - len(onsets)), value=0.)
        if len(onsets) < max_num_phonemes
        else onsets[:max_num_phonemes]
        for onsets in phoneme_onsets
    ])
    
    ret = BerpDataset(
        name=f"{story}/{subject}",
        sample_rate=int(info["sfreq"]),
        
        phonemes=story_stim.phonemes,
        p_word=story_stim.p_word,
        word_lengths=story_stim.word_lengths,
        candidate_phonemes=story_stim.candidate_phonemes,
        
        word_onsets=word_onsets,
        phoneme_onsets=phoneme_onsets,
        
        X_ts=X_ts,
        X_variable=X_variable,
        
        Y=eeg.get_data().T
    )
    
    return ret


for story_name in tokenized:
    for subject in subjects:
        ds = produce_dataset(story_name, subject, info)
        with open(f"{subject}.{story_name}.pkl", "wb") as f:
            pickle.dump(ds, f)
