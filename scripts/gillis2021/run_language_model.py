# +
from argparse import ArgumentParser, Namespace
from collections import Counter
import logging
from pathlib import Path
import pickle
import re
import sys
sys.path.append(str(Path(".").resolve().parent.parent))
from typing import List

import h5py
import mne
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import pad
from tqdm.auto import tqdm
# -

IS_INTERACTIVE = False
try:
    get_ipython()  # type: ignore
except NameError: pass
else: IS_INTERACTIVE = True
IS_INTERACTIVE

# %load_ext autoreload
# %autoreload 2

from berp.datasets import NaturalLanguageStimulusProcessor
from berp.languages import dutch

p = ArgumentParser()
p.add_argument("tokenized_path", type=Path)
p.add_argument("aligned_words_path", type=Path)
p.add_argument("aligned_phonemes_path", type=Path)
p.add_argument("-m", "--model", default="GroNLP/gpt2-small-dutch")
p.add_argument("-n", "--n_candidates", type=int, default=10)
p.add_argument("--vocab_path", type=Path, default="../../data/gillis2021/vocab.pkl")
p.add_argument("--celex_path", type=Path, default="../../data/gillis2021/celex_dpw_cx.txt")

if IS_INTERACTIVE:
    args = Namespace(tokenized_path=Path("DKZ_1.tokenized.txt"),
                     aligned_words_path=Path("DKZ_1.words.csv"),
                     aligned_phonemes_path=Path("DKZ_1.phonemes.csv"),
                     model="GroNLP/gpt2-small-dutch",
                     n_candidates=1000,
                     vocab_path=Path("../../data/gillis2021/vocab.pkl"),
                     celex_path=Path("../../data/gillis2021/celex_dpw_cx.txt"))
else:
    args = p.parse_args()

PAD_PHONEME = "_"

story_name = args.tokenized_path.stem.rsplit(".tokenized")[0]
tokenized = args.tokenized_path.read_text().strip()

# +
words_df = pd.read_csv(args.aligned_words_path)
phonemes_df = pd.read_csv(args.aligned_phonemes_path)

# Filter for current story.
print(story_name)
assert story_name in words_df.story.unique()
assert story_name in phonemes_df.story.unique()
words_df = words_df[words_df.story == story_name]
phonemes_df = phonemes_df[phonemes_df.story == story_name]
# -

# ## Prepare frequency data

with args.vocab_path.open("rb") as f:
    vocab = pickle.load(f)

# Convert to neg-log2-freq
total_freq = sum(vocab.values())
vocab = {word: -np.log2(freq / total_freq)
                  for word, freq in vocab.items()}

words_df["frequency"] = words_df.text.map(vocab)
# Put words with missing frequency in the lowest 2 percentile. (Get last bin of quantile cut.)
oov_freq = pd.qcut(words_df.frequency, 50, retbins=True, duplicates="drop")[1][-1]
words_df["frequency"] = words_df.frequency.fillna(oov_freq)

# ----

# ## Process story language data


celex_phonemizer = dutch.CelexPhonemizer(args.celex_path)

# +
phonemes = sorted(dutch.smits_ipa_chars) + [PAD_PHONEME]

proc = NaturalLanguageStimulusProcessor(phonemes=phonemes, hf_model=args.model,
                                        num_candidates=args.n_candidates,
                                        disallowed_re=f"[^{''.join(dutch.celex_chars)}]",
                                        phonemizer=celex_phonemizer)

# +
tokens = tokenized.split(" ")

# Prepare proc metadata input.
word_to_token = words_df \
    .astype({"original_idx": int}) \
    .groupby("original_idx") \
    .apply(lambda x: list(x.tok_idx)).to_dict()
ground_truth_phonemes = phonemes_df[~phonemes_df.original_idx.isna()] \
    .astype({"original_idx": int}) \
    .groupby("original_idx").apply(lambda xs: list(xs.text)).to_dict()
# Convert CGN representation to IPA representation.
ground_truth_phonemes = {
    idx: [dutch.convert_to_smits_ipa(dutch.cgn_ipa_mapping[phon])
          for phon in phons]
    for idx, phons in ground_truth_phonemes.items()
}

# Prepare word-level features.
word_features = dict(words_df.groupby(["original_idx"])
                     .apply(lambda xs: torch.tensor(xs.iloc[0].frequency).unsqueeze(0)))

stim = proc(tokens, word_to_token, word_features, ground_truth_phonemes)
# -

celex_phonemizer.missing_counter.most_common(50)

with open(f"{story_name}.pkl", "wb") as f:
    pickle.dump(stim, f)
