# +
from argparse import ArgumentParser, Namespace
from collections import Counter
import logging
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

p = ArgumentParser()
p.add_argument("tokenized_path", type=Path)
p.add_argument("aligned_words_path", type=Path)
p.add_argument("aligned_phonemes_path", type=Path)
p.add_argument("-m", "--model", default="GroNLP/gpt2-small-dutch")
p.add_argument("-n", "--n_candidates", type=int, default=10)
p.add_argument("--vocab_path", type=Path, default="../../data/gillis2021/vocab.pkl")

if IS_INTERACTIVE:
    args = Namespace(tokenized_path=Path("tokenized/DKZ_1.txt"),
                     aligned_words_path=Path("aligned_words.csv"),
                     aligned_phonemes_path=Path("aligned_phonemes.csv"),
                     model="GroNLP/gpt2-small-dutch",
                     n_candidates=10,
                     vocab_path=Path("../../data/gillis2021/vocab.pkl"))
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

# +
# TODO casing
# -

# NB # in CGN denotes cough, sneeze, etc.
celex_cgn_mapping = {
    "&:": "2",
    "@": "@",
    "A": "A",
    "AU": "A+",
    "E": "E",
    "EI": "E+",
    "G": "G",
    "I": "I",
    "N": "N",
    "O": "O",
    "S": "S",
    "U": "Y",
    "UI": "Y+",
    "a:": "a",
    "b": "b",
    "d": "d",
    "e:": "e",
    "f": "f",
    "h": "h",
    "i:": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o:": "o",
    "p": "p",
    "r": "r",
    "s": "s",
    "t": "t",
    "u:": "u",
    "v": "v",
    "w": "w",
    "x": "x",
    "y:": "y",
    "z": "z",
}


def convert_celex_to_cgn(celex):
    # Greedily consume phonemes
    celex_keys = sorted(celex_cgn_mapping.keys(), key=lambda code: -len(code))
    ret = []
    orig = celex
    i = 0
    while celex:
        for key in celex_keys:
            if celex.startswith(key):
                ret.append(celex_cgn_mapping[key])
                celex = celex[len(key):]
                break
        else:
            raise KeyError(f"{orig} -> {celex}")
            
        i += 1
        if i == 10:
            break
            
    return ret


# Load CELEX pronunciation database. Keep only the most frequent pronunciation for a word.
phonemizer_df = pd.read_csv("../../data/gillis2021/celex_dpw_cx.txt", sep="\\", header=None,
                            usecols=[1, 2, 6], names=["word", "inl_freq", "celex_syl"]).dropna() \
    .sort_values("inl_freq", ascending=False) \
    .drop_duplicates(subset="word").set_index("word")
phonemizer_df["celex"] = phonemizer_df.celex_syl.str.replace(r"[\[\]]", "", regex=True)
phonemizer_df

punct_only_re = re.compile(r"^[.?!:'\"]+$")
def celex_phonemizer(string):
    if punct_only_re.match(string):
        return ""

    celex_form = phonemizer_df.loc[string].celex
    cgn_form = convert_celex_to_cgn(celex_form)
    return cgn_form


celex_chars = set([char for celex in phonemizer_df.celex.tolist() for char in celex])

# +
phonemes = sorted(phonemes_df.text.unique()) + [PAD_PHONEME]

proc = NaturalLanguageStimulusProcessor(phonemes=phonemes, hf_model=args.model,
                                        num_candidates=args.n_candidates,
                                        disallowed_re=f"[^{''.join(celex_chars)}]",
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

# Prepare word-level features.
word_features = dict(words_df.groupby(["original_idx"])
                     .apply(lambda xs: torch.tensor(xs.iloc[0].frequency).unsqueeze(0)))

stim = proc(tokens, word_to_token, word_features, ground_truth_phonemes)
# -

with open(f"{story_name}.pkl", "wb") as f:
    pickle.dump(stim, f)
