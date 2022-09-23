"""
Prepare confusion matrices for use in the model.
"""

# +
from argparse import ArgumentParser, Namespace
from pathlib import Path
import pickle
import sys

sys.path.append(str(Path(".").resolve().parent.parent))
# -

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %load_ext autoreload
# %autoreload 2

from berp.languages.dutch import convert_to_smits_ipa, cgn_ipa_mapping

IS_INTERACTIVE = False
try:
    get_ipython()  # type: ignore
except NameError: pass
else: IS_INTERACTIVE = True
IS_INTERACTIVE

parser = ArgumentParser()
parser.add_argument("confusion_consonants_path", type=Path)
parser.add_argument("confusion_vowels_path", type=Path)
parser.add_argument("dataset_path", type=Path)
parser.add_argument("output_path", type=Path)

if IS_INTERACTIVE:
    args = Namespace(confusion_path=Path("../../data/gillis2021/confusion/phon2_conf_matrix_gate5.dat"),
                     dataset_path=Path("DKZ_1.pkl"),
                     output_path=Path("confusion.npz"))
else:
    args = parser.parse_args()

# Prepare to map from CELEX-DISC coding used in Smits et al 2003 responses to IPA
disc_to_ipa = {
    'p': 'p',
    't': 't',
    'k': 'k',
    'b': 'b',
    'd': 'd',
    'g': 'g',
    '-': 'dʒ',
    'f': 'f',
    's': 's',
    'S': 'ʃ',
    'x': 'x',
    'v': 'v',
    'z': 'z',
    'Z': 'ʒ',
    'h': 'h',
    'r': 'r',
    'l': 'l',
    'w': 'w',
    'j': 'j',
    'm': 'm',
    'n': 'n',
    'N': 'ŋ',
    'A': 'ɑ',
    'E': 'ɛ',
    'I': 'ɪ',
    'O': 'ɔ',
    '}': 'ʏ',
    '@': 'ə',
    'i': 'i',
    'u': 'u',
    'y': 'y',
    'e': 'e',
    'o': 'o',
    '|': 'œ',
    'a': 'a',
    'K': 'ɛi',
    'L': 'œy',
    'M': 'ɑu'
}

# ## Load and prepare from matrices
#
# Downloaded from https://www.mpi.nl/world/dcsp/diphones/

# +
conf_df = pd.read_csv("../../data/gillis2021/confusion/phon2_conf_matrix_gate5.dat", sep="\s+")

# They represent stimulus as row and response as column. Transpose to match dataset expectation.
conf_df = conf_df.T

conf_df.columns = conf_df.columns.map(disc_to_ipa)
conf_df.index = conf_df.index.map(disc_to_ipa)
conf_df
# -

if IS_INTERACTIVE:
    plt.subplots(figsize=(10,10))
    sns.heatmap(conf_df / conf_df.sum(axis=0))

with open(args.dataset_path, "rb") as f:
    dataset = pickle.load(f)

# +
# These phonemes are special in the dataset and we'll manually add them to the confusion matrix.
MAGIC_PHONEMES = ["_"]
for phon in MAGIC_PHONEMES:
    assert phon in dataset.phonemes

diag_mean = np.diag(conf_df).mean()
for phon in MAGIC_PHONEMES:
    conf_df.loc[phon, phon] = diag_mean
conf_df.fillna(0., inplace=True)
# -

conf_df

if IS_INTERACTIVE:
    normed = conf_df + 1
    normed = normed / normed.sum(axis=0)
    normed
    
    plt.subplots(figsize=(10, 10))
    sns.heatmap(normed)

if IS_INTERACTIVE:
    from sklearn.decomposition import PCA
    normed_without_magic = normed.drop(columns=MAGIC_PHONEMES)
    vals = PCA(n_components=2).fit_transform(normed_without_magic.values.T)

    plt.subplots(figsize=(10, 10))
    plt.scatter(vals[:, 0], vals[:, 1])
    for phon, (x, y) in zip(normed_without_magic.columns, vals):
        plt.text(x, y, phon)

# ## Final checks and updates

assert conf_df.index.tolist() == conf_df.columns.tolist()

ds_phonemes = set(dataset.phonemes)
conf_phonemes = set(conf_df.index.tolist())
print(ds_phonemes - conf_phonemes)
assert set(dataset.phonemes).issubset(set(conf_df.index.tolist()))

# Reindex confusion matrices according to dataset phonemes.
conf_df = conf_df.reindex(index=dataset.phonemes, columns=dataset.phonemes)
conf_df

np.savez(args.output_path,
         confusion=conf_df.to_numpy(),
         phonemes=conf_df.index.tolist())
