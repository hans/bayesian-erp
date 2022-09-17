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
    args = Namespace(confusion_consonants_path=Path("../../data/gillis2021/confusion/smits_2003_consonants.csv"),
                     confusion_vowels_path=Path("../../data/gillis2021/confusion/smits_2003_vowels.csv"),
                     dataset_path=Path("DKZ_1.pkl"),
                     output_path=Path("confusion.npz"))
else:
    args = parser.parse_args()

# ## Load and prepare from matrices
#
# Downloaded from https://www.mpi.nl/world/dcsp/diphones/

# +
# conf5_theirs = pd.read_csv("../../data/gillis2021/confusion/phon2_conf_matrix_gate5.dat", sep="\s+")
# conf5_theirs

# +
# conf5_theirs / conf5_theirs.sum(axis=0)

# +
# plt.subplots(figsize=(10,10))
# sns.heatmap(conf5_theirs / conf5_theirs.sum(axis=0))

# +
# conf5_theirs.columns
# -

# ## Load and prepare IPA-level confusions

consonants_df = pd.read_csv(args.confusion_consonants_path)
vowels_df = pd.read_csv(args.confusion_vowels_path)
with open(args.dataset_path, "rb") as f:
    dataset = pickle.load(f)

# +
# Simplify along third dimension.
consonants_df_gate1 = consonants_df.loc[::2].copy()
consonants_df_gate4 = consonants_df.loc[1::2].copy()
consonants_df_gate4["Stimulus"] = consonants_df_gate1.Stimulus.tolist()

vowels_df_gate1 = vowels_df.loc[::2].copy()
vowels_df_gate4 = vowels_df.loc[1::2].copy()
vowels_df_gate4["Stimulus"] = vowels_df_gate1.Stimulus.tolist()
# -

all_consonants = consonants_df_gate1.Stimulus.tolist()
all_consonants

all_vowels = vowels_df_gate1.Stimulus.tolist()
all_vowels

# +
# Allocate uniform mass to vowels in consonants_df and vice versa.
for consonant_df in [consonants_df_gate1, consonants_df_gate4]:
    consonant_df.loc[:, all_vowels] = \
        np.tile(consonant_df.Vowel.to_numpy()[:, None] / len(all_vowels),
                (1, len(all_vowels)))
    consonant_df.drop(columns=["Vowel"], inplace=True)
for vowel_df in [vowels_df_gate1, vowels_df_gate4]:
    vowel_df.loc[:, all_consonants] = \
        np.tile(vowel_df.Consonant.to_numpy()[:, None] / len(all_consonants),
                (1, len(all_consonants)))
    vowel_df.drop(columns=["Consonant"], inplace=True)

# Concatenate into inventory-wide confusion matrices.
confusion_gate1 = pd.concat([consonants_df_gate1, vowels_df_gate1]) \
    .set_index("Stimulus")
confusion_gate4 = pd.concat([consonants_df_gate4, vowels_df_gate4]) \
    .set_index("Stimulus")

# +
# These phonemes are special in the dataset and we'll manually add them to the confusion matrix.
MAGIC_PHONEMES = ["_"]
for phon in MAGIC_PHONEMES:
    assert phon in dataset.phonemes

for conf in [confusion_gate1, confusion_gate4]:
    diag_mean = np.diag(conf).mean()
    for phon in MAGIC_PHONEMES:
        conf.loc[phon, phon] = diag_mean
    conf.fillna(0., inplace=True)
# -

confusion_gate1

normed = confusion_gate4 + 1
normed = normed / normed.sum(axis=0)
normed

import seaborn as sns
import matplotlib.pyplot as plt
plt.subplots(figsize=(10, 10))
sns.heatmap(normed)

# +
from sklearn.decomposition import PCA
normed_without_magic = normed.drop(columns=MAGIC_PHONEMES)
vals = PCA(n_components=2).fit_transform(normed_without_magic.values.T)

plt.subplots(figsize=(10, 10))
plt.scatter(vals[:, 0], vals[:, 1])
for phon, (x, y) in zip(normed_without_magic.columns, vals):
    plt.text(x, y, phon)
# -

# ## Final checks and updates

assert confusion_gate1.index.tolist() == confusion_gate4.index.tolist()
assert confusion_gate1.index.tolist() == confusion_gate1.columns.tolist()
assert confusion_gate4.index.tolist() == confusion_gate4.columns.tolist()

ds_phonemes = set(dataset.phonemes)
conf_phonemes = set(confusion_gate1.index.tolist())
print(ds_phonemes - conf_phonemes)
assert set(dataset.phonemes).issubset(set(confusion_gate1.index.tolist()))

# Reindex confusion matrices according to dataset phonemes.
confusion_gate1 = confusion_gate1.reindex(index=dataset.phonemes, columns=dataset.phonemes)
confusion_gate4 = confusion_gate4.reindex(index=dataset.phonemes, columns=dataset.phonemes)
confusion_gate1

np.savez(args.output_path,
         confusion=confusion_gate4.to_numpy(),  # TODO which to choose?
         confusion_gate1=confusion_gate1.to_numpy(),
         confusion_gate4=confusion_gate4.to_numpy(),
         phonemes=confusion_gate1.index.tolist())


