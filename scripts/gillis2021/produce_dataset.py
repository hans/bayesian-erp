# +
from pathlib import Path
import re
import sys
sys.path.append(str(Path(".").resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import transformers
# -

# %load_ext autoreload
# %autoreload 2

from berp.generators.stimulus import NaturalLanguageStimulusGenerator
from berp.models.reindexing_regression import RRDataset

# +
# TODO update with argv
model = "GroNLP/gpt2-small-dutch"
n_candidates = 10
tokenized_dir = Path("tokenized")
words_path = Path("aligned_words.csv")
phonemes_path = Path("aligned_phonemes.csv")

PAD_PHONEME = "_"
# -

tokenized = {}
for tokenized_path in tokenized_dir.glob("*.txt"):
    with tokenized_path.open() as f:
        tokenized[tokenized_path.stem] = f.read().strip()

words_df = pd.read_csv(words_path)
words_df.head(10)

phonemes_df = pd.read_csv(phonemes_path)
phonemes_df


# ----

# +
# TODO casing
# -

# Dumb phonemizer which just drops characters not in the phon vocabulary
# TODO make not dumb
def phonemizer(string):
    return [phon for phon in string if phon in phonemes]


phonemes = sorted(phonemes_df.text.unique()) + [PAD_PHONEME]
gen = NaturalLanguageStimulusGenerator(phonemes=phonemes, hf_model=model,
                                       num_candidates=n_candidates,
                                       phonemizer=phonemizer)

tokens = tokenized["DKZ_1"].split(" ")
# Find all tokens that are not covered by row(s) of words_df.
mask_token_ids = set(np.arange(len(tokens))) - set(words_df.tok_idx)
token_mask = np.ones(len(tokens), dtype=bool)
token_mask[list(mask_token_ids)] = False

# Also mask tokens which are not the first for a word.
secondary_subword = words_df.original_idx.diff(1) == 0
token_mask[words_df[secondary_subword].tok_idx.to_numpy()] = False

word_to_token = words_df.groupby("original_idx").apply(lambda x: list(x.tok_idx)).to_dict()
ground_truth_phonemes = phonemes_df[~phonemes_df.original_idx.isna()].astype({"original_idx": int}) \
    .groupby("original_idx").apply(lambda xs: list(xs.text)).to_dict()

gen(tokens, token_mask, word_to_token,
    ground_truth_phonemes=ground_truth_phonemes)

# +
input_ids = torch.tensor(gen._tokenizer.convert_tokens_to_ids(tokenized["DKZ_1"].split(" "))).unsqueeze(0)
# dumb truncation
input_ids = input_ids[:, :100]  # gen._model.config.n_positions]
print(input_ids.shape)

# TODO pad, truncate, batch, etc.
candidate_ids = gen.get_predictive_topk({"input_ids": input_ids})
# gen.get_candidate_phonemes(input_ids, 10)

# +
# TODO is _clean_word dropping necessary diacritics?
# -

[gen._tokenizer.convert_ids_to_tokens(cand_i[:5])
 for cand_i in candidate_ids[1][0, :20]]

ground_truth_phonemes = phonemes_df.iloc[:1000].groupby("tok_idx").apply(lambda xs: list(xs.text)).tolist()
ground_truth_phonemes[:3]

candidate_phonemes = gen.get_candidate_phonemes(candidate_ids[1], max_num_phonemes=5,
                                                ground_truth_phonemes=ground_truth_phonemes)

[phonemes[i] for i in candidate_phonemes[0][0, 0, 5]]


def produce_dataset(story, subject) -> RRDataset:
    pass
