from pathlib import Path
import pickle

import torch

from berp.datasets import NestedBerpDataset, BerpDataset, NaturalLanguageStimulus
from berp.datasets.eeg import load_eeg_dataset


dataset_path = "workflow/gillis2021/data/dataset/GroNLP/gpt2-small-dutch/DKZ_1/microaverage.pkl"
stim_path = "workflow/gillis2021/data/stimulus/GroNLP/gpt2-small-dutch/DKZ_1.pkl"
outdir = "test/integration"

with open(dataset_path, "rb") as f:
    ds: BerpDataset = pickle.load(f)
with open(stim_path, "rb") as f:
    stim: NaturalLanguageStimulus = pickle.load(f)

ds.add_stimulus(stim)

# Now massively subset data.
orig_name = ds.name
ds = ds[0:1000]
# And pretend it didn't happen :)
ds.name = orig_name
ds.global_slice_indices = None
# Remove padding
ds.phoneme_onsets = ds.phoneme_onsets[:, :ds.word_lengths.max()]

print(f"{len(ds.word_onsets)} words remaining.")
print(ds.phoneme_onsets.shape)
print(ds.word_lengths)
retained_word_idxs = torch.arange(len(ds.word_onsets))

# Subset stim accordingly.
stim.word_ids = stim.word_ids[retained_word_idxs]
stim.word_lengths = ds.word_lengths
stim.word_features = stim.word_features[retained_word_idxs]
stim.p_candidates = ds.p_candidates
stim.candidate_ids = stim.candidate_ids[retained_word_idxs]
# Remove cached property
del stim.__dict__["candidate_phonemes"]

ds.p_candidates = None
ds.phonemes = None
ds.word_lengths = None
ds.candidate_phonemes = None
ds.add_stimulus(stim)
ds.check_shapes()

del stim.__dict__["candidate_phonemes"]


with (Path(outdir) / "DKZ_1.microaverage.pkl").open("wb") as f:
    pickle.dump(ds, f)
with (Path(outdir) / "DKZ_1.pkl").open("wb") as f:
    pickle.dump(stim, f)