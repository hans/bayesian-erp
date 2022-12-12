from pathlib import Path
import pickle
from typing import Tuple, NamedTuple, Optional, Dict, List

import numpy as np
import pytest
import torch

from berp.datasets import BerpDataset, NaturalLanguageStimulus


ConfusionData = Dict


class IntegrationHarnessRequest(NamedTuple):
    workflow: str
    story_name: str
    subject: str
    run: Optional[str]
    model: str
    features_spec: str
    dataset_spec: str

    @property
    def dataset_path(self):
        return f"{self.workflow}.{self.story_name}.{self.subject}{'.' + self.run if self.run is not None else ''}.pkl"

    @property
    def stimulus_path(self):
        return f"{self.workflow}.{self.story_name}{'.' + self.run if self.run is not None else ''}.pkl"

    @property
    def confusion_path(self):
        return f"{self.workflow}.confusion.npz"


class IntegrationHarness(NamedTuple):
    dataset_path: Path
    stimulus_path: Path
    confusion_path: Path

    stimulus_name: str
    features_spec: str
    dataset_spec: str

    @classmethod
    def from_request(cls, req: IntegrationHarnessRequest, root_dir: Optional[Path] = None):
        if root_dir is None:
            root_dir = Path(__file__).parent
        
        dataset_path = root_dir / req.dataset_path
        stimulus_path = root_dir / req.stimulus_path
        confusion_path = root_dir / req.confusion_path
        if dataset_path.exists() and stimulus_path.exists() and confusion_path.exists():
            # HACK assumption
            stim_name = f"{req.story_name}{'/' + req.run if req.run is not None else ''}"
        else:
            ds, stim, confusion = make_integration_harness_data(req)
            stim_name = stim.name

            with open(dataset_path, "wb") as f:
                pickle.dump(ds, f)
            with open(stimulus_path, "wb") as f:
                pickle.dump(stim, f)
            np.savez(confusion_path, **confusion)

        return cls(
            dataset_path=dataset_path,
            stimulus_path=stimulus_path,
            confusion_path=confusion_path,
            stimulus_name=stim_name,
            features_spec=req.features_spec,
            dataset_spec=req.dataset_spec)


def make_integration_harness_data(
    req: IntegrationHarnessRequest,
    retain_samples: int = 1000,
) -> Tuple[BerpDataset, NaturalLanguageStimulus, ConfusionData]:
    # Compute paths relative to project root.
    root_dir = Path(__file__).parent.parent.parent
    dataset_path = root_dir / f"workflow/{req.workflow}/data/dataset/{req.model}/{req.story_name}/{req.subject}{'/' + req.run if req.run is not None else ''}.pkl"
    stim_path = root_dir / f"workflow/{req.workflow}/data/stimulus/{req.model}/{req.story_name}{'/' + req.run if req.run is not None else ''}.pkl"

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

    # Prepare dummy confusion matrix.
    confusion = np.eye(len(stim.phonemes))
    confusion += 1e-2
    confusion /= confusion.sum(axis=0, keepdims=True)
    confusion = {"confusion": confusion, "phonemes": stim.phonemes}

    return ds, stim, confusion


integration_requests = {
    "gillis2021": ("gillis2021", "DKZ_1", "microaverage", None, "GroNLP/gpt2-small-dutch", "dkz", "dkz_microaverage"),
    "heilbron2022": ("heilbron2022", "old-man-and-the-sea", "sub17", "run1", "distilgpt2", "heilbron", "heilbron"),
}
integration_requests = {k: IntegrationHarnessRequest(*v) for k, v in integration_requests.items()}


@pytest.fixture(scope="session",
                params=integration_requests.values(),
                ids=list(integration_requests.keys()))
def integration_harness(request) -> IntegrationHarness:
    """
    Prepare integration test harness for different data pipelines.

    Returns a tuple of
    - dataset path
    - stimulus path
    - confusion path
    - name of stimulus
    """
    return IntegrationHarness.from_request(request.param)