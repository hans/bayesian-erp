import logging
from pathlib import Path
import re
from typing import Tuple, List
import unicodedata

import mne
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from berp.generators.stimulus import Stimulus
from berp.models.reindexing_regression import RRDataset  # todo move out of generators

L = logging.getLogger(__name__)


def strip_accents(s):
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")

info_re = re.compile(r"S(\d+)")
punct_re = re.compile(r"[^A-Za-z]")

def process_fulltext_token(t):
    ret = strip_accents(t.lower())
    ret = punct_re.sub("", ret)
    return ret


def align_stimulus_fulltext(stim_df: pd.DataFrame, tokens_flat):
    """
    Prepare an alignment between the experimental stimuli and the fulltext token
    indices, so that we can provide surprisal predictors.

    Adds a new column `tok_pos` to `stim_df` describing the corresponding
    position for each row in the fulltext. (inplace)
    """

    stop = False

    tok_cursor = 0
    tok_el = process_fulltext_token(tokens_flat[tok_cursor])

    # For each element in surp_df, record the index of the corresponding element
    # in the token sequence or surprisal df.
    tok_pos = []
    for _, row in tqdm(stim_df.iterrows(), total=len(stim_df)):
        if stop:
            break

        df_el = punct_re.sub("", row.Word.lower())
        print(row.Word, "::", df_el, "::")

        # Track how many elements in a reference we have skipped. If this
        # is excessive, we'll quit rather than looping infinitely.
        skip_count = 0
        if stop:
            raise RuntimeError("abort")
            break

        # Find corresponding token in text and append to `tok_pos`.
        try:
            print("\t///", tok_el, df_el)
            while not tok_el.startswith(df_el):
                # Special cases for oddities in the Brennan stim df..
                if tok_el == "is" and df_el == "s":
                    # annotation says "\x1as" which gets stripped
                    break
                elif tok_el == "had" and df_el == "d":
                    # annotation says "\x1ad" which gets stripped
                    break

                tok_cursor += 1
                skip_count += 1
                if skip_count > 20:
                    stop = True
                    break

                tok_el = process_fulltext_token(tokens_flat[tok_cursor])
                print("\t//", tok_el)

            print("\tMatch", df_el, tok_el)
            tok_pos.append(tok_cursor)

            # If we matched only a subset of the token, then cut off what we
            # matched and proceed.
            if tok_el != df_el:
                tok_el = tok_el[len(df_el):]
            else:
                tok_cursor += 1
                tok_el = process_fulltext_token(tokens_flat[tok_cursor])
        except IndexError:
            # print("iex", row, tok_cursor, tok_el)
            stop = True
            break

    stim_df["tok_pos"] = tok_pos


class BrennanAdapter(object):
    """
    Adapter for Brennan et al 2018 Alice in Wonderland dataset.
    """

    def __init__(self, eeg_dir: Path):
        self.eeg_dir = eeg_dir

        for subdir in ["eeg", "stimulus", "aligned"]:
            assert (self.eeg_dir / subdir).exists()

        stim_path = eeg_dir / "stimuli" / "AliceChapterOne-EEG.csv"
        self._stim_df = pd.read_csv(
            stim_path,
            index_col=None) \
            .rename(columns=dict(Position="word_idx",
                                 Sentence="sentence_idx",
                                 Segment="segment_idx")) \
            .drop(columns=["LogFreq_Prev", "LogFreq_Next"]) \
            .set_index(["segment_idx", "sentence_idx", "word_idx"])

        self._presentation_dfs = {}

        self._load_eeg()
        self._load_stimulus(eeg_dir / "text" / "alice-ch1.txt")

    def _load_eeg(self):
        """
        Load EEG signal with MNE.
        """
        self._raw_data, self._annots, self._run_ranges = {}, {}, {}
        paths = list((self.eeg_dir / "eeg").glob("S*", ))

        for subject_dir in tqdm(paths, "loading subject data"):
            subject_id = int(info_re.match(subject_dir.name).group(1).lstrip("0"))
            L.debug("Loading subject %i", subject_id)

            self._raw_data[subject_id], self._presentation_dfs[subject_id] = \
                self._load_eeg_single_subject(subject_id, subject_dir)
            
            # DEV: stop here.
            break

    def _load_eeg_single_subject(self, subject_id: int, path) -> Tuple[mne.io.Raw, pd.DataFrame]:
        raw: mne.io.Raw = mne.io.read_raw(path / ("S%02i_alice-raw.fif" % subject_id),
                                          preload=True)

        # Prepare presentation df, specifying when this particular subject
        # observed each particular word.

        # Load segments; word onsets will be computed relative to segment annotation points.
        n_segment_annotations = len(raw.annotations)
        segment_data = pd.DataFrame(list(raw.annotations))
        segment_data["description"] = segment_data.description.astype(int)
        segment_data = segment_data.set_index("description") \
            .rename(columns={"onset": "segment_onset"})
        segment_data.index.name = "segment_idx"

        presentation_df = pd.merge(self._stim_df, segment_data[["segment_onset"]],
                                   how="inner",
                                   left_index=True, right_index=True)
        presentation_df["onset"] += presentation_df.segment_onset
        presentation_df["offset"] += presentation_df.segment_onset

        # Remove segment annotations ; we want just the words.
        raw.annotations.delete(np.arange(n_segment_annotations))

        # Add annotations based on stim_df.
        for (segment_idx, sentence_idx, word_idx), row in presentation_df.iterrows():
            raw.annotations.append(row.onset, row.offset - row.onset,
                                   description=f"{sentence_idx}_{word_idx}")

        return raw, presentation_df

    def _load_stimulus(self, raw_text_path):
        """
        Load and compute natural language input for model. This includes predictive
        distributions over words at each timestep.
        """
        assert raw_text_path.exists()
        with raw_text_path.open("r") as f:
            raw_text = f.read()
        
        # TODO double check with preproc script
        tokens_flat = raw_text.strip().replace("\n", " ").split()

        # TODO pre tokenize? will need to redo aligner to match tokenization
        aligned_df = self._stim_df.copy()
        align_stimulus_fulltext(aligned_df, tokens_flat)

        p_word, candidate_ids = self._get_predictive_topk(encoding)
        

    def _preprocess(self, subject_id,
                    filter_window: Tuple[float, float]) -> mne.io.Raw:
        L.debug("Preprocessing subject %i", subject_id)
        raw = self._raw_data[subject_id]

        # Band-pass filter.
        raw = raw.filter(*filter_window)

        # Interpolate bad channels.
        raw = raw.interpolate_bads()

        return raw

    def get_dataset(self, subject_idx) -> RRDataset:
        raise NotImplementedError()

    def get_stimulus(self, subject_idx) -> Stimulus:
        raise NotImplementedError()