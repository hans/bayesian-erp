from __future__ import annotations
import dataclasses
from dataclasses import dataclass
import logging
import re
from typing import List, Optional, Callable, Dict, Tuple, Union

import numpy as np
import torch
from torch.nn.functional import pad
from torchtyping import TensorType
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm, trange
from typeguard import typechecked

from berp.typing import DIMS, is_probability, is_log_probability

L = logging.getLogger(__name__)

# Type variables
B, N_W, N_C, N_F, N_F_T, N_P, V_W = \
    DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_F_T, DIMS.N_P, DIMS.V_W
T, S = DIMS.T, DIMS.S

# Type aliases
Phoneme = str
intQ = Optional[Union[int, np.integer]]

def default_phonemizer(string) -> List[Phoneme]:
    return list(string)


@typechecked
@dataclass
class BerpDataset:
    """
    Defines a time series dataset for reindexing regression.

    The predictors are stored in two groups:

    1. `X_ts`: Time-series predictors, which are sampled at the same rate as `Y`.
    2. `X_variable`: Latent-onset predictors, `batch` many whose onset is to be inferred
       by the model.

    All tensors are padded on the N_P axis on the right to the maximum word length.
    """

    name: str

    sample_rate: int

    phonemes: List[str]
    """
    Phoneme vocabulary.
    """

    p_word: TensorType[B, N_C, is_log_probability]
    """
    Predictive distribution over expected candidate words at each time step,
    derived from a language model.
    """

    word_lengths: TensorType[B, int]
    """
    Length of ground-truth words in phonemes. Can be used to unpack padded
    ``N_P`` axes.
    """

    candidate_phonemes: TensorType[B, N_C, N_P, int]
    """
    Phoneme ID sequence for each word and alternate candidate set.
    """

    word_onsets: TensorType[B, float]
    """
    Onset of each word in seconds, relative to the start of the sequence.
    """

    phoneme_onsets: TensorType[B, N_P, float]
    """
    Onset of each phoneme within each word in seconds, relative to the start of
    the corresponding word. Column axis should be padded with 0s.
    """

    X_ts: TensorType[T, N_F_T, float]

    X_variable: TensorType[B, N_F, float]
    """
    Word-level features whose onset is to be determined by the model.
    """

    Y: TensorType[T, S, float]
    """
    Response data.
    """

    def __len__(self):
        return self.Y.shape[0]

    @property
    def n_samples(self):
        return len(self)

    @property
    def n_total_features(self):
        return self.X_variable.shape[1] + self.X_ts.shape[1]

    def __getitem__(self, key):
        """
        Extract a number of samples from the dataset.
        The resulting dataset has adjusted times to match the new sample start point.
        """
        if isinstance(key, slice):
            if key.step is not None:
                raise ValueError("Step size not supported.")

            start_sample = key.start or 0
            end_sample = key.stop or self.n_samples

            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate

            # Find which word indices should be retained for these time boundaries.
            # TODO add some slack on end? we don't want to include words exactly at the right boundary
            word_mask = (self.word_onsets >= start_time) & (self.word_onsets <= end_time)
            keep_word_indices = torch.where(word_mask)[0]

            # Subset word-level features.
            word_onsets = self.word_onsets[keep_word_indices]
            phoneme_onsets = self.phoneme_onsets[keep_word_indices]
            X_variable = self.X_variable[keep_word_indices]

            # Subtract onset data so that t=0 -> sample 0.
            word_onsets = word_onsets - start_time
            phoneme_onsets = phoneme_onsets - start_time

            ret = dataclasses.replace(self,
                name=f"{self.name}/slice:{start_sample}:{end_sample}",

                p_word=self.p_word[keep_word_indices],
                word_lengths=self.word_lengths[keep_word_indices],
                candidate_phonemes=self.candidate_phonemes[keep_word_indices],

                word_onsets=word_onsets,
                phoneme_onsets=phoneme_onsets,

                X_ts=self.X_ts[key],
                X_variable=X_variable,

                Y=self.Y[key],
            )

            return ret

        return super().__getitem__(key)

    def ensure_torch(self, dtype=torch.float32) -> BerpDataset:
        """
        Convert all tensors to torch tensors.
        """
        self.p_word = torch.as_tensor(self.p_word, dtype=torch.float32)
        self.word_lengths = torch.as_tensor(self.word_lengths)
        self.candidate_phonemes = torch.as_tensor(self.candidate_phonemes)
        self.word_onsets = torch.as_tensor(self.word_onsets, dtype=torch.float32)
        self.phoneme_onsets = torch.as_tensor(self.phoneme_onsets, dtype=torch.float32)
        self.X_ts = torch.as_tensor(self.X_ts, dtype=torch.float32)
        self.X_variable = torch.as_tensor(self.X_variable, dtype=torch.float32)
        self.Y = torch.as_tensor(self.Y, dtype=torch.float32)

        return self

    def subset_sensors(self, sensors: List[int]) -> BerpDataset:
        """
        Subset sensors in response variable. Returns a copy.
        """
        return dataclasses.replace(self, Y=self.Y[:, sensors])


class NestedBerpDataset(object):
    """
    Represents a grouped Berp dataset as a list of time series intervals.
    This makes the data amenable to cross validation by the standard sklearn API --
    can index via integer values.

    Each element in the resulting dataset corresponds to a fraction of an original
    subject's sub-dataset, `1/n_splits` large.
    """

    def __init__(self, datasets: List[BerpDataset], n_splits=2):
        # Shape checks. Everything but batch axis should match across
        # subjects. Batch axis should match within-subject between
        # X and Y.
        for dataset in datasets:
            assert dataset.X_ts.shape[1:] == datasets[0].X_ts.shape[1:]
            assert dataset.X_variable.shape[1:] == datasets[0].X_variable.shape[1:]
            assert dataset.Y.shape[1:] == datasets[0].Y.shape[1:]
            assert dataset.X_ts.shape[0] == dataset.Y.shape[0]

        self.datasets = datasets
        self.n_datasets = len(datasets)
        self.set_n_splits(n_splits)

    def set_n_splits(self, n_splits):
        self.n_splits = n_splits

        # Maps integer indices on this dataset into slices of individual sub-datasets.
        self.flat_idxs: List[Tuple[int, slice]] = []
        for i, dataset in enumerate(self.datasets):
            split_size = int(np.ceil(len(dataset) / self.n_splits))
            for split_offset in range(0, len(dataset), split_size):
                self.flat_idxs.append((i, slice(split_offset, split_offset + split_size)))

    @property
    def shape(self):
        # Define shape property so that sklearn indexing thinks we're an ndarray,
        # and will index with an ndarray of indices rather than scalars+concatenate.
        # Then we can make sure the output is still a NestedBerpDataset :)
        return (len(self),)

    # TODO will be super slow to always typecheck. remove once we know this works
    @typechecked
    def __getitem__(self, key: Union[int, np.integer, np.ndarray]
                    ) -> Union[BerpDataset, NestedBerpDataset]:
        if isinstance(key, (int, np.integer)):
            dataset, split = self.flat_idxs[key]
            return self.datasets[dataset][split]
        elif isinstance(key, np.ndarray):
            return NestedBerpDataset([self[i] for i in key], n_splits=self.n_splits)
        else:
            raise NotImplementedError(f"Unsupported key type {type(key)}")

    def __len__(self):
        return len(self.flat_idxs)

    def iter_datasets(self):
        return iter(self.datasets)

    def subset_sensors(self, sensors: List[int]) -> NestedBerpDataset:
        """
        Subset sensors in response variable. Returns a copy.
        """
        return NestedBerpDataset([dataset.subset_sensors(sensors) for dataset in self.datasets],
                                 n_splits=self.n_splits)


@typechecked
@dataclass
class NaturalLanguageStimulus:
    """
    Word-level stimulus representation. This is the output of an alignment
    procedure in which token-level predictive prior distributions are aggregated
    to word-level predictive prior distributions, and then also decomposed into
    phoneme-level representations.

    TODO improve description
    """

    phonemes: List[Phoneme]
    """
    Phoneme vocabulary.
    """

    pad_phoneme_id: int
    """
    Index of padding phoneme in phoneme vocabulary.
    """

    word_ids: TensorType[N_W, torch.long]
    """
    For each row in the dataset, the ID of the corresponding word in the
    source corpus.
    """

    word_lengths: TensorType[N_W, int]
    """
    Length of each ground-truth word in the dataset (in number of phonemes).
    """

    word_features: TensorType[N_W, N_F, float]
    """
    Arbitrary word-level features.
    """

    p_word: TensorType[N_W, N_C, torch.float, is_log_probability]
    """
    Prior predictive distribution over words at each timestep. Each
    row is a proper log-probability distribution.
    """    

    candidate_phonemes: TensorType[N_W, N_C, N_P, torch.long]
    """
    For each candidate in each prior predictive, the corresponding
    phoneme sequence. Sequences are padded with `pad_phoneme_id`.
    """

    @property
    def word_surprisals(self) -> TensorType[N_W, torch.float]:
        """
        Get surprisals of ground-truth words (in bits; log-2).
        """
        return -torch.log2(self.p_word[:, 0])


class NaturalLanguageStimulusProcessor(object):
    """
    Processes a natural-language stimulus for use in the Berp models.

    Computes incremental token-level prior predictive distributions and
    aggregates these at the word level. Also represents these prior
    predictives in terms of word phonemes.
    TODO find a better general description :)
    """

    def __init__(self,
                 phonemes: List[Phoneme],
                 hf_model: str,
                 phonemizer: Optional[Callable[[str], List[Phoneme]]] = None,
                 num_candidates: int = 10,
                 batch_size=8,
                 disallowed_re=r"[^a-z]",
                 pad_phoneme="_"):

        self.phonemes = np.array(phonemes)
        self.phoneme2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        assert pad_phoneme in self.phoneme2idx, "Provide pad phoneme please"
        self.pad_phoneme = pad_phoneme
        self.pad_phoneme_id = self.phoneme2idx[pad_phoneme]
        self.phonemizer = phonemizer or default_phonemizer

        self.num_candidates = num_candidates

        self.batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self._model = AutoModelForCausalLM.from_pretrained(hf_model)

        if self._tokenizer.pad_token is None:
            logging.warn("Tokenizer is missing pad token; using EOS token " +
                         self._tokenizer.eos_token)
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Pre-compute mask of allowed tokens (== which have content after cleaning)
        self.disallowed_re = disallowed_re
        self.vocab_mask = torch.ones(self._tokenizer.vocab_size, dtype=torch.bool)
        for token, idx in self._tokenizer.vocab.items():  # type: ignore
            if self._clean_word(token) == "":
                self.vocab_mask[idx] = False

    def _clean_word(self, word: str) -> str:
        return re.sub(self.disallowed_re, "", word.lower())

    def _clean_sentences(self, sentences: List[str]) -> List[str]:
        return [re.sub(r"[^a-z\s]", "", sentence.lower()).strip()
                for sentence in sentences]

    def get_predictive_topk(self, input_ids: TensorType[B, "n_times", torch.long],
                            ) -> Tuple[TensorType[B, N_W, N_C, is_probability],
                                       TensorType[B, N_W, N_C, int]]:
        """
        For each sentence and each token, retrieve a top-K predictive
        distribution over the next word, along with a list of the top K
        candidate token IDs.

        Args:
            batch_tok: Preprocessed batch of sentences

        Returns:
            p_token: log-probability of each candidate token
            candidate_ids:
        """

        with torch.no_grad():
            model_outputs = self._model(input_ids)[0].log_softmax(dim=2)

        # Ignore disallowed tokens.
        model_outputs[:, :, ~self.vocab_mask] = -torch.inf
        # Sample top N candidates per item + predicted word.
        # NB t=0 here prior to reindexing corresponds to model output after consuming 1st token
        _, candidate_ids = torch.topk(model_outputs[:, :-1], k=self.num_candidates,
                                      dim=2)

        # NB we start at t=1 because that is where predictions start.
        gt_token_ids = input_ids[:, 1:]

        # Prepend ground truth token as top candidate. If it already exists elsewhere,
        # drop it; otherwise, drop the lowest-probability candidate.
        candidate_ids = torch.cat([gt_token_ids.unsqueeze(2), candidate_ids], dim=2)
        existing_candidate_indices = (candidate_ids == gt_token_ids[:, :, None]).nonzero()
        for x, y, z in existing_candidate_indices:
            if z == 0:
                # Expected.
                continue
            elif z == candidate_ids.shape[2]:
                # Will be dropped anyway. Nothing to do.
                continue

            # Drop ground truth ID, shift left, and add a dummy at end.
            candidate_ids[x, y] = torch.cat(
                [candidate_ids[x, y, :z],
                    candidate_ids[x, y, z+1:],
                    torch.tensor([0])]
            )

        # Drop N+1th.
        candidate_ids = candidate_ids[:, :, :-1]

        # Reindex and normalize predictive distribution.
        p_token = torch.gather(
            model_outputs,
            dim=2,
            index=candidate_ids
        )

        # Renormalize over candidate axis and return to logspace.
        p_token = (p_token - p_token.max(dim=2, keepdim=True)[0]).exp()
        p_token /= p_token.sum(dim=2, keepdim=True)
        p_token = p_token.log()

        return p_token, candidate_ids

    def get_candidate_phonemes(self, candidate_ids: TensorType[B, N_W, N_C, int],
                               max_num_phonemes: int,
                               ground_truth_phonemes: Optional[List[List[Phoneme]]] = None
                               ) -> Tuple[TensorType[B, N_W, N_C, N_P, torch.long],
                                          TensorType[B, N_W, int]]:
        """
        For a given batch of candidate words, compute a full tensor of phonemes for
        each candidate word, padded to length ``max_num_phonemes`` on the last
        dimension.

        Args:
            ground_truth_phonemes: Phoneme sequences for the ground-truth
                words. If not provided, standard phonemizer is used.

        Returns:
            candidate_phonemes:
            word_lengths: lengths of ground truth word
        """
        candidate_tokens = self._tokenizer.convert_ids_to_tokens(
            candidate_ids.reshape(-1).tolist())

        # Convert tokens to phoneme sequences.
        candidate_phoneme_seqs = []
        for i, tok in enumerate(candidate_tokens):
            tok = self._clean_word(tok)

            # If we are looking at a ground-truth token and there is a reference
            # phonemization, use that. Otherwise automate.
            if ground_truth_phonemes is not None and i % candidate_ids.shape[2] == 0:
                if i // candidate_ids.shape[2] >= len(ground_truth_phonemes):
                    # This is probably a padding token. All good, let it pass
                    phoneme_seq = []
                else:
                    phoneme_seq = ground_truth_phonemes[i // candidate_ids.shape[2]]
            else:
                phoneme_seq = self.phonemizer(tok)

            # NB phoneme_seq may be None. catch that.
            candidate_phoneme_seqs.append(phoneme_seq or [])
        
        word_lengths = torch.tensor([len(tok) for tok in candidate_phoneme_seqs]) \
            .reshape(*candidate_ids.shape)[:, :, 0]

        candidate_phonemes = torch.stack([
            pad(torch.tensor([self.phoneme2idx[p] for p in phoneme_seq]),
                (0, max_num_phonemes - len(phoneme_seq)),
                value=self.pad_phoneme_id)
            for phoneme_seq in candidate_phoneme_seqs
        ])
        candidate_phonemes = candidate_phonemes.reshape(
            *candidate_ids.shape, max_num_phonemes).long()

        return candidate_phonemes, word_lengths

    def __call__(self, tokens: List[str],
                 token_mask: List[bool],
                 word_to_token: Dict[int, List[int]],
                 word_features: Dict[int, torch.Tensor],
                 ground_truth_phonemes: Optional[Dict[int, List[Phoneme]]] = None,
                 ) -> NaturalLanguageStimulus:
        """
        Args:
            token_mask: While all tokens need to be used to compute surprisals,
                only some tokens will have corresponding neural data. This boolean
                mask specifies which tokens should be included in the returned
                dataset.
            word_features: Tensor associating each word ID with a set of features.
            ground_truth_phonemes: Phoneme sequences for the ground-truth words.
        """
        assert len(tokens) == len(token_mask)
        assert len(tokens) >= len(word_to_token), \
            str((len(tokens), len(word_to_token)))
        assert len(word_to_token) == len(ground_truth_phonemes), \
            str((len(word_to_token), len(ground_truth_phonemes)))
        assert len(word_to_token) == len(word_features)

        # By default, map tokens to word ID -1. This helps us easily catch tokens
        # that should be dropped.
        nonword_id = -1
        token_to_word = nonword_id * torch.ones(len(tokens)).long()
        for word_id, token_idxs in word_to_token.items():
            for token_idx in token_idxs:
                token_to_word[token_idx] = word_id
        # Words are not contiguous necessarily -- prepare to convert IDs to indices.
        word_id_to_idx = {word_id: idx for idx, word_id in enumerate(sorted(word_to_token.keys()))}
        
        # Split text into distinct inputs based on model maxlen
        # and then batch.
        # TODO overlap for better contextual predictions
        token_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        max_len = 32  # DEV self._model.config.n_positions
        token_inputs = [token_ids[i:i+max_len]
                        for i in range(0, len(token_ids), max_len)]

        token_mask = torch.tensor(token_mask)

        # Pad to max_len.
        token_inputs = torch.stack([
            pad(torch.tensor(tok_ids),
                (0, max_len - len(tok_ids)),
                value=self._tokenizer.pad_token_id)
            for tok_ids in token_inputs
        ])

        # Pre-compute maximum number of phonemes.
        # TODO use ground truth phonemes here
        max_num_phonemes = max(len(self.phonemizer(tok)) for tok in tokens)

        # NB some words may never end up affecting a sample (e.g. the first word in the
        # batches constructed here). So we'll also need to keep a mask for that and do
        # final masking at the end.
        num_words = len(word_to_token)
        touched_words = torch.zeros(num_words).bool()

        i = 0
        word_lengths = torch.zeros(num_words, dtype=torch.long)
        p_word = torch.zeros((num_words, self.num_candidates), dtype=torch.float)
        candidate_phonemes = torch.zeros((num_words, self.num_candidates, max_num_phonemes), dtype=torch.long)
        # Track the word ID that produced each sample.
        word_ids = torch.zeros(num_words, dtype=torch.long)
        for i in trange(0, len(token_inputs), self.batch_size):
            batch = token_inputs[i:i+self.batch_size]

            # Keep track of which token indices are in the batch.
            # Account for the fact that the batch may not be maximum rows and may be padded
            # in the final row.
            batch_token_idxs = torch.arange(i * max_len, min(len(tokens) - 1, (i + self.batch_size) * max_len))

            # TODO for BPE models, we really should keep predicting each candidate until we
            # reach a BPE boundary. Otherwise the candidates are likely to be subwords,
            # while the ground truth word is a longer full word. This will bork the
            # incremental prediction dynamics in ways I don't want to have to figure out.
            # Better to implement this correctly, which will be expensive. For every
            # ground-truth word, for each candidate, keep predicting on a beam until
            # we reach a token with BPE separator as the argmax output.
            batch_p_token, batch_candidate_token_ids = self.get_predictive_topk(batch)

            # Aggregate token-level outputs at the word level.
            # HACK: For now, we'll just take the first token associated with each word.
            # In the long run, we should intelligently aggregate the subword outputs.
            # This only makes sense once we also are doing more intelligent candidate
            # retrieval (see previous long comment).

            # TODO currently redundant with token mask calc in produce_dataset.py
            batch_word_ids = token_to_word[batch_token_idxs]
            last_word_id = None
            drop_subword_mask = torch.ones(len(batch_token_idxs), dtype=torch.bool)
            for i, word_id in enumerate(batch_word_ids):
                if last_word_id == word_id and word_id != 0:
                    # HACK: We're seeing the subword of an already observed word. Drop it.
                    drop_subword_mask[i] = False
                
                last_word_id = word_id

            # Extract relevant token masks and combine with subword mask.
            batch_mask = token_mask[batch_token_idxs] & drop_subword_mask
            # Mask out any tokens which don't correspond to a word.
            batch_mask = batch_mask & (batch_word_ids != nonword_id)
            # TODO check why this fails sometimes
            # assert batch_mask.sum() == len(set(batch_word_ids.numpy()) - {0})
            batch_word_ids_pad = batch_word_ids[:]
            if batch_mask.shape[0] % max_len > 0:
                # Pad so that we reach a multiple of max_len.
                batch_mask = pad(batch_mask, (0, max_len - batch_mask.shape[0] % max_len), value=False)
                batch_word_ids_pad = pad(batch_word_ids, (0, max_len - batch_word_ids.shape[0] % max_len),
                                         value=nonword_id)
            batch_mask = batch_mask.reshape((batch.shape[0], max_len))
            # Ignore mask on first token in each sample, since other methods won't
            # have outputs for this token.
            batch_mask = batch_mask[:, 1:]
            batch_word_ids_pad = batch_word_ids_pad.reshape((batch.shape[0], max_len))[:, 1:]

            batch_ground_truth_phonemes = None
            if ground_truth_phonemes is not None:
                batch_ground_truth_phonemes = [ground_truth_phonemes.get(word_id.item(), None)
                                               for word_id in batch_word_ids]
            batch_candidate_phonemes, batch_word_lengths = self.get_candidate_phonemes(
                batch_candidate_token_ids, max_num_phonemes, batch_ground_truth_phonemes)

            # Also ignore words with zero length.
            batch_mask = batch_mask & (batch_word_lengths > 0)

            # Drop rows which correspond to non-words. Flattens first two axes in the process.
            batch_word_lengths = batch_word_lengths[batch_mask]
            batch_p_token = batch_p_token[batch_mask]
            batch_candidate_phonemes = batch_candidate_phonemes[batch_mask]

            # Track which word idxs were retained.
            batch_retained_word_ids = batch_word_ids_pad[batch_mask].flatten()

            batch_num_samples = batch_candidate_phonemes.shape[0]
            assert batch_num_samples == len(batch_retained_word_ids)
            assert batch_p_token.shape[0] == batch_num_samples
            assert batch_word_lengths.shape[0] == batch_num_samples

            # Get target indices in contiguous output arrays. (NB word IDs are not
            # contiguous.)
            batch_word_idxs = torch.tensor([word_id_to_idx[word_id.item()]
                                            for word_id in batch_retained_word_ids])

            word_lengths[batch_word_idxs] = batch_word_lengths
            p_word[batch_word_idxs] = batch_p_token
            candidate_phonemes[batch_word_idxs] = batch_candidate_phonemes
            word_ids[batch_word_idxs] = batch_retained_word_ids
            touched_words[batch_word_idxs] = True

        # Sanity check
        assert (p_word[~touched_words] == 0).all()

        # Finally, filter out words which never had data computed.
        word_ids = word_ids[touched_words]
        word_lengths = word_lengths[touched_words]
        p_word = p_word[touched_words]
        candidate_phonemes = candidate_phonemes[touched_words]

        # Reindex word-level features.
        if word_features is not None:
            word_features = torch.stack([word_features[word_id.item()]
                                         for word_id in word_ids])

        return NaturalLanguageStimulus(
            phonemes=self.phonemes,
            pad_phoneme_id=self.pad_phoneme_id,
            word_ids=word_ids,
            word_lengths=word_lengths,
            word_features=word_features,
            p_word=p_word,
            candidate_phonemes=candidate_phonemes,
        )

        # word_surprisals = -p_word[:, 0].log() / np.log(2)

        # # TODO include in return
        # print(word_ids)

        # phoneme_onsets, phoneme_onsets_global, word_onsets = \
        #     self.sample_stream(word_lengths, max_num_phonemes)

        # return Stimulus(word_lengths, phoneme_onsets, phoneme_onsets_global, word_onsets,
        #                 word_surprisals, p_word.log(), candidate_phonemes)