import itertools
import logging
import re
from typing import List, NamedTuple, Tuple

import numpy as np
import pyro.distributions as dist
import torch
from torch.nn.functional import pad
from torchtyping import TensorType
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from transformers.tokenization_utils_base import BatchEncoding
from tqdm.auto import tqdm

from berp.typing import DIMS, is_probability, is_log_probability


L = logging.getLogger(__name__)

# Type variables
B, N_W, N_C, N_F, N_P, V_W = DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_P, DIMS.V_W
T, S = DIMS.T, DIMS.S


class Stimulus(NamedTuple):

    word_lengths: TensorType[N_W, torch.int]
    phoneme_onsets: TensorType[N_W, N_P, torch.float]
    phoneme_onsets_global: TensorType[N_W, N_P, torch.float]
    word_onsets: TensorType[N_W, torch.float]
    word_surprisals: TensorType[N_W, torch.float]
    p_word: TensorType[N_W, N_C, torch.float, is_log_probability]
    candidate_phonemes: TensorType[N_W, N_C, N_P, torch.long]


class StimulusGenerator(object):

    def __init__(self,
                 phon_delay_range: Tuple[float, float] = (0.04, 0.1),
                 word_delay_range: Tuple[float, float] = (0.01, 0.1)):
        self.phon_delay_range = phon_delay_range
        self.word_delay_range = word_delay_range

        self.first_onset = 1.0  # TODO magic to make epoching not break

    def __call__(self, *args, **kwargs) -> Stimulus:
        raise NotImplementedError()

    def sample_stream(self, word_lengths: TensorType[N_W, int],
                      max_num_phonemes: int,
                      ) -> Tuple[TensorType[N_W, N_P, float],
                                 TensorType[N_W, N_P, float],
                                 TensorType[N_W, float]]:
        num_words = len(word_lengths)

        phoneme_onsets = rand_unif(*self.phon_delay_range, num_words, max_num_phonemes)
        phoneme_onsets[:, 0] = 0.
        phoneme_onsets[torch.arange(max_num_phonemes) >= word_lengths.unsqueeze(1)] = 0.
        phoneme_onsets = phoneme_onsets.cumsum(1)
        word_delays = rand_unif(*self.word_delay_range, num_words)
        word_onsets = (torch.cat([torch.tensor([self.first_onset]),
                                  phoneme_onsets[:-1, -1]])
                                + word_delays).cumsum(0)
        
        # Make phoneme_onsets global (not relative to word onset).
        phoneme_onsets_global = phoneme_onsets + word_onsets.view(-1, 1)

        return phoneme_onsets, phoneme_onsets_global, word_onsets


def rand_unif(low, high, *shape) -> torch.Tensor:
    return torch.rand(*shape) * (high - low) + low


class RandomStimulusGenerator(StimulusGenerator):

    def __init__(self,
                 num_words: int = 100,
                 num_candidates: int = 10,
                 num_phonemes: int = 5,
                 phoneme_voc_size: int = 18,
                 word_surprisal_params: Tuple[float, float] = (1., 0.5),
                 **kwargs):
        super().__init__(**kwargs)

        self.num_words = num_words
        self.num_candidates = num_candidates
        self.num_phonemes = num_phonemes
        self.word_surprisal_params = word_surprisal_params

        # Generate phoneme set
        self.phonemes = np.array(list("abcdefghijklmnopqrstuvwxyz"[:phoneme_voc_size - 1] + "_"))
        self.phoneme2idx = {p: idx for idx, p in enumerate(self.phonemes)}

    def __call__(self) -> Stimulus:
        word_lengths = 1 + dist.Binomial(self.num_phonemes - 1, 0.5) \
            .sample((self.num_words, self.num_candidates)).long()  # type: ignore
        gt_word_lengths = word_lengths[:, 0]

        candidate_phonemes = torch.randint(0, len(self.phonemes) - 2,
                                          (self.num_words,
                                           self.num_candidates,
                                           self.num_phonemes))
        # Use padding token when word length exceeded.
        # TODO can have candidates with different lengths
        pad_idx = self.phoneme2idx["_"]
        pad_mask = (torch.arange(self.num_phonemes) >= word_lengths[:, :, None])
        candidate_phonemes[pad_mask] = pad_idx

        phoneme_onsets, phoneme_onsets_global, word_onsets = \
            self.sample_stream(gt_word_lengths, self.num_phonemes)

        word_surprisals: torch.Tensor = dist.LogNormal(*self.word_surprisal_params) \
            .sample((self.num_words,))  # type: ignore

        # Calculate p_word using surprisal; allocate remainder randomly
        p_gt_word = (-word_surprisals).exp()
        remainder = 1 - p_gt_word
        p_candidates = (remainder / (self.num_candidates - 1)).view(-1, 1) \
            * torch.ones(self.num_words, self.num_candidates - 1)
        p_word = torch.cat([p_gt_word.view(-1, 1), p_candidates], dim=1) \
            .log()

        return Stimulus(gt_word_lengths, phoneme_onsets, phoneme_onsets_global,
                        word_onsets, word_surprisals, p_word, candidate_phonemes)


class NaturalLanguageStimulusGenerator(StimulusGenerator):

    def __init__(self,
                 phonemes: List[str],
                 hf_model: str,
                 num_candidates: int = 10,
                 batch_size=8,
                 **kwargs):
        super().__init__(**kwargs)

        self.phonemes = np.array(phonemes)
        self.phoneme2idx = {p: idx for idx, p in enumerate(self.phonemes)}

        self.num_candidates = num_candidates

        self.batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self._model = AutoModelForCausalLM.from_pretrained(hf_model)

        if self._tokenizer.pad_token is None:
            logging.warn("Tokenizer is missing pad token; using EOS token " +
                         self._tokenizer.eos_token)
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Pre-compute mask of allowed tokens (== which have content after cleaning)
        self.vocab_mask = torch.ones(self._tokenizer.vocab_size, dtype=torch.bool)
        for token, idx in self._tokenizer.vocab.items():  # type: ignore
            if self._clean_word(token) == "":
                self.vocab_mask[idx] = False

    def _clean_word(self, word: str) -> str:
        return re.sub(r"[^a-z]", "", word.lower())

    def _clean_sentences(self, sentences: List[str]) -> List[str]:
        return [re.sub(r"[^a-z\s]", "", sentence.lower()).strip()
                for sentence in sentences]

    def get_predictive_topk(self, batch_tok: BatchEncoding,
                            ) -> Tuple[TensorType[B, N_W, N_C, is_probability],
                                       TensorType[B, N_W, N_C, int]]:
        """
        For each sentence and each token, retrieve a top-K predictive
        distribution over the next word, along with a list of the top K
        candidate token IDs.

        Args:
            batch_tok: Preprocessed batch of sentences

        Returns:
            p_word:
            candidate_ids:
        """

        with torch.no_grad():
            model_outputs = self._model(batch_tok["input_ids"])[0].log_softmax(dim=2)

        # Ignore disallowed tokens.
        model_outputs[:, :, ~self.vocab_mask] = -torch.inf
        # Sample top N candidates per item + predicted word.
        # NB t=0 here prior to reindexing corresponds to model output after consuming 1st token
        _, candidate_ids = torch.topk(model_outputs[:, :-1], k=self.num_candidates,
                                      dim=2)

        # NB we start at t=1 because that is where predictions start.
        gt_token_ids = batch_tok["input_ids"][:, 1:]  # type: ignore

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
        p_word = torch.gather(
            model_outputs,
            dim=2,
            index=candidate_ids
        )

        # Renormalize over candidate axis.
        p_word = (p_word - p_word.max(dim=2, keepdim=True)[0]).exp()
        p_word /= p_word.sum(dim=2, keepdim=True)

        return p_word, candidate_ids

    def get_candidate_phonemes(self, candidate_ids: TensorType[B, N_W, N_C, int],
                               max_num_phonemes: int
                               ) -> Tuple[TensorType[B, N_W, N_C, N_P, torch.long],
                                          TensorType[B, N_W, int]]:
        """
        For a given batch of candidate words, compute a full tensor of phonemes for
        each candidate word, padded to length ``max_num_phonemes`` on the last
        dimension.

        Returns:
            candidate_phonemes:
            word_lengths: lengths of ground truth word
        """
        candidate_tokens = self._tokenizer.convert_ids_to_tokens(
            candidate_ids.reshape(-1).tolist())
        candidate_tokens = [self._clean_word(tok) for tok in candidate_tokens]

        word_lengths = torch.tensor([len(tok) for tok in candidate_tokens]) \
            .reshape(*candidate_ids.shape)[:, :, 0]

        candidate_phonemes = torch.stack([
            pad(torch.tensor([self.phoneme2idx[p] for p in token]),
                (0, max_num_phonemes - len(token)),
                value=self.phoneme2idx["_"])
            for token in candidate_tokens
        ])
        candidate_phonemes = candidate_phonemes.reshape(
            *candidate_ids.shape, max_num_phonemes).long()

        return candidate_phonemes, word_lengths

    def __call__(self, sentences: List[str]) -> Stimulus:
        sentences = self._clean_sentences(sentences)

        # Pre-tokenize sentences and split into batches.
        # Also pre-compute relevant size parameters:
        # - total number of words, collapsing over sentences
        # - maximum number of phonemes (in ground-truth word -- we don't care about
        #   truncating candidates, since we'll never be indexing past the ground-truth
        #   final phoneme -- there are no defined time onsets for non-ground-truth
        #   phonemes anyway)
        batches = []
        num_words = 0
        max_num_phonemes = 0
        for i in range(0, len(sentences), self.batch_size):
            batch_sentences = sentences[i:i + self.batch_size]
            batch = self._tokenizer(batch_sentences, padding=True, 
                                    return_tensors="pt", truncation=True,
                                    max_length=self._model.config.max_length)

            # TODO refactor. The function of this code is to heuristically compute how many
            # words in total we have in the final output matrices. To do so, we re-code
            # all of the logic in other methods of this class here. That's not good for
            # sustainability. Ideally we'd use exactly the core logic to compute the number
            # of words as well. In the worst case, could just do a dry run of the whole data
            # prep in order to compute the number of words.
            for encoding in batch.encodings:  # type: ignore
                for i, token_word_id in enumerate(encoding.word_ids):  # type: ignore
                    # Skip first token of each sentence -- no predictions.
                    if i == 0:
                        continue

                    # Count only tokens which have corresponding words in input.
                    # Tokens for which this is not the case have value `None` in
                    # encoding.word_ids
                    if token_word_id is None:
                        continue

                    cleaned_word = self._clean_word(encoding.tokens[i])  # type: ignore
                    # Will be dropped by masking later on.
                    # TODO would be great to not have coupled separate parts of code
                    if len(cleaned_word) == 0:
                        continue

                    num_words += 1
                    max_num_phonemes = max(max_num_phonemes, len(cleaned_word))

            batches.append(batch)

        i = 0
        word_lengths = torch.zeros(num_words, dtype=torch.long)
        p_word = torch.zeros((num_words, self.num_candidates), dtype=torch.float)
        candidate_phonemes = torch.zeros((num_words, self.num_candidates, max_num_phonemes), dtype=torch.long)
        for batch in tqdm(batches):
            batch_p_word, batch_candidate_ids = self.get_predictive_topk(batch)
            batch_candidate_phonemes, batch_word_lengths = self.get_candidate_phonemes(
                batch_candidate_ids, max_num_phonemes)

            # Compute a mask which is `True` iff a word within item
            # corresponds to a real word in the input string (i.e. not special token/padding).
            # Skip BOS token since we don't have data on that in the above variables.
            word_mask: TensorType[B, N_W, bool] = torch.tensor([
                [word_id is not None for word_id in encoding.word_ids[1:]]
                for encoding in batch.encodings
            ])
            # Also ignore words with zero length.
            word_mask = word_mask & (batch_word_lengths > 0)

            # Drop rows which correspond to non-words. Flattens first two axes in the process.
            # TODO track retained indices -- will be important when we have non-random
            # onset data!
            batch_word_lengths = batch_word_lengths[word_mask]
            batch_p_word = batch_p_word[word_mask]
            batch_candidate_phonemes = batch_candidate_phonemes[word_mask]

            batch_num_samples = batch_candidate_phonemes.shape[0]
            assert batch_p_word.shape[0] == batch_num_samples
            start, end = i, i + batch_num_samples

            word_lengths[start:end] = batch_word_lengths
            p_word[start:end] = batch_p_word
            candidate_phonemes[start:end] = batch_candidate_phonemes

            i += batch_num_samples

        word_surprisals = -p_word[:, 0].log() / np.log(2)

        phoneme_onsets, phoneme_onsets_global, word_onsets = \
            self.sample_stream(word_lengths, max_num_phonemes)

        return Stimulus(word_lengths, phoneme_onsets, phoneme_onsets_global, word_onsets,
                        word_surprisals, p_word.log(), candidate_phonemes)