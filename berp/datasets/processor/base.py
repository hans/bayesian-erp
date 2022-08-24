from dataclasses import dataclass
import logging
import re
from typing import *

import numpy as np
import torch
from torch.nn.functional import pad
from torchtyping import TensorType  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm, trange
from typeguard import typechecked

from berp.typing import DIMS, is_probability, is_log_probability, is_positive

L = logging.getLogger(__name__)


# Type variables
B, N_W, N_C, N_F, N_F_T, N_P, V_W = \
    DIMS.B, DIMS.N_W, DIMS.N_C, DIMS.N_F, DIMS.N_F_T, DIMS.N_P, DIMS.V_W
T, S = DIMS.T, DIMS.S

# Type aliases
Phoneme = str

def default_phonemizer(string) -> List[Phoneme]:
    return list(string)


_model_cache = {}
_tokenizer_cache = {}


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
    row is a proper log-e-probability distribution.
    """

    candidate_phonemes: TensorType[N_W, N_C, N_P, torch.long]
    """
    For each candidate in each prior predictive, the corresponding
    phoneme sequence. Sequences are padded with `pad_phoneme_id`.
    """

    @property
    def word_surprisals(self) -> TensorType[N_W, torch.float, is_positive]:
        """
        Get surprisals of ground-truth words (in bits; log-2).
        """
        return -self.p_word[:, 0] / np.log(2)


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

        if hf_model in _model_cache:
            self._model = _model_cache[hf_model]
        else:
            self._model = AutoModelForCausalLM.from_pretrained(hf_model)
            _model_cache[hf_model] = self._model
        
        if hf_model in _tokenizer_cache:
            self._tokenizer = _tokenizer_cache[hf_model]
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(hf_model)
            _tokenizer_cache[hf_model] = self._tokenizer

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

    def get_predictive_topk_words(self, input_ids: TensorType[B, "n_times", torch.long],
                                  word_ids: TensorType[B, "n_times", torch.long],
                                  ) -> Tuple[TensorType[N_W, torch.long],
                                             TensorType[N_W, N_C, is_probability],
                                             List[List[str]]]:
        """
        Compute a top-k predictive distribution over words (NB: not tokens!)
        for the given input token sequence.
        
        This involves computing a top-k predictive distribution over tokens and then
        aggregating.
        TODO will eventually involve beam search.

        Args:
            input_ids: Padded input token matrix for language model of token idxs.
            word_ids: Mapping for each token to the word ID it belongs to.

        Returns:
            ret_word_ids: word ID for each row in the return value.
            p_word: probability of each candidate word. columns are proper
                probability distributions.
            word_strs: string representation of each candidate word.
        """

        # Compute predictive distribution over tokens.
        p_token, candidate_token_ids = self.get_predictive_topk(input_ids)
        # Drop first column of word_ids since these will be lost in the predictive outputs.
        word_ids = word_ids[:, 1:]

        # Flatten along batch+time axis.
        p_token = p_token.view(-1, p_token.shape[-1])
        candidate_token_ids = candidate_token_ids.view(-1, candidate_token_ids.shape[-1])
        word_ids = word_ids.reshape(-1)

        ret_word_ids, p_word, word_strs = [], [], []
        for word_id in set(word_ids.numpy()):
            # Get all elements that correspond to this word.
            word_mask = word_ids == word_id
            word_p_token = p_token[word_mask]
            word_candidate_token_ids = candidate_token_ids[word_mask]

            # Aggregate.
            word_p_word = word_p_token.sum(dim=0)
            # DUMB just take the first one.
            # Could get the GT string though.
            word_candidate_token_ids = word_candidate_token_ids[0]
            word_candidate_strs = self._tokenizer.convert_ids_to_tokens(word_candidate_token_ids)

            ret_word_ids.append(word_id)
            p_word.append(word_p_word)
            word_strs.append(word_candidate_strs)

        return torch.tensor(ret_word_ids), torch.stack(p_word), word_strs


    def get_candidate_phonemes(self, candidate_strs: List[List[str]],
                               max_num_phonemes: int,
                               ground_truth_phonemes: Optional[List[List[Phoneme]]] = None
                               ) -> Tuple[TensorType[N_W, N_C, N_P, torch.long],
                                          TensorType[N_W, int]]:
        """
        For a given batch of candidate words, compute a full tensor of phonemes for
        each candidate word, padded to length ``max_num_phonemes`` on the last
        dimension.

        Args:
            candidate_strs: Candidate word strings. Each sublist's first element
                should correspond to the ground-truth word.
            ground_truth_phonemes: Phoneme sequences for the ground-truth
                words. If not provided, standard phonemizer is used.

        Returns:
            candidate_phonemes:
            word_lengths: lengths of ground truth word
        """

        # TODO integration test: for dummy identity phonemizer, make sure things end up
        # in the right place.
        if ground_truth_phonemes is not None:
            assert len(ground_truth_phonemes) == len(candidate_strs)

        # Convert tokens to padded phoneme sequences.
        candidate_phoneme_seqs, word_lengths = [], []
        for i, candidates_i in enumerate(candidate_strs):
            phoneme_seqs_i = []
            for j, candidate_str in enumerate(candidates_i):
                # If this is the ground truth word and there is a reference
                # phonemization, use that.
                if j == 0 and ground_truth_phonemes is not None:
                    phonemes = ground_truth_phonemes[i]
                else:
                    phonemes = self.phonemizer(self._clean_word(candidate_str))

                # Store ground truth phoneme sequence length.
                if j == 0:
                    word_lengths.append(len(phonemes))

                # Convert to IDs, pad, and append.
                phoneme_ids = [self.phoneme2idx[p] for p in phonemes]
                phoneme_ids += [self.pad_phoneme_id] * (max_num_phonemes - len(phoneme_ids))

                phoneme_seqs_i.append(phoneme_ids)

            candidate_phoneme_seqs.append(phoneme_seqs_i)
        
        candidate_phonemes = torch.tensor(candidate_phoneme_seqs).long()

        return candidate_phonemes, torch.tensor(word_lengths)

    def __call__(self, tokens: List[str],
                 word_to_token: Dict[int, List[int]],
                 word_features: Dict[int, torch.Tensor],
                 ground_truth_phonemes: Optional[Dict[int, List[Phoneme]]] = None,
                 ) -> NaturalLanguageStimulus:
        """
        Args:
            word_to_token: Mapping from word ID to list of token indices. NB that
                some tokens will have no corresponding words.
            word_features: Tensor associating each word ID with a set of features.
            ground_truth_phonemes: Phoneme sequences for the ground-truth words.
        """
        assert len(tokens) >= len(word_to_token), \
            str((len(tokens), len(word_to_token)))
        # Token idxs specified in word_to_token should be within range
        assert all(token_idx >= 0 and token_idx < len(tokens)
                   for token_idxs in word_to_token.values()
                   for token_idx in token_idxs), \
            "word_to_token contains token IDs out of range"
        if ground_truth_phonemes is not None:
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

        # Pad to max_len.
        token_inputs_tensor = torch.stack([
            pad(torch.tensor(tok_ids),
                (0, max_len - len(tok_ids)),
                value=self._tokenizer.pad_token_id)
            for tok_ids in token_inputs
        ])

        # Pre-compute maximum number of phonemes in ground truth words.
        if ground_truth_phonemes is not None:
            max_num_phonemes = max(len(phonemes) for phonemes in ground_truth_phonemes.values())
        else:
            max_num_phonemes = max(len(self.phonemizer(tok)) for tok in tokens)

        # NB some words may never end up affecting a sample (e.g. the first word in the
        # batches constructed here). So we'll also need to keep a mask for that and do
        # final masking at the end.
        num_words = len(word_to_token)
        touched_words = torch.zeros(num_words).bool()

        word_lengths = torch.zeros(num_words, dtype=torch.long)
        p_word = torch.zeros((num_words, self.num_candidates), dtype=torch.float)
        candidate_phonemes = torch.zeros((num_words, self.num_candidates, max_num_phonemes), dtype=torch.long)
        # Track the word ID that produced each sample.
        word_ids = torch.zeros(num_words, dtype=torch.long)
        for i in trange(0, len(token_inputs_tensor), self.batch_size, unit="batch"):
            batch = token_inputs_tensor[i:i+self.batch_size]

            # Keep track of which token indices and word IDs are in the batch.
            # Account for the fact that the batch may not be maximum rows and may be padded
            # in the final row.
            batch_token_idxs = torch.arange(i * max_len, min(len(tokens) - 1, (i + self.batch_size) * max_len))
            batch_word_ids = token_to_word[batch_token_idxs]

            # TODO for BPE models, we really should keep predicting each candidate until we
            # reach a BPE boundary. Otherwise the candidates are likely to be subwords,
            # while the ground truth word is a longer full word. This will bork the
            # incremental prediction dynamics in ways I don't want to have to figure out.
            # Better to implement this correctly, which will be expensive. For every
            # ground-truth word, for each candidate, keep predicting on a beam until
            # we reach a token with BPE separator as the argmax output.
            # batch_p_token, batch_candidate_token_ids = self.get_predictive_topk(batch)
            batch_word_ids_square = batch_word_ids
            if batch_word_ids.shape[0] % max_len != 0:
                batch_word_ids_square = pad(batch_word_ids,
                                            (0, max_len - batch_word_ids.shape[0] % max_len),
                                            value=nonword_id)
            batch_word_ids_square = batch_word_ids_square.reshape(-1, max_len)
            retained_word_ids, batch_p_word, batch_word_strs = \
                self.get_predictive_topk_words(batch, batch_word_ids_square)

            ######### NB all operations beneath this point are on *words*, not tokens.

            # Drop nonword/padding data
            word_mask = retained_word_ids != nonword_id
            retained_word_ids = retained_word_ids[word_mask]
            batch_p_word = batch_p_word[word_mask]
            batch_word_strs = [strs for word_id, strs in zip(retained_word_ids, batch_word_strs)
                               if word_id != nonword_id]

            batch_ground_truth_phonemes = None
            
            if ground_truth_phonemes is not None:
                batch_ground_truth_phonemes = [ground_truth_phonemes.get(word_id.item(), None)
                                               for word_id in retained_word_ids.flatten()]
            batch_candidate_phonemes, batch_word_lengths = self.get_candidate_phonemes(
                batch_word_strs, max_num_phonemes, batch_ground_truth_phonemes)

            # Also ignore words with zero length.
            batch_mask = (batch_word_lengths > 0)
            retained_word_ids = retained_word_ids[batch_mask]
            batch_word_lengths = batch_word_lengths[batch_mask]
            batch_p_word = batch_p_word[batch_mask]
            batch_candidate_phonemes = batch_candidate_phonemes[batch_mask]

            batch_num_samples = batch_candidate_phonemes.shape[0]
            assert batch_num_samples == len(retained_word_ids)
            assert batch_p_word.shape[0] == batch_num_samples
            assert batch_word_lengths.shape[0] == batch_num_samples

            # Get target indices in contiguous output arrays. (NB word IDs are not
            # contiguous.)
            batch_word_idxs = torch.tensor([word_id_to_idx[word_id.item()]
                                            for word_id in retained_word_ids])

            word_lengths[batch_word_idxs] = batch_word_lengths
            p_word[batch_word_idxs] = batch_p_word
            candidate_phonemes[batch_word_idxs] = batch_candidate_phonemes
            word_ids[batch_word_idxs] = retained_word_ids
            touched_words[batch_word_idxs] = True

        # Sanity check
        assert (p_word[~touched_words] == 0).all()

        # Finally, filter out words which never had data computed.
        word_ids = word_ids[touched_words]
        word_lengths = word_lengths[touched_words]
        p_word = p_word[touched_words]
        candidate_phonemes = candidate_phonemes[touched_words]

        # Reindex word-level features.
        word_features_tensor = None
        if word_features is not None:
            word_features_tensor = torch.stack([word_features[word_id.item()]
                                                for word_id in word_ids])

        return NaturalLanguageStimulus(
            phonemes=list(self.phonemes),
            pad_phoneme_id=self.pad_phoneme_id,
            word_ids=word_ids,
            word_lengths=word_lengths,
            word_features=word_features_tensor,
            p_word=p_word,
            candidate_phonemes=candidate_phonemes,
        )