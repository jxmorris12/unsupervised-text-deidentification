from typing import Dict, List, Union

import collections
import random

import datasets
import pandas as pd
import torch
import transformers

from torch.utils.data import Dataset

from masking_span_sampler import MaskingSpanSampler
from utils import dict_union, try_encode_table_tapas

class MaskingTokenizingDataset(Dataset):
    """A PyTorch Dataset that tokenizes strings and, optionally, samples spans of text, and
    randomly masks some inputs.
    """
    dataset: datasets.Dataset
    document_tokenizer: transformers.AutoTokenizer
    profile_tokenizer: transformers.AutoTokenizer
    document_types: List[str] # ["document", "document_redact_ner", "document_redact_lexical"] 
    is_train_dataset: bool
    adversarial_masking: bool
    num_nearest_neighbors: int
    adv_word_mask_map: collections.defaultdict

    def __init__(
            self,
            dataset: datasets.Dataset,
            document_tokenizer: transformers.AutoTokenizer,
            profile_tokenizer: Union[transformers.AutoTokenizer, None],
            max_seq_length: int,
            word_dropout_ratio: float,
            word_dropout_perc: float, 
            profile_row_dropout_perc: float,
            sample_spans: bool,
            document_types: List[str],
            is_train_dataset: bool, # bool so we can not redact the validation data.
            adversarial_masking: bool = False,
            num_nearest_neighbors: int = 0
        ):
        self.dataset = dataset
        self.document_tokenizer = document_tokenizer
        self.profile_tokenizer = profile_tokenizer
        self.max_seq_length = max_seq_length
        self.document_types = document_types
        self.is_train_dataset = is_train_dataset
        self.num_nearest_neighbors = num_nearest_neighbors
        self.profile_row_dropout_perc = profile_row_dropout_perc
        self.adversarial_masking = adversarial_masking

        self.adv_word_mask_map = collections.defaultdict(list)
        # TODO: make command-line flag
        adv_mask_k = 8
        self.adv_word_mask_num = collections.defaultdict(lambda: adv_mask_k)

        assert ((self.num_nearest_neighbors == 0) or self.is_train_dataset), "only need nearest-neighbors when training"

        if self.is_train_dataset:
            self.masking_span_sampler = MaskingSpanSampler(
                word_dropout_ratio=word_dropout_ratio,
                word_dropout_perc=word_dropout_perc,
                sample_spans=sample_spans,
                mask_token=document_tokenizer.mask_token
            )
        else:
            self.masking_span_sampler = None

    def process_grad(self,
            input_ids: torch.Tensor,
            word_ids: torch.Tensor,
            emb_grad: torch.Tensor,
            is_correct: torch.Tensor,
            text_key_id: torch.Tensor
        ) -> None:
        """Called from model on a training step.
        
        Args:
            input_ids: int torch.Tensor of shape (batch_size, max_seq_length)
            word_ids: int torch.Tensor of shape (batch_size, max_seq_length)
                Will have 0 for special tokens and start with word 1 otherwise. So a single
                row might look like: [0, 1, 1, 2, 3, 3, 3, 4, 4, 5, 0, 0, 0, 0, 0]
            emb_grad: float torch.Tensor of shape (vocab_size,)
            is_correct: bool torch.Tensor of shape (batch_size,)
            text_key_id: int torch.Tensor of shape (batch_size,)
        """
        if not self.adversarial_masking: 
            return
        
        # TODO: pass bool tensor indicating which input got it right (add a word) or got it wrong (subtract a word)
        assert len(input_ids.shape) == len(word_ids.shape) == 2
        assert len(emb_grad.shape) == len(text_key_id.shape) == 1
        
        batch_size = input_ids.shape[0]
        assert input_ids.shape == (batch_size, self.max_seq_length)
        assert word_ids.shape == (batch_size, self.max_seq_length)
        assert emb_grad.shape == (self.document_tokenizer.vocab_size,)
        assert is_correct.shape == (batch_size,)
        assert text_key_id.shape == (batch_size,)

        # Get word with maximum gradient from each training example.
        # This mask is True for special tokens like PAD, etc. and False otherwise.
        special_tokens_mask = (
            (
                input_ids[..., None] == torch.tensor(
                    self.document_tokenizer.all_special_ids).to(input_ids.device)
            ).any(dim=2)
        )
        emb_grad_per_token = torch.where(
            special_tokens_mask, torch.zeros_like(input_ids).float().to(input_ids.device),
            emb_grad[input_ids]
        )
        token_num_occurrences = (input_ids[..., None] == input_ids.flatten()).sum(dim=-1)

        # Normalize by number of occurrences, so high-frequency tokens can't contribute too much to
        # the total.
        emb_grad_per_token = emb_grad_per_token / token_num_occurrences

        # Sum emb_grad_per_token among subwords.
        emb_grad_per_word = torch.einsum('bij,bi->bj',
            torch.eye(self.max_seq_length)[word_ids].to(emb_grad_per_token.device), emb_grad_per_token
        )
        # This gives the index of each *word* from each input with the maximum gradient.
        words_ordered_by_grad_norm = (-emb_grad_per_word).argsort(dim=1)
        assert words_ordered_by_grad_norm.shape == (batch_size, self.max_seq_length)

        # Store each word so it'll be masked next time.
        for i in range(batch_size):
            ex_index = text_key_id[i].item()
            # TODO: replace is_correct condition with threshold on the loss???
            if is_correct[i].item():
                # If we got it right, make it harder.
                # Get IDs of subword tokens to mask.
                num_words_to_mask = self.adv_word_mask_num[ex_index]
                assert num_words_to_mask > 0
                words_to_mask_ids = words_ordered_by_grad_norm[i][:num_words_to_mask]
                subwords_to_mask_ids = input_ids[i][
                    (word_ids[i][:, None] == words_to_mask_ids[None]).any(dim=1)
                ]
                assert len(subwords_to_mask_ids) > 0
                # Convert these to words.
                list_of_words_to_mask = (
                    self.document_tokenizer.decode(subwords_to_mask_ids).strip().split()
                )
                self.adv_word_mask_map[ex_index].extend(list_of_words_to_mask)
            else:
                # If we got it wrong, make it easier.
                if len(self.adv_word_mask_map[ex_index]):
                    rand_removal_idx = random.randrange(
                        len(self.adv_word_mask_map[ex_index]))
                    self.adv_word_mask_map[ex_index].pop(rand_removal_idx)
                    # Decrement the number of words to remove before next time.
                    self.adv_word_mask_num[ex_index] = max(
                        self.adv_word_mask_num[ex_index]-1, 1
                    )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def _get_profile_df(self, keys: List[str], values: List[str]) -> pd.DataFrame:
        """Creates a dataframe from a list of keys and list of values. Used for TAPAS and
        other table-based models.
        """
        assert isinstance(keys, list) and len(keys) and isinstance(keys[0], str)
        assert isinstance(values, list) and len(values) and isinstance(values[0], str)
        return pd.DataFrame(columns=keys, data=[values])
    
    def _tokenize_profile(self, ex: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Tokenizes a profile, either with Tapas (dataframe-based) or as a single string."""
        if isinstance(self.profile_tokenizer, transformers.TapasTokenizer):
            prof_keys = ex["profile_keys"].split("||")
            prof_values = ex["profile_values"].split("||")
            if not len(prof_keys):
                raise ValueError("empty profile_keys")
            if not len(prof_values):
                raise ValueError("empty prof_values")
            df = self._get_profile_df(
                keys=prof_keys, values=prof_values
            )
            profile_tokenized = try_encode_table_tapas(
                df=df,
                tokenizer=self.profile_tokenizer,
                max_length=self.max_seq_length,
                query="Who is this?",
                num_cols=64
            )
        else:
            profile_tokenized = self.profile_tokenizer.encode_plus(
                ex["profile"],
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
        return profile_tokenized
    
    def _tokenize_document(self, ex: Dict[str, str], doc_type: str) -> Dict[str, torch.Tensor]:
        """Tokenizes a document."""
        encoding = self.document_tokenizer.encode_plus(
            ex[doc_type],
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        encoding_dict = {k: v for k,v in encoding.items()}
        # Get IDs of individual words, but special tokens have None, so replace those with 0, so
        # as not to get confused with word 0.
        encoding_dict["word_ids"] = torch.tensor([
            [_id+1 if (_id is not None) else 0 for _id in encoding.word_ids()]], dtype=torch.int64)
        return encoding_dict
    
    def _get_nearest_neighbors(self, ex: Dict[str, str]) -> torch.Tensor:
        """Gets the nearest-neighbors of an example. Used for contrastive learning."""
        assert "nearest_neighbor_idxs" in ex
        out_ex = {}
        eligible_neighbor_idxs = [
            _i for _i in ex["nearest_neighbor_idxs"]
        ]
        assert len(eligible_neighbor_idxs) >= self.num_nearest_neighbors
        neighbor_idxs = eligible_neighbor_idxs[:self.num_nearest_neighbors]
        neighbors_tokenized = [
            self._tokenize_profile(ex=self.dataset[n_idx]) for n_idx in neighbor_idxs
        ]
        keys = neighbors_tokenized[0].keys() # probably like {'input_ids', 'attention_mask'}
        for _k in keys:
            out_ex[f"profile_neighbor__{_k}"] = torch.stack([
                n[_k][0] for n in neighbors_tokenized
            ])
        out_ex["profile_neighbor_idxs"] = torch.tensor(neighbor_idxs)
        return out_ex

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Gets an item from the dataset."""
        ex = self.dataset[idx]

        # ex.keys():
        # dict_keys([
        #   'input_text', 'target_text', 'name', 'document', 'profile', 'profile_keys',
        #   'profile_values', 'text_key', 'document_redact_lexical', 'document_redact_ner',
        #   'text_key_id']) and possibly 'nearest_neighbor_idxs'

        out_ex = { "text_key_id": ex["text_key_id"] }
        # 
        # Tokenize documents.
        # 
        if self.is_train_dataset: # Only consider redaction if this is a train dataset!
            if self.adversarial_masking:
                ex["document"] = self.masking_span_sampler.fixed_redact_str(
                    text=ex["document"], words_to_mask=self.adv_word_mask_map[idx])
            else:
                ex["document"] = self.masking_span_sampler.random_redact_str(
                    text=ex["document"])
        
        for doc_type in self.document_types:
            doc_tokenized = self._tokenize_document(ex=ex, doc_type=doc_type)
            for _k, _v in doc_tokenized.items():
                out_ex[f"{doc_type}__{_k}"] = _v[0]

        # 
        # Tokenize profiles.
        # 
        # TODO: Consider permitting separate max_seq_length for profile.
        if self.profile_tokenizer is not None:
            # TODO: finish profile row-sampling.
            if self.is_train_dataset:
                if isinstance(self.profile_tokenizer, transformers.TapasTokenizer):
                    # TODO redact profile_keys and profile_values
                    profile_keys = ex["profile_keys"].split('||')
                    profile_values = ex["profile_values"].split('||')

                    profile_keys_list, profile_values_list = [], []
                    for k, v in zip(profile_keys, profile_values):
                        if random.random() >= self.profile_row_dropout_perc:
                            profile_keys_list.append(k)
                            profile_values_list.append(v)
                    if not len(profile_keys_list):
                        random_idx = random.choice(range(len(profile_keys)))
                        profile_keys_list.append(profile_keys[random_idx])
                        profile_values_list.append(profile_values[random_idx])

                    ex["profile_keys"] = '||'.join(profile_keys_list) 
                    ex["profile_values"] = '||'.join(profile_values_list) 
                else:
                    ex["profile"] = ' || '.join(
                        (
                            r for r in ex["profile"].split(' || ')
                            if random.random() >= self.profile_row_dropout_perc
                         )
                    )

            profile_tokenized = self._tokenize_profile(ex=ex)
            for _k, _v in profile_tokenized.items():
                out_ex[f"profile__{_k}"] = _v[0]
            
            # Also tokenize profiles of nearest-neighbors, if specified.
            # This block of code is how neighbors are provided to our contrastive
            # learning algorithm.
            if self.num_nearest_neighbors > 0:
                out_ex = dict_union(out_ex, self._get_nearest_neighbors(ex))

        return out_ex

