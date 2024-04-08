from typing import Dict, List, Optional, Set

import random

import datasets
import torch
import transformers

from torch.utils.data import Dataset

from masking_span_sampler import MaskingSpanSampler
from utils import dict_union, tokenize_profile

class MaskingTokenizingDataset(Dataset):
    """A PyTorch Dataset that tokenizes strings and, optionally, samples spans of text, and
    randomly masks some inputs.
    """
    dataset: datasets.Dataset
    document_tokenizer: transformers.AutoTokenizer
    profile_tokenizer: Optional[transformers.AutoTokenizer]
    document_types: List[str] # ["document", "document_redact_ner", "document_redact_lexical"] 
    propagate_keys: List[str] # extra keys from dataset to propagate to output.
    is_train_dataset: bool
    adversarial_masking: bool
    num_nearest_neighbors: int
    adv_word_mask_map: Dict[int, Set]
    adv_word_mask_num: Dict[int, int]

    def __init__(
            self,
            dataset: datasets.Dataset,
            document_tokenizer: transformers.AutoTokenizer,
            profile_tokenizer: Optional[transformers.AutoTokenizer],
            max_seq_length: int,
            word_dropout_ratio: float,
            word_dropout_perc: float, 
            profile_row_dropout_perc: float,
            sample_spans: bool,
            document_types: List[str],
            is_train_dataset: bool, # bool so we can not redact the validation data.
            adversarial_masking: bool = False,
            idf_masking: bool = False,
            num_nearest_neighbors: int = 0,
            propagate_keys: Optional[List[str]] = None
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

        self.adv_word_mask_map = {} 
        self.adv_word_mask_num = {}

        self._cache = {}

        if self.is_train_dataset:
            self.masking_span_sampler = MaskingSpanSampler(
                word_dropout_ratio=word_dropout_ratio,
                word_dropout_perc=word_dropout_perc,
                sample_spans=sample_spans,
                mask_token=document_tokenizer.mask_token,
                idf_masking=idf_masking
            )
        else:
            self.masking_span_sampler = None
        

        if propagate_keys is not None:
            self.propagate_keys = propagate_keys
        else:
            self.propagate_keys = []

    def process_grad(self,
            input_ids: torch.Tensor,
            word_ids: torch.Tensor,
            word_importance: torch.Tensor,
            is_correct: torch.Tensor,
            text_key_id: torch.Tensor
        ) -> None:
        """Called from model on a training step.
        
        Args:
            input_ids: int torch.Tensor of shape (batch_size, max_seq_length)
            word_ids: int torch.Tensor of shape (batch_size, max_seq_length)
                Will have 0 for special tokens and start with word 1 otherwise. So a single
                row might look like: [0, 1, 1, 2, 3, 3, 3, 4, 4, 5, 0, 0, 0, 0, 0]
            word_importance: float torch.Tensor of shape (vocab_size,)
            is_correct: bool torch.Tensor of shape (batch_size,)
            text_key_id: int torch.Tensor of shape (batch_size,)
        """
        if not self.adversarial_masking: 
            return
        
        # TODO: pass bool tensor indicating which input got it right (add a word) or got it wrong (subtract a word)
        assert len(input_ids.shape) == len(word_ids.shape) == 2
        assert len(word_importance.shape) == len(text_key_id.shape) == 1
        
        batch_size = input_ids.shape[0]
        assert input_ids.shape == (batch_size, self.max_seq_length)
        assert word_ids.shape == (batch_size, self.max_seq_length)
        assert word_importance.shape == (self.document_tokenizer.vocab_size,)
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
        word_importance_per_token = torch.where(
            special_tokens_mask,
            torch.zeros_like(word_importance[input_ids]),
            word_importance[input_ids]
        )
        token_num_occurrences = (input_ids[..., None] == input_ids.flatten()).sum(dim=-1)

        # Normalize by number of occurrences, so high-frequency tokens can't contribute too much to
        # the total.
        # word_importance_per_token = word_importance_per_token / token_num_occurrences

        # Sum word_importance_per_token among subwords.
        word_importance_per_word = torch.einsum('bij,bi->bj',
            torch.eye(self.max_seq_length)[word_ids].to(word_importance_per_token.device),
            word_importance_per_token
        )
        # This gives the index of each *word* from each input with the maximum gradient.
        words_ordered_by_grad_norm = (-word_importance_per_word).argsort(dim=1)
        assert words_ordered_by_grad_norm.shape == (batch_size, self.max_seq_length)

        # Store each word so it'll be masked next time.
        for i in range(batch_size):
            ex_index = text_key_id[i].item()
            if ex_index not in self.adv_word_mask_map:
                self.adv_word_mask_map[ex_index] = set()
            if ex_index not in self.adv_word_mask_num:
                # TODO make command-line arg
                self.adv_word_mask_num[ex_index] = 16

            # TODO: replace is_correct condition with threshold on the loss?
            if is_correct[i].item():
                # If we got it right, make it harder.
                # Get IDs of subword tokens to mask.
                num_words_to_mask = self.adv_word_mask_num[ex_index]
                assert num_words_to_mask > 0
                words_to_mask_ids = words_ordered_by_grad_norm[i][:num_words_to_mask]

                word_id_masks = (words_to_mask_ids[:, None] == word_ids[i][None, :])
                assert word_id_masks.shape == (num_words_to_mask, self.max_seq_length)
                # We have to decode each word separately so that the spacing works
                # properly.
                for word_id_mask in word_id_masks:
                    subword_ids = input_ids[i][word_id_mask]
                    if len(subword_ids) == 0:
                        print("error: empty subword_ids for ex_index =", ex_index)
                        continue
                    # Convert these to text.
                    word_to_mask = (
                        self.document_tokenizer.decode(subword_ids).strip()
                    )
                    # TODO: how to handle repeat word_to_masks within the topk?
                    #       --> perhaps adv_word_mask_map should map to a set.
                    self.adv_word_mask_map[ex_index].add(word_to_mask)
            else:
                # If we got it wrong, make it easier by removing half of the masked words.
                num_words_to_unmask = round(len(self.adv_word_mask_map[ex_index]) / 2.0)
                if num_words_to_unmask > 0:
                    # Unmask all the words.
                    self.adv_word_mask_map[ex_index] = set()

                    # # Randomly unmask 50% of the masked words.
                    # words_to_unmask = random.sample(
                    #     list(self.adv_word_mask_map[ex_index]),
                    #     num_words_to_unmask
                    # )
                    # self.adv_word_mask_map[ex_index] = (
                    #     self.adv_word_mask_map[ex_index] - set(words_to_unmask)
                    # )
                    # # Decrement the number of words to re-mask before next time.
                    # self.adv_word_mask_num[ex_index] = max(
                    #     int(self.adv_word_mask_num[ex_index]/2), 1
                    # )
                    
    
    def __len__(self) -> int:
        return len(self.dataset)
    
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
            [_id+1 if (_id is not None) else 0 for _id in encoding.word_ids()]], dtype=torch.int64
        )
        return encoding_dict
    
    def _tokenize_profile(self, ex: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Tokenizes a profile, either with Tapas (dataframe-based) or as a single string."""
        res = tokenize_profile(
            tokenizer=self.profile_tokenizer,
            ex=ex,
            max_seq_length=self.max_seq_length
        )
        return res
    
    def _get_tokenized_profile(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.dataset[idx]
        if 'profile__input_ids' in ex:
            out_ex = {}
            for _k, _v in ex.items():
                if _k.startswith('profile__'):
                    out_ex[_k.replace('profile__', '')] = torch.tensor([_v])
            return out_ex
        else:
            return self._tokenize_profile(ex=self.dataset[idx])
    
    def _get_nearest_neighbors(self, idx: int, ex: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Gets the nearest-neighbors of an example. Used for contrastive learning."""
        assert "nearest_neighbor_idxs" in ex
        out_ex = {}
        eligible_neighbor_idxs = [
            _i for _i in ex["nearest_neighbor_idxs"] if _i != idx
        ]
        assert len(eligible_neighbor_idxs) >= self.num_nearest_neighbors
        neighbor_idxs = eligible_neighbor_idxs[:self.num_nearest_neighbors]

        neighbors_tokenized = [
           self._get_tokenized_profile(idx=n_idx) for n_idx in neighbor_idxs if n_idx < len(self.dataset)
        ]
        keys = neighbors_tokenized[0].keys() # probably like {'input_ids', 'attention_mask'}
        for _k in keys:
            out_ex[f"profile_neighbor__{_k}"] = torch.stack([
                n[_k][0] for n in neighbors_tokenized
            ])
        out_ex["profile_neighbor_idxs"] = torch.tensor(neighbor_idxs)
        return out_ex

    def _getitem_uncached(self, idx: int) -> Dict[str, torch.Tensor]:
        """Gets an item from the dataset."""
        ex = self.dataset[idx]

        # ex.keys():
        # dict_keys([
        #   'input_text', 'target_text', 'name', 'document', 'profile', 'profile_keys',
        #   'profile_values', 'text_key', 'document_redact_lexical', 'document_redact_ner',
        #   'text_key_id']) and possibly 'nearest_neighbor_idxs'
        out_ex = { "text_key_id": ex["text_key_id"] }

        # If specified, propagate some keys from input example to dataset output.
        for key in self.propagate_keys:
            out_ex[key] = ex[key]

        # 
        # Tokenize documents.
        # 
        if self.is_train_dataset: # Only consider redaction if this is a train dataset!
            # Shorten documents to speed up masking.
            ex["document"] = " ".join(ex["document"].split(" ")[:self.max_seq_length])

            if self.adversarial_masking:
                if idx not in self.adv_word_mask_map:
                    self.adv_word_mask_map[idx] = []
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
            # profile_row_dropout_perc > 0 indicates we want to use random-row dropout on profiles.
            # 'profile__input_ids' not present in ex indicates we haven't pre-tokenized the profiles.
            if (self.profile_row_dropout_perc > 0) or ("profile__input_ids" not in ex):
                # Re-tokenize profile on-the-fly (this is very slow especially for Tapas!)
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
            else:
                # Use pre-tokenized profile
                for _k, _v in ex.items():
                    if _k.startswith('profile__'):
                        out_ex[_k] = torch.tensor(_v)
            
            # Also tokenize profiles of nearest-neighbors, if specified.
            # This block of code is how neighbors are provided to our contrastive
            # learning algorithm.
            if self.num_nearest_neighbors > 0:
                out_ex = dict_union(out_ex, self._get_nearest_neighbors(idx=idx, ex=ex))
        k_list = ex['profile_keys'].split("||")
        v_list = ex['profile_values'].split("||")
        person_id = v_list[k_list.index('person_id')]
        # person_id is needed to get true and false positives person_ids at the point when metrics are pushed to wandb, to visualize the results. For a new dataset, replace this with whatever relevant id is.
        return out_ex | {"person_id" : person_id} 

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_train_dataset:
            # Don't cache training data. We want to re-generate random stuff
            # every time. (Also it's probably too slow.)
            return self._getitem_uncached(idx=idx)
        else:
            if idx not in self._cache:
                self._cache[idx] = self._getitem_uncached(idx=idx)
            return self._cache[idx]
