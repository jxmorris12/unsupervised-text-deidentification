from typing import Dict, List, Union

import datasets
import pandas as pd
import torch
import transformers

from torch.utils.data import Dataset

from masking_span_sampler import MaskingSpanSampler
from utils import try_encode_table_tapas

class MaskingTokenizingDataset(Dataset):
    """A PyTorch Dataset that tokenizes strings and, optionally, samples spans of text, and
    randomly masks some inputs.
    """
    dataset: datasets.Dataset
    document_tokenizer: transformers.AutoTokenizer
    profile_tokenizer: transformers.AutoTokenizer
    document_types: List[str] # ["document", "document_redact_ner", "document_redact_lexical"] 
    is_train_dataset: bool
    def __init__(
            self,
            dataset: datasets.Dataset,
            document_tokenizer: transformers.AutoTokenizer,
            profile_tokenizer: Union[transformers.AutoTokenizer, None],
            max_seq_length: int,
            word_dropout_ratio: float,
            word_dropout_perc: float, 
            sample_spans: bool,
            document_types: List[str],
            is_train_dataset: bool, # bool so we can make sure not to redact validation data.
        ):
        self.dataset = dataset
        self.document_tokenizer = document_tokenizer
        self.profile_tokenizer = profile_tokenizer
        self.document_types = document_types
        self.is_train_dataset = is_train_dataset
        self.max_seq_length = max_seq_length

        if self.is_train_dataset:
            self.masking_span_sampler = MaskingSpanSampler(
                word_dropout_ratio=word_dropout_ratio,
                word_dropout_perc=word_dropout_perc,
                sample_spans=sample_spans,
                mask_token=document_tokenizer.mask_token
            )
        else:
            self.masking_span_sampler = None
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def _get_profile_df(self, keys: List[str], values: List[str]) -> pd.DataFrame:
        # TODO: why do we have to truncate? Why would we ever get different-length
        # keys and values?
        assert isinstance(keys, list) and len(keys) and isinstance(keys[0], str)
        assert isinstance(values, list) and len(values) and isinstance(values[0], str)
        if len(keys) > len(values):
            keys = keys[:len(values)]
        if len(values) > len(keys):
            values = values[:len(keys)]
        # Arbitrarily limit to 32 columns (TODO: figure this out too. Why doesn't
        # truncate=True already truncate to max_length during tokenization?)
        keys = keys[:32]
        values = values[:32]
        return pd.DataFrame(columns=keys, data=[values])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Gets an item from the dataset."""
        ex = self.dataset[idx]

        # ex.keys():
        # dict_keys([
        #   'input_text', 'target_text', 'name', 'document', 'profile', 'profile_keys',
        #   'profile_values', 'text_key', 'document_redact_lexical', 'document_redact_ner',
        #   'text_key_id'])

        out_ex = { "text_key_id": ex["text_key_id"] }
        # 
        # Tokenize documents.
        # 
        if self.is_train_dataset: # Only consider redaction if this is a train dataset!
            ex["document"] = self.masking_span_sampler.redact(ex["document"])
        for doc_type in self.document_types:
            doc_tokenized = self.document_tokenizer.encode_plus(
                ex[doc_type],
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            for _k, _v in doc_tokenized.items():
                out_ex[f"{doc_type}__{_k}"] = _v[0]

        # 
        # Tokenize profiles.
        # 
        # TODO: Consider permitting separate max_seq_length for profile.
        if self.profile_tokenizer is not None:
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
                )
            else:
                profile_tokenized = self.profile_tokenizer.encode_plus(
                    ex["profile"],
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
            for _k, _v in profile_tokenized.items():
                out_ex[f"profile__{_k}"] = _v[0]

        return out_ex

