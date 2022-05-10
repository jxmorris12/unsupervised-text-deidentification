import datasets
import pytest

import transformers

from masking_tokenizing_dataset import MaskingTokenizingDataset
from utils import create_document_and_profile_from_wikibio

class TestMaskingTokenizingDataset:

    def test_train_data(self):
        split = "train[:4096]"
        train_dataset = datasets.load_dataset(
            "wiki_bio", split=split, version="1.2.0"
        )
        train_dataset = train_dataset.map(
            create_document_and_profile_from_wikibio
        )
        train_dataset = train_dataset.add_column(
            "text_key_id", 
            list(range(len(train_dataset)))
        )
        document_tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base')
        profile_tokenizer = transformers.AutoTokenizer.from_pretrained('google/tapas-base')
        train_tokenizing_dataset = MaskingTokenizingDataset(
            train_dataset,
            document_tokenizer=document_tokenizer,
            profile_tokenizer=profile_tokenizer,
            max_seq_length=64,
            word_dropout_ratio=0.0,
            word_dropout_perc=0.0,
            profile_row_dropout_perc=0.9,
            sample_spans=False,
            num_nearest_neighbors=0,
            document_types=["document"],
            is_train_dataset=True
        )
        for idx in range(len(train_tokenizing_dataset)):
            ex = train_tokenizing_dataset[idx]
            continue