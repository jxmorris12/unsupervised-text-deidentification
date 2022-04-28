from typing import Any, Callable, Dict, List, Union

import functools
import os
import pickle

import datasets
import numpy as np
import pandas as pd
import transformers

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from masking_tokenizing_dataset import MaskingTokenizingDataset
from redact import remove_named_entities_spacy_batch, remove_overlapping_words
from utils import create_document_and_profile_from_wikibio

# 
# TODO: filter data to have > 10 words or something? And maybe a certain
#   number of rows?
# TODO: think about this tomorrow!!
# bad data example:
# [train example 326510] - 0 words in target text
#   {'input_text': {'table': {'column_header': ['name', 'background', 'label', 'origin', 'years_active', 'article_title', 'genre'], 'row_number': [1, 1, 1, 1, 1, 1, 1], 'content': ['hardliner', 'group_or_band', 'runaway wreckords', "st. john 's , newfoundland & labrador , canada", '1999-2004\xa02009-2010', 'hardliner -lrb- band -rrb-\n', 'hard rock']}, 'context': 'hardliner -lrb- band -rrb-\n'}, 'target_text': "'' ''\n"}
# [train example 262678] - 1 word in target text
# {'input_text': {'table': {'column_header': ['caption', 'name', 'known_for', 'death_date', 'image', 'nationality', 'birth_place', 'birth_date', 'article_title', 'death_place'], 'row_number': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'content': ['-lrb- ನಮ ಮ -rrb- ಕ ತ ತ ರ ರ ಣ ಚ ನ', 'kittur chennamma', 'indian freedom fighter', '21 february 1829', 'kittur chenamma.jpg', 'indian', 'kakati , belgaum taluk , british india', '23 october 1778', 'kittur chennamma\n', 'bailhongal taluk']}, 'context': 'kittur chennamma\n'}, 'target_text': "' `` kittur\n"}
# 

class WikipediaDataModule(LightningDataModule):
    dataset_name: str
    dataset_version: str
    dataset_train_split: str
    dataset_val_split: str

    document_model_name_or_path: str
    profile_model_name_or_path: str
    document_tokenizer: transformers.AutoTokenizer
    profile_tokenizer: transformers.AutoTokenizer

    word_dropout_perc: float
    word_dropout_ratio: float
    sample_spans: bool

    train_batch_size: int
    eval_batch_size: int
    max_seq_length: int
    num_workers: int
    mask_token: str
    redaction_strategy: str     # one of ['', 'spacy_ner', 'lexical']
    base_folder: str            # base folder for precomputed_similarities/. defaults to ''.

    train_dataset: datasets.Dataset     # train examples
    val_dataset: datasets.Dataset       # validation examples
    adv_val_dataset: datasets.Dataset   # adversarially-generated validation examples

    def __init__(
        self,
        document_model_name_or_path: str,
        profile_model_name_or_path: str,
        max_seq_length: int,
        dataset_name: str = "wiki_bio",
        dataset_train_split: str = "train[:10%]",
        dataset_val_split: str = "val[:20%]",
        dataset_version: str = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 1,
        word_dropout_ratio: float = 0.0,
        word_dropout_perc: float = 0.0,
        sample_spans: bool = False,
        redaction_strategy = "",
        base_folder = "",
        **kwargs,
    ):
        super().__init__()
        assert dataset_name == "wiki_bio"

        self.document_model_name_or_path = document_model_name_or_path
        self.profile_model_name_or_path = profile_model_name_or_path
        self.document_tokenizer = transformers.AutoTokenizer.from_pretrained(document_model_name_or_path, use_fast=True)
        self.profile_tokenizer = transformers.AutoTokenizer.from_pretrained(profile_model_name_or_path, use_fast=True)
        self.max_seq_length = max_seq_length

        self.word_dropout_ratio = word_dropout_ratio
        self.word_dropout_perc = word_dropout_perc
        self.sample_spans = sample_spans

        self.dataset_name = dataset_name
        self.dataset_train_split = dataset_train_split
        self.dataset_val_split = dataset_val_split
        self.dataset_version = dataset_version

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        assert redaction_strategy in ["", "spacy_ner", "lexical"]
        self.redaction_strategy = redaction_strategy
        self.mask_token = self.document_tokenizer.mask_token
        print(f'Initializing WikipediaDataModule with num_workers = {self.num_workers} and mask token `{self.mask_token}`')
        self.base_folder = base_folder

    def _load_train_and_val_data(self):
        print(f"loading {self.dataset_name}[{self.dataset_version}] split {self.dataset_train_split}")
        self.train_dataset = datasets.load_dataset(
            self.dataset_name, split=self.dataset_train_split, version=self.dataset_version) # wiki_bio train size: 582,659

        print(f"loading {self.dataset_name}[{self.dataset_version}] split {self.dataset_val_split}")
        self.val_dataset = datasets.load_dataset(
            self.dataset_name, split=self.dataset_val_split, version=self.dataset_version) # wiki_bio val size: 72,831

        self.train_dataset = self.train_dataset.map(create_document_and_profile_from_wikibio)
        self.val_dataset = self.val_dataset.map(create_document_and_profile_from_wikibio)
        
        def redact_example(redact_func: Callable, example: Dict, suffix: str):
            # redact 'text1' field
            example[f'document_{suffix}'] = redact_func(example['document'], example['profile'])
            return example

        lexical_redact_func = functools.partial(
            remove_overlapping_words, mask_token=self.mask_token, case_sensitive=False)
        self.train_dataset = self.train_dataset.map(
            lambda ex: redact_example(redact_func=lexical_redact_func, example=ex, suffix='redact_lexical'))
        self.val_dataset = self.val_dataset.map(
            lambda ex: redact_example(redact_func=lexical_redact_func, example=ex, suffix='redact_lexical'))

        ner_redact_func = functools.partial(
            remove_named_entities_spacy_batch, mask_token=self.mask_token
        )
        # TODO: consider fixing this with by setting `new_fingerprint` arg:
        #       https://github.com/huggingface/datasets/issues/3178#issuecomment-1085932904
        self.train_dataset = self.train_dataset.map(
            lambda ex: redact_example(redact_func=ner_redact_func, example=ex, suffix='redact_ner'),
            batched=True)
        self.val_dataset = self.val_dataset.map(
            lambda ex: redact_example(redact_func=ner_redact_func, example=ex, suffix='redact_ner'),
            batched=True)

        # Add index column to dataset, so that we can track which profiles match to which
        # documents from precomputed embeddings.
        self.train_dataset = self.train_dataset.add_column(
            "text_key_id", 
            list(   
                range(
                    len(self.train_dataset)
                    )
            )
        )
        self.val_dataset = self.val_dataset.add_column(
            "text_key_id", 
            list(   
                range(
                    len(self.val_dataset)
                    )
            )
        )
        self.columns = [
            "text_key_id", # Indices of item in original dataset  (int)
                           # (used for getting precomputed nearest-neighbors)

            "document",
                    # [original] First paragraph of wikipedia page (str)
            "document_redact_ner",
                    # [redacted_ner] First paragraph of wikipedia page (str)
            "document_redact_lexical",
                    # [redacted_lexical] First paragraph of wikipedia page (str)

            "profile", # Table from wikipedia infobox (str)
            "profile_keys", # Keys to table from wikipedia infobox (str)
            "profile_values", # Values to table from wikipedia infobox (str)
        ]
        # self.train_dataset.set_format(type=None, columns=self.columns)
        # self.val_dataset.set_format(type=None, columns=self.columns)

    def _load_adv_val_data(self):
        # Load column with indices of adversarial examples, since it's not just 0-1000, some examples in the
        # dataset don't have adversarial examples.
        adv_idxs = list(map(int, open('adv_csvs/results_idx.txt').readlines()))[:1000]
        adv_val_dataset = { "text_key_id": adv_idxs }

        # Load CSV files with adversarial examples generated at different values of k.
        for k in [1, 10, 100, 1000]:
            df = pd.read_csv(f'adv_csvs/results_{k}_1000.csv')
            perturbed_text = df['perturbed_text'].map(
                lambda t: (
                    t
                    .replace('<mask>', self.mask_token)
                    .replace('<SPLIT>', '\n')
                    .replace('-lrb- ', '(').replace(' -rrb-', ')')
                    .strip()
                )
            )
            adv_val_dataset[f"adv_document_{k}"] = perturbed_text.tolist()
        
        val_n = len([i for i in adv_idxs if i < len(self.val_dataset)])
        adv_val_dataset = { k: v[:val_n] for k,v in adv_val_dataset.items() }
        self.adv_val_dataset = datasets.Dataset.from_dict(adv_val_dataset)

    def setup(self, stage: str) -> None:
        self._load_train_and_val_data()
        self._load_adv_val_data()

    def prepare_data(self) -> None:
        # automatically download dataset
        datasets.load_dataset(self.dataset_name)

    def train_dataloader(self) -> DataLoader:
        train_tokenizing_dataset = MaskingTokenizingDataset(
            self.val_dataset,
            document_tokenizer=self.document_tokenizer,
            profile_tokenizer=self.profile_tokenizer,
            max_seq_length=self.max_seq_length,
            word_dropout_ratio=self.word_dropout_ratio,
            word_dropout_perc=self.word_dropout_perc,
            sample_spans=self.sample_spans,
            document_types=["document", "document_redact_ner", "document_redact_lexical"],
            is_train_dataset=True
        )
        return DataLoader(
            train_tokenizing_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False # Only shuffle for train
        )

    def val_dataloader(self) -> List[DataLoader]:
        val_tokenizing_dataset = MaskingTokenizingDataset(
            self.val_dataset,
            document_tokenizer=self.document_tokenizer,
            profile_tokenizer=self.profile_tokenizer,
            max_seq_length=self.max_seq_length,
            word_dropout_ratio=0.0,
            word_dropout_perc=0.0,
            sample_spans=False,
            document_types=["document", "document_redact_ner", "document_redact_lexical"],
            is_train_dataset=False
        )
        adv_val_tokenizing_dataset = MaskingTokenizingDataset(
            self.adv_val_dataset,
            document_tokenizer=self.document_tokenizer,
            profile_tokenizer=None,
            max_seq_length=self.max_seq_length,
            word_dropout_ratio=0.0,
            word_dropout_perc=0.0,
            sample_spans=False,
            document_types=['adv_document_1', 'adv_document_10', 'adv_document_100', 'adv_document_1000'],
            is_train_dataset=False
        )
        return [
            DataLoader(
                val_tokenizing_dataset,
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                shuffle=False
            ),
            DataLoader(
                adv_val_tokenizing_dataset,
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                shuffle=False
            )
        ]
