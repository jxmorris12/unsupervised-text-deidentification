from typing import Any, Callable, Dict, List, Union

import functools
import os
import pickle

import datasets
import numpy as np
import pandas as pd

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from redact import remove_named_entities_spacy_batch, remove_overlapping_words
from utils import get_table_minus_name, name_from_table_rows

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

    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    mask_token: str
    redaction_strategy: str     # one of ['', 'spacy_ner', 'lexical']
    base_folder: str            # base folder for precomputed_similarities/. defaults to ''.

    train_dataset: datasets.Dataset     # train examples
    val_dataset: datasets.Dataset       # validation examples
    adv_val_dataset: datasets.Dataset   # adversarially-generated validation examples

    def __init__(
        self,
        dataset_name: str = "wiki_bio",
        dataset_train_split: str = "train[:10%]",
        dataset_val_split: str = "val[:20%]",
        dataset_version: str = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 1,
        mask_token: str = "[MASK]",
        redaction_strategy = "",
        base_folder = "",
        **kwargs,
    ):
        super().__init__()
        assert dataset_name == "wiki_bio"
        self.dataset_name = dataset_name
        self.dataset_train_split = dataset_train_split
        self.dataset_val_split = dataset_val_split
        self.dataset_version = dataset_version

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        assert redaction_strategy in ["", "spacy_ner", "lexical"]
        self.redaction_strategy = redaction_strategy
        self.mask_token = mask_token
        print(f'Initializing WikipediaDataModule with num_workers = {self.num_workers} and mask token `{self.mask_token}`')
        self.base_folder = base_folder

    def _load_train_and_val_data(self):
        print(f"loading {self.dataset_name}[{self.dataset_version}] split {self.dataset_train_split}")
        self.train_dataset = datasets.load_dataset(
            self.dataset_name, split=self.dataset_train_split, version=self.dataset_version) # wiki_bio train size: 582,659

        print(f"loading {self.dataset_name}[{self.dataset_version}] split {self.dataset_val_split}")
        self.val_dataset = datasets.load_dataset(
            self.dataset_name, split=self.dataset_val_split, version=self.dataset_version) # wiki_bio val size: 72,831

        def create_document_and_profile_from_wikibio_instance(ex: Dict) -> Dict:
            """
            transforms wiki_bio example into (document, profile) pair

            >>> ex['target_text']
            'walter extra is a german award-winning aerobatic pilot , chief aircraft designer and founder of extra....
            >>> ex['input_text']
            {'table': {'column_header': ['nationality', 'name', 'article_title', 'occupation', 'birth_date'], 'row_number': [1, 1, 1, 1, 1], 'content': ['german', 'walter extra', 'walter extra\n', 'aircraft designer and manufacturer', '1954']}, 'context': 'walter extra\n'}
            """
            # replace weird textual artifacts: -lrb- with ( and -rrb- with )
            fixed_target_text = ex['target_text'].replace('-lrb- ', '(').replace(' -rrb-', ')')
            # transform table to str
            table_info = ex['input_text']['table']
            table_rows = list(zip(
                map(lambda s: s.strip(), table_info['column_header']),
                map(lambda s: s.strip(), table_info['content']))
            )
            table_text = (
                '\n'.join([' | '.join(row) for row in table_rows])
            )
            table_text_without_name = (
                '\n'.join([' | '.join(row) for row in get_table_minus_name(table_rows)])
            )
            # return example: transformed table + first paragraph
            return {
                'name': name_from_table_rows(table_rows),
                'document': fixed_target_text,          # First paragraph of biography
                'profile': table_text,                  # Table re-printed as a string
                'profile_without_name': table_text_without_name, # Table with name removed
                'text_key': ex['target_text'] + ' ' + table_text, # store (document, profile) str key
            }

        self.train_dataset = self.train_dataset.map(create_document_and_profile_from_wikibio_instance)
        self.val_dataset = self.val_dataset.map(create_document_and_profile_from_wikibio_instance)
        
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
            "profile_without_name" # Table from wikipedia infobox, with name removed (str)
        ]
        self.train_dataset.set_format(type=None, columns=self.columns)
        self.val_dataset.set_format(type=None, columns=self.columns)

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
        
        val_n = len(self.val_dataset)
        adv_val_dataset = { k: v[:val_n] for k,v in adv_val_dataset.items() }
        self.adv_val_dataset = datasets.Dataset.from_dict(adv_val_dataset)

    def setup(self, stage: str) -> None:
        self._load_train_and_val_data()
        self._load_adv_val_data()

    def prepare_data(self) -> None:
        # automatically download dataset
        datasets.load_dataset(self.dataset_name)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False # Only shuffle for train
        )

    def val_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                self.val_dataset,
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                shuffle=False
            ),
            DataLoader(
                self.adv_val_dataset,
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                shuffle=False
            )
        ]
