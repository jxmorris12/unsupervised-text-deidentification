from typing import Any, Callable, Dict, List, Union

import functools
import os
import pickle

import datasets
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from redact import remove_named_entities_spacy_batch, remove_overlapping_words
from utils import name_from_table_rows

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

    def setup(self, stage: str) -> None:
        # TODO: argparse for split 
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
            table_text = '\n'.join([' | '.join(row) for row in table_rows])
            # return example: transformed table + first paragraph
            return {
                'name': name_from_table_rows(table_rows),
                'document': fixed_target_text,          # First paragraph of biography
                'profile': table_text,             # Table re-printed as a string
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

            "profile" # Table from wikipedia infobox (str)
        ]
        self.train_dataset.set_format(type=None, columns=self.columns)
        self.val_dataset.set_format(type=None, columns=self.columns)

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

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
