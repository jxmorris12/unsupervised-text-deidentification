from typing import Any, Callable, Dict, List, Union

import functools
import os
import pickle

import datasets
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from redact import remove_named_entities_spacy_batch, remove_overlapping_words
from utils import name_from_table_rows

class WikipediaDataModule(LightningDataModule):
    str_to_idx: Dict[str, int]  # maps un-redacted profile text (string) to index in training set
    dataset_name: str
    max_seq_length: int
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    tokenizer: AutoTokenizer
    redaction_strategy: str     # one of ['', 'spacy_ner', 'lexical']
    base_folder: str            # base folder for precomputed_similarities/. defaults to ''.

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str = "wiki_bio",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 1,
        redaction_strategy = "",
        base_folder = "",
        **kwargs,
    ):
        super().__init__()
        assert dataset_name == "wiki_bio"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        assert redaction_strategy in ["", "spacy_ner", "lexical"]
        self.redaction_strategy = redaction_strategy
        print(f'Initializing WikipediaDataModule with num_workers = {self.num_workers}')
        self.base_folder = base_folder

    def setup(self, stage: str) -> None:
        # TODO: change split here
        train_split = 'train[:10%]'
        val_split = 'val[:20%]'
        self.train_dataset = datasets.load_dataset(self.dataset_name, split=train_split) # wiki_bio train size: 582,659
        self.val_dataset = datasets.load_dataset(self.dataset_name, split=val_split) # wiki_bio val size: 72,831

        # TODO: create a utility for loading this stuff
        # TODO: don't load similarities unless we're training with hard negatives
        # TODO: better nomenclature than 'hard negative'?
        k = 2048
        train_save_folder = os.path.join(self.base_folder, 'precomputed_similarities', f'{self.dataset_name}__{train_split}__{k}')
        assert os.path.exists(train_save_folder), f'no precomputed similarities at folder {train_save_folder}'
        val_save_folder = os.path.join(self.base_folder, 'precomputed_similarities', f'{self.dataset_name}__{val_split}__{k}')
        assert os.path.exists(val_save_folder), f'no precomputed similarities at folder {val_save_folder}'
        train_str_to_idx_path = os.path.join(train_save_folder, 'str_to_idx.p') 
        val_str_to_idx_path = os.path.join(val_save_folder, 'str_to_idx.p') 
        self.str_to_idx = (
            pickle.load(open(train_str_to_idx_path, 'rb')) |
            pickle.load(open(val_str_to_idx_path, 'rb'))
        )

        def create_document_and_profile_from_wikibio_instance(ex: Dict) -> Dict:
            """
            transforms wiki_bio example into (document, profile) pair

            >>> ex['target_text']
            'walter extra is a german award-winning aerobatic pilot , chief aircraft designer and founder of extra....
            >>> ex['input_text']
            {'table': {'column_header': ['nationality', 'name', 'article_title', 'occupation', 'birth_date'], 'row_number': [1, 1, 1, 1, 1], 'content': ['german', 'walter extra', 'walter extra\n', 'aircraft designer and manufacturer', '1954']}, 'context': 'walter extra\n'}
            """
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
                'document': ex['target_text'],          # First paragraph of biography
                'profile': table_text,                  # Table re-printed as a string
                'text_key': ex['target_text'] + ' ' + table_text, # store (document, profile) str key
            }

        self.train_dataset = self.train_dataset.map(create_document_and_profile_from_wikibio_instance)
        self.val_dataset = self.val_dataset.map(create_document_and_profile_from_wikibio_instance)
        
        def redact_example(redact_func: Callable, example: Dict, suffix: str):
            # redact 'text1' field
            example[f'document_{suffix}'] = redact_func(example['document'], example['profile'])
            return example

        lexical_redact_func = functools.partial(remove_overlapping_words, mask_token=self.tokenizer.mask_token, case_sensitive=False)
        self.train_dataset = self.train_dataset.map(
            lambda ex: redact_example(redact_func=lexical_redact_func, example=ex, suffix='redact_lexical'))
        self.val_dataset = self.val_dataset.map(
            lambda ex: redact_example(redact_func=lexical_redact_func, example=ex, suffix='redact_lexical'))

        ner_redact_func = lambda t1, t2: remove_named_entities_spacy_batch(t1, mask_token=self.tokenizer.mask_token)
        self.train_dataset = self.train_dataset.map(
            lambda ex: redact_example(redact_func=ner_redact_func, example=ex, suffix='redact_ner'),
            batched=True)
        self.val_dataset = self.val_dataset.map(
            lambda ex: redact_example(redact_func=ner_redact_func, example=ex, suffix='redact_ner'),
            batched=True)

        # tokenize dataset
        self.train_dataset = self.train_dataset.map(
            functools.partial(self.convert_to_features),
            batched=True,
        )
        self.val_dataset = self.val_dataset.map(
            functools.partial(self.convert_to_features),
            batched=True,
        )
        self.columns = [
            "text_key_id", # Indices of item in original dataset 
                           # (used for getting precomputed nearest-neighbors)

            "document",
                    # [original] First paragraph of wikipedia page
            "document_redact_ner",
                    # [redacted_ner] First paragraph of wikipedia page
            "document_redact_lexical",
                    # [redacted_lexical] First paragraph of wikipedia page

            "profile" # Table from wikipedia infobox
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
            shuffle=True # Only shuffle for train
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers
        )

    def convert_to_features(self, example_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenizes `example_batch`, which includes 'document' and 'profile' as keys.
        
        includes `text_key_id` column, which includes the IDs (indices) of each
            str text2 in the original training set. Used for matching to precomputed nearest neighbors.

        """
        ids_dict = {
            "text_key_id": np.array([self.str_to_idx[s] for s in example_batch["text_key"]])
        }
        # If the preceding line produces a KeyError, stored data in str_to_idx doesn't match the data
        # that was just loaded from the dataset. Either str_to_idx refers to a different split or dataset,
        # or it actually refers to a different version (i.e. it was precomputed with an older version of
        # wiki_bio data).
        return example_batch | ids_dict