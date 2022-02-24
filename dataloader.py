from typing import Any, Dict, List, Union

import functools
import os
import pickle

import datasets
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from redact import remove_named_entities_spacy, remove_overlapping_words

class WikipediaDataModule(LightningDataModule):
    str_to_idx: Dict[str, int]  # maps un-redacted profile text (string) to index in training set

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str = "wiki_bio",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 1,
        redaction_strategy = "",
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        assert dataset_name == "wiki_bio"
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        assert redaction_strategy in ["", "spacy_ner", "word_overlap"]
        self.redaction_strategy = redaction_strategy
        print(f'Initializing WikipediaDataModule with num_workers = {self.num_workers}')

    def setup(self, stage: str) -> None:
        # TODO: change split here
        split = 'train[:10%]'
        self.train_dataset = datasets.load_dataset(self.dataset_name, split=split) # wiki_bio train size: 582,659
        self.val_dataset = datasets.load_dataset(self.dataset_name, split='val[:10%]') # wiki_bio val size: 72,831
        self.test_dataset = datasets.load_dataset(self.dataset_name, split='test[:10%]') # wiki_bio test size: 72,831


        # TODO: create a utility for loading this stuff
        # TODO: don't load similarities unless we're training with hard negatives
        # TODO: better nomenclature than 'hard negative'?
        k = 128
        save_folder = os.path.join('precomputed_similarities', f'{self.dataset_name}__{split}__{k}')
        assert os.path.exists(save_folder), f'no precomputed similarities at folder {save_folder}'
        str_to_idx_path = os.path.join(save_folder, 'str_to_idx.p') 
        self.str_to_idx = pickle.load(open(str_to_idx_path, 'rb'))

        # split dataset 'text' in half: ['text1', 'text2']
        # TODO: sentence-tokenize and take second half?
        def map_ex(ex):
            """
            transforms wiki_bio example into (text1, text2) pair

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
                'text1': ex['target_text'],          # First paragraph of biography
                'text2': table_text,                 # Table re-printed as a string
                'text_key': ex['target_text'] + ' ' + table_text, # store (document, profile) str key
            }

        self.train_dataset = self.train_dataset.map(map_ex)
        self.val_dataset = self.val_dataset.map(map_ex)
        self.test_dataset = self.test_dataset.map(map_ex)

        # Redact text, if specified
        if self.redaction_strategy:
            if self.redaction_strategy == "spacy_ner":
                redact_func = lambda t1, t2: remove_named_entities_spacy(t1)
            elif self.redaction_strategy == "word_overlap":
                redact_func = remove_overlapping_words
            else:
                raise ValueError(f'unknown redaction strategy {self.redaction_strategy}')
            
            def redact_dataset(ex):
                # redact 'text1' field
                ex['text1'] = redact_func(ex['text1'], ex['text2'])
                return ex

            self.train_dataset = self.train_dataset.map(redact_dataset)
            self.val_dataset = self.val_dataset.map(redact_dataset)
            self.test_dataset = self.test_dataset.map(redact_dataset)

        # tokenize dataset
        self.train_dataset = self.train_dataset.map(
            functools.partial(self.convert_to_features, include_ids=True),
            batched=True,
        )
        self.val_dataset = self.val_dataset.map(
            self.convert_to_features,
            batched=True,
        )
        self.test_dataset = self.test_dataset.map(
            self.convert_to_features,
            batched=True,
        )
        self.columns = [
            "text1_attention_mask", "text1_input_ids",
            "text2_attention_mask", "text2_input_ids",
        ]
        self.train_dataset.set_format(type="torch", columns=["text_key_id"] + self.columns)
        self.val_dataset.set_format(type="torch", columns=self.columns)
        self.test_dataset.set_format(type="torch", columns=self.columns)

    def prepare_data(self) -> None:
        # automatically download dataset & tokenizer
        datasets.load_dataset(self.dataset_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

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

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers
        )

    def convert_to_features(self, example_batch: Dict[str, Any], include_ids=False) -> Dict[str, Any]:
        """Tokenizes `example_batch`, which includes 'text1' and 'text2' as keys.
        
        If `include_ids` is True, includes `text2_ids` column, which includes the IDs (indices) of each
            str text2 in the original training set. Used for matching to precomputed nearest neighbors.

        """
        # Tokenize the text/text pairs
        text1_features = self.tokenizer.batch_encode_plus(
            example_batch["text1"],
            max_length=self.max_seq_length,
            padding=True,
            truncation=True
        )
        text1_features = { f'text1_{k}': v for k,v in text1_features.items() }
        text2_features = self.tokenizer.batch_encode_plus(
            example_batch["text2"],
            max_length=self.max_seq_length,
            padding=True,
            truncation=True
        )
        text2_features = { f'text2_{k}': v for k,v in text2_features.items() }
        ids_dict = {}

        if include_ids:
            ids_dict["text_key_id"] = np.array([self.str_to_idx[s] for s in example_batch["text_key"]])
        return (text1_features | text2_features | ids_dict)