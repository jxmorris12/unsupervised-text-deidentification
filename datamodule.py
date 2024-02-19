from typing import Any, Callable, Dict, List, Union

# import cPickle as pickle
import pickle
import functools
import os
import re

import datasets
import numpy as np
import pandas as pd
import torch
import transformers

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from masking_tokenizing_dataset import MaskingTokenizingDataset
from redact import remove_named_entities_bert_batch, remove_named_entities_spacy_batch, remove_overlapping_words, remove_words_val_idf
from utils import create_document_and_profile, dict_union, tokenize_profile, wikibio_example_has_non_redacted_rows

# 
# TODO: Consider filtering data to have > 10 words or something? And maybe a certain
#   number of rows?
# bad data example:
# [train example 326510] - 0 words in target text
#   {'input_text': {'table': {'column_header': ['name', 'background', 'label', 'origin', 'years_active', 'article_title', 'genre'], 'row_number': [1, 1, 1, 1, 1, 1, 1], 'content': ['hardliner', 'group_or_band', 'runaway wreckords', "st. john 's , newfoundland & labrador , canada", '1999-2004\xa02009-2010', 'hardliner -lrb- band -rrb-\n', 'hard rock']}, 'context': 'hardliner -lrb- band -rrb-\n'}, 'target_text': "'' ''\n"}
# [train example 262678] - 1 word in target text
# {'input_text': {'table': {'column_header': ['caption', 'name', 'known_for', 'death_date', 'image', 'nationality', 'birth_place', 'birth_date', 'article_title', 'death_place'], 'row_number': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'content': ['-lrb- ನಮ ಮ -rrb- ಕ ತ ತ ರ ರ ಣ ಚ ನ', 'kittur chennamma', 'indian freedom fighter', '21 february 1829', 'kittur chenamma.jpg', 'indian', 'kakati , belgaum taluk , british india', '23 october 1778', 'kittur chennamma\n', 'bailhongal taluk']}, 'context': 'kittur chennamma\n'}, 'target_text': "' `` kittur\n"}
# 


datasets.utils.logging.set_verbosity_error()


class DataModule(LightningDataModule):
    dataset_name: str
    dataset_version: str

    dataset_train_split: str
    dataset_val_split: str
    dataset_test_split: str

    document_model_name_or_path: str
    profile_model_name_or_path: str
    document_tokenizer: transformers.AutoTokenizer
    profile_tokenizer: transformers.AutoTokenizer

    word_dropout_perc: float
    word_dropout_ratio: float
    profile_row_dropout_perc: float
    sample_spans: bool
    adversarial_masking: bool
    do_bert_ner_redaction: bool

    train_batch_size: int
    eval_batch_size: int
    max_seq_length: int
    num_workers: int
    mask_token: str
    base_folder: str

    # If `num_nearest_neighbors` is set, will tokenize nearest-neighbors
    # in the train dataset and return, alongside the regular examples.
    num_nearest_neighbors: int

    train_dataset: datasets.Dataset     # train examples
    val_dataset: datasets.Dataset       # validation examples
    test_dataset: datasets.Dataset      # test examples

    adv_val_dataset: datasets.Dataset   # adversarially-generated validation examples

    def __init__(
        self,
        document_model_name_or_path: str,
        profile_model_name_or_path: str,
        max_seq_length: int,
        local_data_path:str = "",
        dataset_source: str = "wiki_bio",
        dataset_name: str = "wiki_bio",
        dataset_train_split: str = "train",
        dataset_val_split: str = "val",
        dataset_test_split: str = "test[:1%]",
        dataset_version: str = "1.2.0",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 1,
        word_dropout_ratio: float = 0.0,
        word_dropout_perc: float = 0.0,
        profile_row_dropout_perc: float = 0.0,
        adversarial_masking: bool = False,
        idf_masking: bool = False,
        num_nearest_neighbors: int = 0,
        sample_spans: bool = False,
        do_bert_ner_redaction: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert dataset_name == "wiki_bio"
        #assert datasets.__version__[0] == '2', "need datasets v2 for datamodule"
        assert "train" in dataset_train_split
        assert "val" in dataset_val_split
        assert "test" in dataset_test_split

        self.document_model_name_or_path = document_model_name_or_path
        self.profile_model_name_or_path = profile_model_name_or_path
        self.document_tokenizer = transformers.AutoTokenizer.from_pretrained(document_model_name_or_path, use_fast=True)
        self.profile_tokenizer = transformers.AutoTokenizer.from_pretrained(profile_model_name_or_path, use_fast=True)
        self.max_seq_length = max_seq_length
        self.do_bert_ner_redaction = do_bert_ner_redaction

        self.word_dropout_ratio = word_dropout_ratio
        self.word_dropout_perc = word_dropout_perc
        self.profile_row_dropout_perc = profile_row_dropout_perc
        self.sample_spans = sample_spans
        self.idf_masking = idf_masking
        self.adversarial_masking = adversarial_masking
        self.num_nearest_neighbors = num_nearest_neighbors
        
        self.dataset_source = dataset_source 
        self.local_data_path = local_data_path
        self.dataset_name = dataset_name
        self.dataset_train_split = dataset_train_split
        self.dataset_val_split = dataset_val_split
        self.dataset_test_split = dataset_test_split
        self.dataset_version = dataset_version

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        if idf_masking and (word_dropout_ratio == 0 or word_dropout_perc == 0):
            assert "can only do idf_masking with word_dropout_ratio > 0"

        if adversarial_masking and (word_dropout_ratio > 0):
            assert "must choose one or the other, either adversarial masking or random masking in document"

        if torch.cuda.is_available() and self.num_workers < 4:
            print(f'Warning: set num_workers to {self.num_workers}, expect dataloader bottleneck')
        self.mask_token = self.document_tokenizer.mask_token
        print(f'Initializing WikipediaDataModule with num_workers = {self.num_workers} and mask token `{self.mask_token}`')
        self.base_folder = os.path.dirname(os.path.abspath(__file__))

    def _load_train_and_val_data(self):
        version_str = ''  # change this any time any of the data-loading changes (to regenerate fingerprints)

        print(f"loading {self.dataset_source if self.dataset_source == 'parquet' else self.dataset_name}[{'' if self.dataset_source == 'parquet' else self.dataset_version}] split {self.dataset_train_split}")
        train_percent = float(self.dataset_train_split.split(":")[-1].split("%")[0]) # example form of dataset_train_split = "train[:100%]"
        val_percent = float(self.dataset_val_split.split(":")[-1].split("%")[0])
        test_percent = 100 - train_percent - val_percent
        self.dataset = datasets.Dataset.from_parquet(self.local_data_path).train_test_split(train_size=float(train_percent / 100)) if self.dataset_source == "parquet" else None
        
        self.train_dataset = self.dataset['train'] if self.dataset_source == "parquet" else datasets.load_dataset(self.dataset_name, split=self.dataset_train_split, version=self.dataset_version)
        print(f"train_dataset size: {len(self.train_dataset)}")
         # wiki_bio val size: 72,831
        #print(f"loading {self.dataset_name}[{self.dataset_version}] split {self.dataset_val_split}")
        self.val_test_dataset = self.dataset['test'].train_test_split(test_size=float(test_percent / (test_percent + val_percent)))
        self.val_dataset = self.val_test_dataset['train'] if self.dataset_source == "parquet" else datasets.load_dataset(self.dataset_name, split=self.dataset_val_split, version=self.dataset_version)
        print(f"val_dataset size: {len(self.val_dataset)}")
        # wiki_bio test size: 72,831
        #print(f"loading {self.dataset_name} split {self.dataset_test_split}")
        self.test_dataset = self.val_test_dataset['test'] if self.dataset_source == "parquet" else datasets.load_dataset(self.dataset_name, split=self.dataset_val_split, version=self.dataset_version)
        print(f"test_dataset size: {len(self.test_dataset)}")
        self.train_dataset = self.train_dataset.map(
            create_document_and_profile, num_proc=1, fn_kwargs={"dataset_source" : self.dataset_source})
        self.val_dataset = self.val_dataset.map(
            create_document_and_profile, num_proc=1, fn_kwargs={"dataset_source" : self.dataset_source})
        self.test_dataset = self.test_dataset.map(
            create_document_and_profile, num_proc=1, fn_kwargs={"dataset_source" : self.dataset_source})
        def redact_example(
                redact_func: Callable,
                example: Dict,
                suffix: str,
                include_profile: bool = True
            ):
            if include_profile:
                example[f'document_{suffix}'] = redact_func(example['document'], example['profile'])
            else:
                example[f'document_{suffix}'] = redact_func(example['document'])
            return example

        # Lexical (word overlap) redaction
        lexical_redact_func = functools.partial(
            remove_overlapping_words, mask_token=self.mask_token)
        self.val_dataset = self.val_dataset.map(
            lambda ex: redact_example(
                redact_func=lexical_redact_func, example=ex, suffix='redact_lexical', include_profile=True),
                num_proc=1,
        )
        self.test_dataset = self.test_dataset.map(
            lambda ex: redact_example(
                redact_func=lexical_redact_func, example=ex, suffix='redact_lexical', include_profile=True),
                num_proc=1,
        )

        # Redact training data too
        self.train_dataset = self.train_dataset.map(
            lambda ex: redact_example(
                redact_func=lexical_redact_func, example=ex, suffix='redact_lexical', include_profile=True),
                num_proc=1
        )

        #  Spacy NER redaction
        spacy_ner_redact_func = functools.partial(
            remove_named_entities_spacy_batch, mask_token=self.mask_token
        )
        self.val_dataset = self.val_dataset.map(
            lambda ex: redact_example(redact_func=spacy_ner_redact_func, example=ex, suffix='redact_ner', include_profile=False),
            batched=True,
            num_proc=1,
        )

        # BERT-NER redaction
        if self.do_bert_ner_redaction:
            bert_ner_redact_func = functools.partial(
                remove_named_entities_bert_batch, mask_token=self.mask_token
            )
            self.val_dataset = self.val_dataset.map(
                lambda ex: redact_example(redact_func=bert_ner_redact_func, example=ex, suffix='redact_ner_bert', include_profile=False),
                batched=True,
                num_proc=1,
            )
            self.test_dataset = self.test_dataset.map(
                lambda ex: redact_example(redact_func=bert_ner_redact_func, example=ex, suffix='redact_ner_bert', include_profile=False),
                batched=True,
                num_proc=1,
            )

        # BM25/IDF-based redaction  (20%, 40%, 60%, 80%)
        idf_redact_func = lambda p: functools.partial(
            remove_words_val_idf, p=p, mask_token=self.mask_token)
        self.val_dataset = self.val_dataset.map(
            lambda ex: redact_example(
                redact_func=idf_redact_func(0.2), example=ex, suffix='redact_idf_20', include_profile=False
            ),
            num_proc=1,
        )
        self.val_dataset = self.val_dataset.map(
            lambda ex: redact_example(
                redact_func=idf_redact_func(0.4), example=ex, suffix='redact_idf_40', include_profile=False),
            num_proc=1,
        )
        self.val_dataset = self.val_dataset.map(
            lambda ex: redact_example(
                redact_func=idf_redact_func(0.6), example=ex, suffix='redact_idf_60', include_profile=False
            ),
            num_proc=1,
        )
        self.val_dataset = self.val_dataset.map(
            lambda ex: redact_example(
                redact_func=idf_redact_func(0.8), example=ex, suffix='redact_idf_80', include_profile=False
            ),
            num_proc=1,
        )

        # Pre-tokenize profiles
        def tokenize_profile_ex(ex: Dict) -> Dict:
            tokenized_profile = tokenize_profile(
                tokenizer=self.profile_tokenizer,
                ex=ex,
                max_seq_length=self.max_seq_length
            )
            return dict_union(ex, {f'profile__{k}': v[0] for k, v in tokenized_profile.items()})

        self.train_dataset = self.train_dataset.map(
            tokenize_profile_ex,
            num_proc=max(1, self.num_workers),
        )
        self.val_dataset = self.val_dataset.map(
            tokenize_profile_ex,
            num_proc=max(1, self.num_workers),
        )
        self.test_dataset = self.test_dataset.map(
            tokenize_profile_ex,
            num_proc=max(1, self.num_workers),
        )

        # Add index column to dataset, so that we can track which profiles match to which
        # documents from precomputed embeddings.
        self.train_dataset = self.train_dataset.add_column(
            "text_key_id", list(range(len(self.train_dataset)))
        )
        self.val_dataset = self.val_dataset.add_column(
            "text_key_id", list(range(len(self.val_dataset)))
        )
        self.test_dataset = self.test_dataset.add_column(
            "text_key_id", list(range(len(self.test_dataset)))
        )

        # # Truncate length of adv_val_dataset if it's too long. We need the profiles
        # # from self.val_dataset to evaluate it.
        # val_n = len([i for i in range(len(self.adv_val_dataset)) if i < len(self.val_dataset)])
        # adv_val_dataset_dict = self.adv_val_dataset.to_dict()
        # adv_val_dataset_dict = { k: v[:val_n] for k,v in adv_val_dataset_dict.items() }
        # self.adv_val_dataset = datasets.Dataset.from_dict(adv_val_dataset_dict)

        # Load nearest-neighbors to train set, if requested.
        if self.num_nearest_neighbors > 0:
            base_folder = os.path.dirname(os.path.abspath(__file__))
            # Load train nearest-neighbors
            # embeddings/profile/model_3_3/train_nn.p
            train_nn_file_path = os.path.join(base_folder, 'embeddings', 'profile', 'model_3_3', 'train_nn.p')
            # train_nn_file_path = os.path.join(base_folder, 'nearest_neighbors', f'nn__{self.dataset_train_split}__256.p')
            assert os.path.exists(train_nn_file_path)
            print("Loading train nearest-neighbors from:", train_nn_file_path)
            train_nearest_neighbors = pickle.load(open(train_nn_file_path, 'rb'))
            assert len(train_nearest_neighbors) >= len(self.train_dataset)
            train_nearest_neighbors = train_nearest_neighbors[:len(self.train_dataset)]
            self.train_dataset = self.train_dataset.add_column(
                "nearest_neighbor_idxs", train_nearest_neighbors
            )
            # # Load val nearest-neighbors
            # # embeddings/profile/model_3_3/val_nn.p
            # val_nn_file_path = os.path.join(base_folder, 'embeddings', 'profile', 'model_3_3', 'val_nn.p')
            # # val_nn_file_path = os.path.join(base_folder, 'nearest_neighbors', f'nn__{self.dataset_val_split}__256.p')
            # assert os.path.exists(val_nn_file_path)
            # print("Loading val nearest-neighbors from:", val_nn_file_path)
            # val_nearest_neighbors = pickle.load(open(val_nn_file_path, 'rb'))
            # assert len(val_nearest_neighbors) >= len(self.val_dataset)
            # val_nearest_neighbors = val_nearest_neighbors[:len(self.val_dataset)]
            # self.val_dataset = self.val_dataset.add_column(
            #     "nearest_neighbor_idxs", val_nearest_neighbors
            # )
            # # Load neighbors into adv_val_dataset too
            # adv_val_nearest_neighbors = val_nearest_neighbors[:len(self.adv_val_dataset)]
            # self.adv_val_dataset = self.adv_val_dataset.add_column(
            #     "nearest_neighbor_idxs", adv_val_nearest_neighbors
            # )
        
        # Now disable parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
            

    # def _load_adv_val_data(self):
    #     # Load column with indices of adversarial examples, since it's not just 0-1000, some examples in the
    #     # dataset don't have adversarial examples.
    #     adv_idxs = list(
    #         map(
    #             int,
    #             open(os.path.join(self.base_folder, 'all_adv_csvs', 'adv_csvs/model_1/results_idx.txt')).readlines()
    #             )
    #     )[:1000]
    #     adv_val_dataset = { "text_key_id": adv_idxs }
    #
    #     # Load CSV files with adversarial examples generated at different values of k.
    #     for k in [1, 10, 100, 1000]:
    #         df = pd.read_csv(os.path.join(self.base_folder, 'all_adv_csvs', f'adv_csvs/model_1/results_{k}_1000.csv'))
    #         perturbed_text = df['perturbed_text'].map(
    #             lambda t: (
    #                 t
    #                 .replace('<mask>', self.mask_token)
    #                 .replace('<SPLIT>', '\n')
    #                 .replace('-lrb- ', '(').replace(' -rrb-', ')')
    #                 .strip()
    #             )
    #         )
    #         adv_val_dataset[f"adv_document_{k}"] = perturbed_text.tolist()
    #
    #     self.adv_val_dataset = datasets.Dataset.from_dict(adv_val_dataset)

    def setup(self, stage: str) -> None:
        # self._load_adv_val_data()
        self._load_train_and_val_data()

    def train_dataloader(self) -> DataLoader:
        train_tokenizing_dataset = MaskingTokenizingDataset(
            self.train_dataset,
            document_tokenizer=self.document_tokenizer,
            profile_tokenizer=self.profile_tokenizer,
            max_seq_length=self.max_seq_length,
            word_dropout_ratio=self.word_dropout_ratio,
            word_dropout_perc=self.word_dropout_perc,
            profile_row_dropout_perc=self.profile_row_dropout_perc,
            sample_spans=self.sample_spans,
            adversarial_masking=self.adversarial_masking,
            idf_masking=self.idf_masking,
            num_nearest_neighbors=self.num_nearest_neighbors,
            document_types=["document"],
            is_train_dataset=True
        )
        # sampler = torch.utils.data.RandomSampler(
        #     train_tokenizing_dataset, replacement=True,
        #     num_samples=len(self.train_dataset)
        # )
        return DataLoader(
            train_tokenizing_dataset,
            # sampler=sampler,
            persistent_workers=(self.num_workers > 0),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )

    def val_dataloader(self) -> List[DataLoader]:
        document_types = [
            "document",
            "document_redact_ner",
            "document_redact_lexical", 
            "document_redact_idf_20",  "document_redact_idf_40",
            "document_redact_idf_60",  "document_redact_idf_80"
        ]
        # if self.do_bert_ner_redaction:
            # document_types += ["document_redact_ner_bert"]
        
        val_tokenizing_dataset = MaskingTokenizingDataset(
            self.val_dataset,
            document_tokenizer=self.document_tokenizer,
            profile_tokenizer=self.profile_tokenizer,
            max_seq_length=self.max_seq_length,
            word_dropout_ratio=0.0,
            word_dropout_perc=0.0,
            profile_row_dropout_perc=0.0,
            sample_spans=False,
            adversarial_masking=False,
            idf_masking=False,
            num_nearest_neighbors=self.num_nearest_neighbors,
            document_types=document_types,
            is_train_dataset=False
        )
        # adv_val_tokenizing_dataset = MaskingTokenizingDataset(
        #     self.adv_val_dataset,
        #     document_tokenizer=self.document_tokenizer,
        #     profile_tokenizer=None,
        #     max_seq_length=self.max_seq_length,
        #     word_dropout_ratio=0.0,
        #     word_dropout_perc=0.0,
        #     profile_row_dropout_perc=0.0,
        #     sample_spans=False,
        #     adversarial_masking=False,
        #     idf_masking=False,
        #     document_types=["adv_document_1", "adv_document_10", "adv_document_100", "adv_document_1000"],
        #     is_train_dataset=False
        # )
        return DataLoader(
            val_tokenizing_dataset,
            batch_size=self.eval_batch_size,
            num_workers=min(self.num_workers, 8),
            persistent_workers=(self.num_workers > 0),
            pin_memory=True,
            shuffle=False
        )
        # return [
        #     DataLoader(
        #         val_tokenizing_dataset,
        #         batch_size=self.eval_batch_size,
        #         num_workers=min(self.num_workers, 8),
        #         persistent_workers=(self.num_workers > 0),
        #         pin_memory=True,
        #         shuffle=False
        #     ),
        #     DataLoader(
        #         adv_val_tokenizing_dataset,
        #         batch_size=self.eval_batch_size,
        #         num_workers=min(self.num_workers, 8),
        #         persistent_workers=(self.num_workers > 0),
        #         pin_memory=True,
        #         shuffle=False
        #     )
        # ]

    def test_dataloader(self) -> DataLoader:
        document_types = [
            "document", "document_redact_lexical",
        ]
        if self.do_bert_ner_redaction:
            document_types += ["document_redact_ner_bert"]

        test_tokenizing_dataset = MaskingTokenizingDataset(
            self.test_dataset,
            document_tokenizer=self.document_tokenizer,
            profile_tokenizer=self.profile_tokenizer,
            max_seq_length=self.max_seq_length,
            word_dropout_ratio=0.0,
            word_dropout_perc=0.0,
            profile_row_dropout_perc=0.0,
            sample_spans=False,
            adversarial_masking=False,
            idf_masking=False,
            # num_nearest_neighbors=self.num_nearest_neighbors,
            document_types=document_types,
            is_train_dataset=False
        )
        return DataLoader(
            test_tokenizing_dataset,
            batch_size=self.eval_batch_size,
            num_workers=min(self.num_workers, 8),
            persistent_workers=(self.num_workers > 0),
            pin_memory=True,
            shuffle=False
        )
