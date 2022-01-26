from typing import Any, Dict, List, Union

import datasets

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class WikipediaDataModule(LightningDataModule):

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str = "wiki_bio",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        print(f'Initializing WikipediaDataModule with num_workers = {self.num_workers}')

    def setup(self, stage: str) -> None:
        self.dataset = datasets.load_dataset(self.dataset_name)

        # filter dataset by text length
        self.dataset = self.dataset.filter(lambda x: len(x['text']) > 200)

        # split dataset 'text' in half: ['text1', 'text2']
        # TODO: sentence-tokenize and take second half?
        def map_ex(x):
            L = len(x['text'])
            x['text1'] = x['text'][:L//2].strip()
            x['text2'] = x['text'][L//2:].strip()
            return x
        self.dataset = self.dataset.map(map_ex)

        # tokenize dataset
        self.dataset = self.dataset.map(
            self.convert_to_features,
            batched=True,
        )
        self.columns = [
            "text1_attention_mask", "text1_input_ids",
            "text2_attention_mask", "text2_input_ids"
        ]
        for split in self.dataset.keys():
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self) -> None:
        # automatically download dataset & tokenizer
        datasets.load_dataset(self.dataset_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset["validation"],
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers
            )
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.num_workers)
                for x in self.eval_splits
            ]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch: Dict[str, Any]) -> Dict[str, Any]:
        # Tokenize the text/text pairs
        text1_features = self.tokenizer.batch_encode_plus(
            example_batch["text1"],
            max_length=self.max_seq_length,
            padding=True,
            truncation=True
        )
        text1_features = { f'text1_{k}': v for k,v in text1_features.items() }
        text2_features = self.tokenizer.batch_encode_plus(
            example_batch["text1"],
            max_length=self.max_seq_length,
            padding=True,
            truncation=True
        )
        text2_features = { f'text2_{k}': v for k,v in text2_features.items() }

        return (text1_features | text2_features)