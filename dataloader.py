from typing import Any, Dict, List, Union

import datasets

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from redact import remove_named_entities_spacy

class WikipediaDataModule(LightningDataModule):

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
        assert redaction_strategy in ["", "spacy_ner"]
        self.redaction_strategy = redaction_strategy
        print(f'Initializing WikipediaDataModule with num_workers = {self.num_workers}')

    def setup(self, stage: str) -> None:
        self.dataset = datasets.load_dataset(self.dataset_name)

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
                'text1': ex['target_text'],     # First paragraph of biography
                'text2': table_text,            # Table re-printed as a string
            }
        self.dataset = self.dataset.map(map_ex)

        # Redact text, if specified
        if self.redaction_strategy:
            if self.redaction_strategy == "spacy_ner":
                redact_func = remove_named_entities_spacy
            else:
                raise ValueError(f'unknown redaction strategy {self.redaction_strategy}')
            
            def redact_dataset(ex):
                # redact 'text1' field
                ex['text1'] = redact_func(ex['text1'])
                return ex

            self.dataset = self.dataset.map(redact_dataset)
            breakpoint()

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
        """Tokenizes `example_batch`, which includes 'text1' and 'text2' as keys."""
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

        return (text1_features | text2_features)