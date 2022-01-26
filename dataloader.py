from typing import List, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class WikipediaDataModule(LightningDataModule):

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str = "wiki_bio",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str) -> None:
        self.dataset = datasets.load_dataset(self.dataset_name)

        # filter dataset by text length
        self.dataset = self.dataset.filter(lambda x: len(x['text']) > 200)

        # split dataset 'text' in half: ['text1', 'text2']
        # TODO: sentence-tokenize and take second half?
         def map_ex(ex):
            L = len(x['text'])
            x['text1'] = x['text'][:L//2].strip()
            x['text2'] = x['text'][L//2:].strip()
            return x

        # tokenize dataset
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self) -> None:
        # automatically download dataset & tokenizer
        datasets.load_dataset(self.dataset_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        texts_or_text_pairs = list(zip(example_batch["text1"], example_batch["text2"]))

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        return features