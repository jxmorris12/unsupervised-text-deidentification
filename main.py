""" adapted from "Finetune Transformers Models with PyTorch Lightning"
https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
"""
from typing import List, Tuple, Union

import numpy as np

from datasets import load_dataset, load_metric
from pytorch_lightning import Trainer, seed_everything
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, 
    Trainer, TrainingArguments
)

from dataloader import WikipediaDataModule
from model import DocumentProfileMatchingTransformer

# dataset = load_dataset("")

# TODO is this the right thing to use?
# model = AutoModel.from_pretrained()

model_name = "distilbert-base-uncased"
dataset_name = "yelp_polarity"

def main():
    seed_everything(42)

    dm = WikipediaDataModule(
        model_name_or_path=model_name,
        dataset_name=dataset_name
    )
    dm.setup("fit")
    model = DocumentProfileMatchingTransformer(
        model_name_or_path=model_name,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
    )

    trainer = Trainer(max_epochs=1, gpus=AVAIL_GPUS)
    trainer.fit(model, dm)

if __name__ == '__main__': main()