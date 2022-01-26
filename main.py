""" adapted from "Finetune Transformers Models with PyTorch Lightning"
https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
"""
from typing import List, Tuple, Union

import os

import numpy as np
import torch

from datasets import load_dataset, load_metric
from pytorch_lightning import Trainer, seed_everything

from dataloader import WikipediaDataModule
from model import DocumentProfileMatchingTransformer

model_name = "distilbert-base-uncased"
dataset_name = "yelp_polarity"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

num_cpus = min(os.cpu_count(), 8)

def main():
    seed_everything(42)

    dm = WikipediaDataModule(
        model_name_or_path=model_name,
        dataset_name=dataset_name,
        num_workers=num_cpus,
    )
    dm.setup("fit")
    model = DocumentProfileMatchingTransformer(
        model_name_or_path=model_name,
        eval_splits=dm.eval_splits,
        num_workers=num_cpus,
    )

    trainer = Trainer(max_epochs=1, gpus=torch.cuda.device_count())
    trainer.fit(model, dm)

if __name__ == '__main__': main()