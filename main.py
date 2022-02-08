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
dataset_name = "wiki_bio"
redaction_strategy = "spacy_ner"

USE_WANDB = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

num_cpus = min(os.cpu_count(), 8)

batch_size = 128
learning_rate = 1e-4
def main():
    seed_everything(42)

    dm = WikipediaDataModule(
        model_name_or_path=model_name,
        dataset_name=dataset_name,
        num_workers=num_cpus,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        redaction_strategy=redaction_strategy,
    )
    dm.setup("fit")
    model = DocumentProfileMatchingTransformer(
        model_name_or_path=model_name,
        eval_splits=dm.eval_splits,
        num_workers=num_cpus,
        learning_rate=learning_rate,
    )

    loggers = []

    if USE_WANDB:
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(
            WandbLogger(
                project='deid-wikibio', 
                config={
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                },
                job_type='train',
                entity='jack-morris',
            )
        )

    trainer = Trainer(
        max_epochs=3,
        limit_train_batches=0.1,
        gpus=torch.cuda.device_count(),
        logger=loggers
    )
    trainer.fit(model, dm)

if __name__ == '__main__': main()