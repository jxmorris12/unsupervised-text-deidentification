""" adapted from "Finetune Transformers Models with PyTorch Lightning"
https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
"""

import faulthandler; faulthandler.enable()

from typing import List, Tuple, Union

import argparse
import os

import numpy as np
import torch

from datasets import load_dataset, load_metric
from pytorch_lightning import Trainer, seed_everything

from dataloader import WikipediaDataModule
from model import DocumentProfileMatchingTransformer, DocumentProfileMatchingTransformerWithHardNegatives


args_dict = {
    'model_name': 'distilbert-base-uncased',
    'dataset_name': 'wiki_bio',
    'batch_size': 256,
    'max_seq_length': 64,
    'learning_rate': 1e-4,
    'redaction_strategy': 'spacy_ner', # ['spacy_ner', 'word_overlap', '']
    'use_hard_negatives': True
}

USE_WANDB = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

num_cpus = min(os.cpu_count(), 12)

batch_size = 256
max_seq_length = 64

def main(args: argparse.Namespace):
    seed_everything(42)

    print("creating data module")
    dm = WikipediaDataModule(
        model_name_or_path=args.model_name,
        dataset_name=args.dataset_name,
        num_workers=num_cpus,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        redaction_strategy=args.redaction_strategy,
    )
    dm.setup("fit")

    model_cls = (
        DocumentProfileMatchingTransformerWithHardNegatives 
        if args.use_hard_negatives
        else DocumentProfileMatchingTransformer
    )
    model = model_cls(
        dataset_name=args.dataset_name,
        model_name_or_path=args.model_name,
        num_workers=num_cpus,
        learning_rate=args.learning_rate,
    )

    loggers = []

    if USE_WANDB:
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(
            WandbLogger(
                project='deid-wikibio', 
                config=args_dict,
                job_type='train',
                entity='jack-morris',
            )
        )

    print("creating Trainer")
    trainer = Trainer(
        max_epochs=10,
        log_every_n_steps=min(len(dm.train_dataloader()), 50),
        limit_train_batches=1.0, # change this to make training faster (1.0 = full train set)
        gpus=torch.cuda.device_count(),
        logger=loggers
    )
    trainer.fit(model, dm)

if __name__ == '__main__':
    args = argparse.Namespace(**args_dict)
    main(args)
