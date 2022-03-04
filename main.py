""" adapted from "Finetune Transformers Models with PyTorch Lightning"
https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
"""

# 
#   Produce tracebacks for SIGSEGV, SIGFPE, SIGABRT, SIGBUS and SIGILL signals
#           see docs.python.org/3/library/faulthandler.html
# 
# import faulthandler; faulthandler.enable()
# 

from typing import List, Tuple, Union

import argparse
import os

import numpy as np
import torch

from datasets import load_dataset, load_metric
from pytorch_lightning import Trainer, seed_everything

from dataloader import WikipediaDataModule
from model import DocumentProfileMatchingTransformer


args_dict = {
    'epochs': 8,
    'model_name': 'distilbert-base-uncased',
    'dataset_name': 'wiki_bio',
    'batch_size': 256,
    'max_seq_length': 64,
    'learning_rate': 2e-4,
    'redaction_strategy': 'word_overlap', # ['spacy_ner', 'word_overlap', '']
    'loss_fn': 'hard_negatives', # ['infonce', 'hard_negatives', 'exact']
    'num_neighbors': 1024,
}

USE_WANDB = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

num_cpus = min(os.cpu_count(), 12)

def main(args: argparse.Namespace):
    seed_everything(42)

    print(f"creating data module with redaction strategy '{args.redaction_strategy}'")
    dm = WikipediaDataModule(
        model_name_or_path=args.model_name,
        dataset_name=args.dataset_name,
        num_workers=min(8, num_cpus),
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        redaction_strategy=args.redaction_strategy,
    )
    dm.setup("fit")
    
    model = DocumentProfileMatchingTransformer(
        dataset_name=args.dataset_name,
        model_name_or_path=args.model_name,
        num_workers=min(8, num_cpus),
        learning_rate=args.learning_rate,
        loss_fn=args.loss_fn,
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
    
    from pytorch_lightning.loggers import CSVLogger
    # TODO set experiment name
    # TODO make this show up, I don't think it does (maybe because I usually kill runs before they finish?).
    loggers.append(CSVLogger("logs", name="deid_exp"))

    print("creating Trainer")
    trainer = Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=min(len(dm.train_dataloader()), 50),
        limit_train_batches=1.0, # change this to make training faster (1.0 = full train set)
        limit_val_batches=1.0,
        gpus=torch.cuda.device_count(),
        logger=loggers
    )
    trainer.fit(model, dm)

if __name__ == '__main__':
    args = argparse.Namespace(**args_dict)
    main(args)
