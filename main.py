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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataloader import WikipediaDataModule
from model import DocumentProfileMatchingTransformer

USE_WANDB = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

num_cpus = os.cpu_count()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--redaction_strategy', type=str, default='',
        choices=('spacy_ner', 'lexical', '')
    )
    parser.add_argument('--loss_fn', type=str, default='exact',
        choices=('exact', 'nearest_neighbors', 'num_neighbors')
    )
    parser.add_argument('--num_neighbors', type=int, default=512)
    parser.add_argument('--word_dropout_ratio', type=float, default=0.0,
        help='percentage of the time to apply word dropout')
    parser.add_argument('--word_dropout_perc', type=float, default=0.5,
        help='when word dropout is applied, percentage of words to apply it to')

    args = parser.parse_args()
    args.dataset_name = 'wiki_bio'
    return args


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
        max_seq_length=args.max_seq_length,
        loss_fn=args.loss_fn,
        num_neighbors=args.num_neighbors,
        redaction_strategy=args.redaction_strategy,
        word_dropout_ratio=args.word_dropout_ratio,
        word_dropout_perc=args.word_dropout_perc,
    )

    loggers = []

    if USE_WANDB:
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(
            WandbLogger(
                project='deid-wikibio', 
                config=vars(args),
                job_type='train',
                entity='jack-morris',
            )
        )
    
    from pytorch_lightning.loggers import CSVLogger
    # TODO set experiment name same as W&B run name?
    # TODO make this show up, I don't think it does 
    # (maybe because I usually kill runs before they finish?).
    loggers.append(CSVLogger("logs"))

    # TODO: properly early stop with val metric that corresponds to args.redaction_strategy
    val_metric = "val_exact/document/loss"
    callbacks = [
        ModelCheckpoint(monitor=val_metric),
        EarlyStopping(monitor=val_metric, min_delta=0.00, patience=3, verbose=False, mode="min")
    ]

    print("creating Trainer")
    trainer = Trainer(
        default_root_dir="saves",
        callbacks=callbacks,
        max_epochs=args.epochs,
        log_every_n_steps=min(len(dm.train_dataloader()), 50),
        limit_train_batches=1.0, # change this to make training faster (1.0 = full train set)
        limit_val_batches=1.0,
        gpus=torch.cuda.device_count(),
        logger=loggers
    )
    trainer.fit(model, dm)

if __name__ == '__main__':
    main(get_args())
