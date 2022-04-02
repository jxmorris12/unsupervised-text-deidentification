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
import time

import numpy as np
import torch

from datasets import load_dataset, load_metric
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

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
    parser.add_argument('--document_model_name', type=str, default='roberta-base')
    parser.add_argument('--profile_model_name', type=str, default='roberta-base')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--redaction_strategy', type=str, default='',
        choices=('spacy_ner', 'lexical', '')
    )
    parser.add_argument('--word_dropout_ratio', type=float, default=0.0,
        help='percentage of the time to apply word dropout')
    parser.add_argument('--word_dropout_perc', type=float, default=0.5,
        help='when word dropout is applied, percentage of words to apply it to')
    
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
        help='factor to decrease learning rate by on drop')
    parser.add_argument('--lr_scheduler_patience', type=int, default=3,
    help='factor to decrease learning rate by on drop [unit: epochs]')

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
        redaction_strategy=args.redaction_strategy,
    )
    dm.setup("fit")
    
    model = DocumentProfileMatchingTransformer(
        document_model_name_or_path=args.model_name,
        profile_model_name_or_path=args.model_name,
        dataset_name=args.dataset_name,
        num_workers=min(8, num_cpus),
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        redaction_strategy=args.redaction_strategy,
        word_dropout_ratio=args.word_dropout_ratio,
        word_dropout_perc=args.word_dropout_perc,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
    )

    loggers = []

    day = time.strftime(f'%Y-%m-%d-%H%M')
    # NOTE(js): `args.model_name[:4]` just grabs "elmo" or "bert"; feel free to change later
    exp_name = f'{args.model_name}_{day}'
    if args.redaction_strategy:
        exp_name += f'__redact_{args.redaction_strategy}'
    if args.word_dropout_ratio:
        exp_name += f'__dropout_{args.word_dropout_perc}_{args.word_dropout_ratio}'

    if USE_WANDB:
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(
            WandbLogger(
                name=exp_name,
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
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(monitor=val_metric),
        EarlyStopping(monitor=val_metric, min_delta=0.00, patience=5, verbose=False, mode="min")
    ]

    print("creating Trainer")
    trainer = Trainer(
        default_root_dir=f"saves/{exp_name}",
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
