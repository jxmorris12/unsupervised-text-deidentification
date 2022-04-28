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
from transformers import AutoTokenizer

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
    parser.add_argument('--num_validations_per_epoch', type=int, default=1,
        help='number of times to validate per epoch')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_seq_length', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=2e-5)

    parser.add_argument('--document_model_name', '--document_model', type=str, default='roberta-base')
    parser.add_argument('--profile_model_name', '--profile_model', type=str, default='roberta-base')
    
    parser.add_argument('--word_dropout_ratio', type=float, default=0.0,
        help='percentage of the time to apply word dropout')
    parser.add_argument('--word_dropout_perc', type=float, default=0.5,
        help='when word dropout is applied, percentage of words to apply it to')
    parser.add_argument('--pretrained_profile_encoder', action='store_true', default=False,
        help=('whether to fix profile encoder and just train document encoder. ' 
            '[if false, does coordinate ascent alternating models across epochs]'))
    
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
        help='factor to decrease learning rate by on drop')
    parser.add_argument('--lr_scheduler_patience', type=int, default=3,
        help='patience for lr scheduler [unit: epochs]')

    parser.add_argument('--sample_spans', action='store_true',
        default=False, help='sample spans from the document randomly during training')
    parser.add_argument('--adversarial_mask_k_tokens', '--adv_k', 
        type=int, default=0, help='number of tokens to adversarially mask')

    parser.add_argument('--dataset_name', type=str, default='wiki_bio')
    parser.add_argument('--dataset_train_split', type=str, default='train[:10%]')
    parser.add_argument('--dataset_version', type=str, default='1.2.0')

    args = parser.parse_args()
    args.dataset_val_split = 'val[:20%]'
    return args


def main(args: argparse.Namespace):
    assert torch.cuda.is_available(), "need CUDA for training!"
    seed_everything(42)

    print(f"creating data module with document mask token {doc_mask_token}")
    dm = WikipediaDataModule(
        document_model_name_or_path=args.document_model_name,
        profile_model_name_or_path=args.profile_model_name,
        max_seq_length=args.max_seq_length,
        dataset_name=args.dataset_name,
        dataset_train_split=args.dataset_train_split,
        dataset_val_split=args.dataset_val_split,
        dataset_version=args.dataset_version,
        word_dropout_ratio=args.word_dropout_ratio,
        word_dropout_perc=args.word_dropout_perc,
        sample_spans=args.sample_spans,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=min(8, num_cpus),
    )
    dm.setup("fit")
    
    # model = DocumentProfileMatchingTransformer.load_from_checkpoint(
        # distilbert-distilbert model
        #    '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/distilbert-base-uncased__dropout_0.8_0.8/deid-wikibio_default/1irhznnp_130/checkpoints/epoch=25-step=118376.ckpt',
        # roberta-distilbert model
        # '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/roberta__distilbert-base-uncased__dropout_0.8_0.8/deid-wikibio_default/1f7mlhxn_162/checkpoints/epoch=16-step=309551.ckpt',
    model = DocumentProfileMatchingTransformer(
        document_model_name_or_path=args.document_model_name,
        profile_model_name_or_path=args.profile_model_name,
        num_workers=min(8, num_cpus),
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pretrained_profile_encoder=args.pretrained_profile_encoder,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        adversarial_mask_k_tokens=args.adversarial_mask_k_tokens,
    )

    loggers = []

    exp_name = args.document_model_name
    if args.profile_model_name != args.document_model_name:
        exp_name += f'__{args.profile_model_name}'
    if args.sample_spans:
        exp_name += f'__sample_spans'
    if args.adversarial_mask_k_tokens:
        exp_name += f'__adv_{args.adversarial_mask_k_tokens}'
    if args.word_dropout_ratio:
        exp_name += f'__dropout_{args.word_dropout_perc}_{args.word_dropout_ratio}'
    if args.pretrained_profile_encoder:
        exp_name += '__fixprof'
    # day = time.strftime(f'%Y-%m-%d-%H%M')
    # exp_name += f'_{day}'

    # exp_name aliases
    exp_name = exp_name.replace('roberta-base', 'roberta')
    exp_name = exp_name.replace('sentence-transformers/paraphrase-MiniLM-L6-v2', 'st/paraphrase')

    if USE_WANDB:
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(
            name=exp_name,
            project='deid-wikibio-2', 
            config=vars(args),
            job_type='train',
            entity='jack-morris',
        )
        wandb_logger.watch(model)
        loggers.append(
            wandb_logger
        )
    
    from pytorch_lightning.loggers import CSVLogger
    # TODO set experiment name same as W&B run name?
    # TODO make this show up, I don't think it does 
    # (maybe because I usually kill runs before they finish?).
    loggers.append(CSVLogger("logs"))

    # TODO: argparse for val_metric
    # val_metric = "val/document/loss"
    # val_metric = "val/document_redact_lexical/loss"
    val_metric = "val/document_redact_ner/loss"
    early_stopping_patience = (args.lr_scheduler_patience * 5 * args.num_validations_per_epoch)
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(monitor=val_metric),
        EarlyStopping(monitor=val_metric, min_delta=0.00, patience=early_stopping_patience, verbose=True, mode="min")
    ]

    print("creating Trainer")
    trainer = Trainer(
        default_root_dir=f"saves/{exp_name}",
        val_check_interval=1/args.num_validations_per_epoch,
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
