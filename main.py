from typing import List, Tuple, Union

import argparse
import glob
import os
import time

import numpy as np
import torch

from datasets import load_dataset, load_metric
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from transformers import AutoTokenizer

from datamodule import DataModule
from utils import model_cls_dict

USE_WANDB = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# num_cpus = len(os.sched_getaffinity(0)) # starts a bunch of processes(cumbersome to manage) without any improvement in speed
# num_cpus = 4 * torch.cuda.device_count() # starts a single process and so easier to manage
num_cpus = 1 # increasing count isn't helping

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--debug_mode", action="store_true", 
                    help="Take only first 100 rows of the dataset.") 
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--checkpoint_vnum', type=str, default='')

    parser.add_argument('--loss_function', '--loss_fn', '--loss', type=str,
                        choices=['coordinate_ascent', 'contrastive_cross_attention', 'contrastive'],
                        default='coordinate_ascent',
                        help='loss function to use for training'
                        )

    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='\% of validation to use. ONLY reduce this for debugging.')

    parser.add_argument('--num_validations_per_epoch', type=int, default=1,
                        help='number of times to validate per epoch')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--grad_norm_clip', type=float, default=10.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--precision', type=int, default=32)

    parser.add_argument('--num_nearest_neighbors', '--n', type=int, default=0,
                        help='number of negative samples for contrastive loss'
                        )

    parser.add_argument('--document_model_name', '--document_model', type=str,
                        default='roberta', choices=['distilbert', 'bert', 'roberta', 'pmlm-r', 'pmlm-a', 'longformer'])
    parser.add_argument('--profile_model_name', '--profile_model', type=str,
                        default='distilbert', choices=['distilbert', 'bert', 'roberta', 'tapas'])

    parser.add_argument('--shared_embedding_dim', '--e', type=int, default=768,
                        help='size of shared embeddings')

    parser.add_argument('--word_dropout_ratio', type=float, default=0.0,
                        help='percentage of the time to apply word dropout')
    parser.add_argument('--word_dropout_perc', type=float, default=0.5,
                        help='when word dropout is applied, percentage of words to apply it to')
    parser.add_argument('--profile_row_dropout_perc', type=float,
                        default=0.0, help='\% of rows to dropout')
    parser.add_argument('--pretrained_profile_encoder', '--freeze_profile_encoder',
                        action='store_true', default=False,
                        help=('whether to fix profile encoder and just train document encoder. '
                              '[if false, does coordinate ascent alternating models across epochs]')
                        )

    parser.add_argument('--lr_scheduler_factor', type=float, default=0.4,
                        help='factor to decrease learning rate by on drop')
    parser.add_argument('--lr_scheduler_patience', type=int, default=6,
                        help='patience for lr scheduler [unit: epochs]')
    parser.add_argument('--model_ckpt_save_top_k', type=int, default=1,
                        help=('number of models to save total for each checkpoint monitored. 1 means '
                              'every model will be overwritten and only the best one will stay. -1 means '
                              'to retain all models from all checkpoints.'))

    parser.add_argument('--sample_spans', action='store_true',
                        default=False, help='sample spans from the document randomly during training')
    parser.add_argument('--adversarial_masking', '--adv_k',
                        default=False, action='store_true', help='whether to do adversarial masking'
                        )
    parser.add_argument('--idf_masking', default=False, action='store_true',
                        help='whether to do idf-based masking (via bm25)'
                        )
    
    parser.add_argument('--dataset_source', type=str, default='huggingface', help='Source of the dataset?', choices=['huggingface', 'parquet'])
    parser.add_argument('--local_data_path', type=str, default='', help='Only required if data needs to be loaded from a local directory')
    parser.add_argument('--dataset_name', type=str, default='wiki_bio')
    parser.add_argument('--dataset_train_split', type=str, default='train[:100%]')
    parser.add_argument('--dataset_val_split', type=str, default='val[:100%]')
    parser.add_argument('--dataset_test_split', type=str, default='test[:100%]')
    parser.add_argument('--dataset_version', type=str, default='1.2.0')

    parser.add_argument('--wandb_run_id', type=str, default=None,
                        help='run id for weights & biases')
    
    parser.add_argument('--wandb_project_name', type=str, default='deid-wikibio-6',
                        help='project name for weights & biases')
    
    parser.add_argument('--wandb_entity', type=str, default='jack-morris', help='entity which have access for weights & biases')

    args = parser.parse_args()

    if args.checkpoint_path and args.checkpoint_vnum:
        raise ValueError('cannot provide both checkpoint_path and checkpoint_vnum')

    if args.checkpoint_vnum:
        checkpoint_paths = glob.glob(f'saves/*/*/*{args.checkpoint_vnum}/checkpoints/*.ckpt')
        if not checkpoint_paths:
            raise ValueError(f'Found no checkpoints for vnum {args.checkpoint_vnum}')
        else:
            args.checkpoint_path = checkpoint_paths[-1]
            print("loading model from", args.checkpoint_path)

    if args.loss_function == 'contrastive_cross_attention':
        # TODO: implement contrastive with just in-batch negatives
        assert args.num_nearest_neighbors > 0, "nead nearest-neighbors for contrastive learning"

    return args


def transformers_name_from_name(name: str) -> str:
    if name == 'roberta':
        return 'roberta-base'
    elif name == 'tapas':
        #return 'google/tapas-base-finetuned-wtq'
        return 'google/tapas-base'
    elif name == 'bert':
        return 'bert-base-uncased'
    elif name == 'distilbert':
        return 'distilbert-base-uncased'
    elif name == 'pmlm-r':
        return 'jxm/u-PMLM-R'
    elif name == 'pmlm-a':
        return 'jxm/u-PMLM-A'
    elif name == 'longformer':
        return 'allenai/longformer-base-4096'
    else:
        return f'unsupported model name {name}'


def main(args: argparse.Namespace):
    assert torch.cuda.is_available(), "need CUDA for training!"
    seed_everything(42)
    document_model = transformers_name_from_name(args.document_model_name)
    profile_model = transformers_name_from_name(args.profile_model_name)
    print("profile_model :", profile_model)
    model_cls = model_cls_dict[args.loss_function]

    # roberta-tapas trained on 0.5/0.5/0.5 dropout for 110 epochs /22 hours:
    # checkpoint_path = "/home/jxm3/research/deidentification/unsupervised-deidentification/saves/ca__roberta__tapas__dropout_0.5_0.5_0.5/deid-wikibio-2_default/2ai8js2r_328/checkpoints/epoch=110-step=50615.ckpt"
    checkpoint_path = args.checkpoint_path

    if checkpoint_path:
        model = model_cls.load_from_checkpoint(
            checkpoint_path,
            document_model_name_or_path=document_model,
            profile_model_name_or_path=profile_model,
            learning_rate=args.learning_rate,
            pretrained_profile_encoder=args.pretrained_profile_encoder,
            lr_scheduler_factor=args.lr_scheduler_factor,
            lr_scheduler_patience=args.lr_scheduler_patience,
            train_batch_size=args.batch_size,
            num_workers=min(8, num_cpus),
            gradient_clip_val=args.grad_norm_clip,
            label_smoothing=args.label_smoothing,
            shared_embedding_dim=args.shared_embedding_dim,
        )
        model.path_to_save_checkpoints = os.path.dirname(os.path.abspath(checkpoint_path))
    else:
        model = model_cls(
            document_model_name_or_path=document_model,
            profile_model_name_or_path=profile_model,
            learning_rate=args.learning_rate,
            pretrained_profile_encoder=args.pretrained_profile_encoder,
            lr_scheduler_factor=args.lr_scheduler_factor,
            lr_scheduler_patience=args.lr_scheduler_patience,
            train_batch_size=args.batch_size,
            num_workers=min(8, num_cpus),
            gradient_clip_val=args.grad_norm_clip,
            label_smoothing=args.label_smoothing,
            shared_embedding_dim=args.shared_embedding_dim,
        )

    loggers = []

    lf_short = {
        'coordinate_ascent': 'ca',
        'contrastive_cross_attention': 'cca',
        'contrastive': 'co',
    }[args.loss_function]
    exp_name = lf_short + '__' + args.document_model_name
    if args.profile_model_name != args.document_model_name:
        exp_name += f'__{args.profile_model_name}'
    if args.sample_spans:
        exp_name += f'__sample_spans'
    if args.adversarial_masking:
        exp_name += f'__adv'
    if args.idf_masking:
        exp_name += f'__idf'
    if args.num_nearest_neighbors:
        exp_name += f'__n_{args.num_nearest_neighbors}'
    if args.word_dropout_ratio or args.profile_row_dropout_perc:
        exp_name += f'__dropout_{args.word_dropout_perc}_{args.word_dropout_ratio}_{args.profile_row_dropout_perc}'
    if args.pretrained_profile_encoder:
        exp_name += '__fixprof'
    if args.shared_embedding_dim != 768:
        exp_name += f'__e{args.shared_embedding_dim}'
    if args.label_smoothing:
        exp_name += f'__ls{args.label_smoothing}'
    # day = time.strftime(f'%Y-%m-%d-%H%M')
    # exp_name += f'_{day}'

    # exp_name aliases
    exp_name = exp_name.replace('roberta-base', 'roberta')
    exp_name = exp_name.replace('sentence-transformers/paraphrase-MiniLM-L6-v2', 'st/paraphrase')

    if USE_WANDB:
        import wandb
        from pytorch_lightning.loggers import WandbLogger

        wandb_project = args.wandb_project_name
        wandb_entity = args.wandb_entity
        if args.loss_function == 'contrastive_cross_attention':
            wandb_project += '-cross-encoder'

        wandb_logger = WandbLogger(
            name=exp_name,
            project=wandb_project,
            config=vars(args),
            job_type='train',
            entity=wandb_entity,
            id=args.wandb_run_id  # None, or set to a str for resuming a run
        )
        wandb_logger.watch(model, log_graph=False)
        loggers.append(
            wandb_logger
        )
        ckpt_save_path = os.path.join("./", wandb_logger.name, wandb_logger.version, "checkpoints")
        if model.path_to_save_checkpoints == "":
            model.path_to_save_checkpoints = os.path.abspath(ckpt_save_path)

    dm = DataModule(
        debug_mode = args.debug_mode if args.debug_mode else False,
        local_data_path=args.local_data_path,
        dataset_source=args.dataset_source,
        document_model_name_or_path=document_model,
        profile_model_name_or_path=profile_model,
        max_seq_length=args.max_seq_length,
        dataset_name=args.dataset_name,
        dataset_train_split=args.dataset_train_split,
        dataset_val_split=args.dataset_val_split,
        dataset_test_split=args.dataset_test_split,
        dataset_version=args.dataset_version,
        word_dropout_ratio=args.word_dropout_ratio,
        word_dropout_perc=args.word_dropout_perc,
        profile_row_dropout_perc=args.profile_row_dropout_perc,
        adversarial_masking=args.adversarial_masking,
        idf_masking=args.idf_masking,
        sample_spans=args.sample_spans,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=round(num_cpus / torch.cuda.device_count()),
        num_nearest_neighbors=args.num_nearest_neighbors,
    )
    dm.setup("fit")

    from pytorch_lightning.loggers import CSVLogger
    # TODO set experiment name same as W&B run name?
    # TODO make this show up, I don't think it does 
    # (maybe because I usually kill runs before they finish?).
    loggers.append(CSVLogger("logs"))

    # TODO: argparse for val_metric?
    # val_metric = "val/document/loss"
    # val_metric = "val/document_redact_lexical/loss"
    # val_metric = "val/document_redact_ner/loss"
    # val_metric = "val/document_redact_adversarial_100/loss"
    val_metric = "val/document_redact_idf_total/loss"
    # val_metric = "train/loss"
    early_stopping_patience = (args.lr_scheduler_patience * 50 * args.num_validations_per_epoch)

    if torch.cuda.device_count() == 1:
        checkpoints = [
            ModelCheckpoint(
                monitor="val/document/loss",
                mode="min",
                filename="{epoch}-{step}",
                save_last=True,
                save_top_k=args.model_ckpt_save_top_k,
                save_on_train_epoch_end=True,
            ),
            # ModelCheckpoint(monitor="val/document_redact_adversarial_100/loss", mode="min", filename="{epoch}-{step}-adv100_loss", save_last=True, save_top_k=args.model_ckpt_save_top_k),
            ModelCheckpoint(
                monitor="val/document_redact_idf_total/loss",
                mode="min",
                filename="{epoch}-{step}-idf_total",
                save_last=True,
                save_top_k=args.model_ckpt_save_top_k,
                save_on_train_epoch_end=True
            )
        ]
    else:
        checkpoints = [
            ModelCheckpoint(
                monitor=None,
                filename="{epoch}-{step}",
                save_last=True,
                save_top_k=args.model_ckpt_save_top_k,
                save_on_train_epoch_end=True,
            ),
        ]

    callbacks = [
                    LearningRateMonitor(logging_interval='epoch'),
                    #
                    #
                    # EarlyStopping(monitor=val_metric, min_delta=0.00, patience=early_stopping_patience, verbose=True, mode="min")
                    #
                ] + checkpoints

    print("creating Trainer")
    trainer = Trainer(
        default_root_dir=f"saves/{exp_name}",
        val_check_interval=(1.0 / args.num_validations_per_epoch),
        callbacks=callbacks,
        max_epochs=args.epochs,
        log_every_n_steps=min(len(dm.train_dataloader()), 50),
        # limit_train_batches=10, # change this to make training faster (1.0 = full train set),
        limit_train_batches=1.0,
        limit_test_batches=0,
        limit_val_batches=args.limit_val_batches,
        devices=torch.cuda.device_count(),
        accelerator='gpu',
        logger=loggers,
        num_sanity_val_steps=0,
        strategy=('ddp' if torch.cuda.device_count() > 1 else 'auto'),
        precision=args.precision
    )
    trainer.fit(
        model=model,
        datamodule=dm,
        ckpt_path=checkpoint_path
    )


if __name__ == '__main__':
    main(get_args())
