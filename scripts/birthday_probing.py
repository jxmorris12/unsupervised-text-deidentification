import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

from typing import Any, Dict, List, Tuple

import argparse
import datetime
import os
import re

import datasets
import numpy as np
import torch
import torchmetrics
import transformers

from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm

from dataloader import WikipediaDataModule
from model import CoordinateAscentModel
from model_cfg import model_paths_dict

num_cpus = len(os.sched_getaffinity(0))
USE_WANDB = False

def process_dataset_example(ex: Dict) -> Dict[str, Any]:
    """Parses example. Returns tuple (idx, month, day) or None."""
    profile = ex['profile']
    date_str_matches = re.search(r"birth_date \|\| ([\d]{1,4} [a-z]+ [\d]{1,4})", profile)
    try:
        date_str = date_str_matches.group(1)
        dt = datetime.datetime.strptime(date_str, "%d %B %Y")
        ex['birthday'] = (ex['text_key_id'], dt.month-1, dt.day-1)
    except (AttributeError, ValueError) as e:
        # no birthdate or date in unknown format (just skip these ones)
        ex['birthday'] = None
    return ex

def process_dataset(dataset: datasets.Dataset) -> List[Tuple[int, int, int]]:
    new_dataset = dataset.map(process_dataset_example)
    filtered_dataset = new_dataset.filter(lambda ex: ex['birthday'])
    return filtered_dataset['birthday']

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a model using probing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_key', type=str, default='model_8_ls0.1', help='path to model key (see model_cfg.py)')
    parser.add_argument('--dataset_train_split', type=str, default='train[:10%]')
    parser.add_argument('--num_validations_per_epoch', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    args = parser.parse_args()
    args.dataset_val_split = 'val[:20%]'

    return args

class BirthdayDataModule(LightningDataModule):
    train_dataset: List[Tuple[int, int, int]]
    val_dataset: List[Tuple[int, int, int]]
    batch_size: int
    def __init__(self, dm: WikipediaDataModule, batch_size: int = 128):
        super().__init__()
        self.train_dataset = process_dataset(dm.train_dataset)
        self.val_dataset = process_dataset(dm.val_dataset)
        self.batch_size = batch_size
        self.num_workers = min(4, num_cpus)

    def setup(self, stage: str) -> None:
        return

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False # Only shuffle for train
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )


class BirthdayModel(LightningModule):
    """Probes the PROFILE for birthday info."""
    profile_embeddings: torch.Tensor
    classifier: torch.nn.Module
    learning_rate: float
    
    def __init__(self, model: CoordinateAscentModel, learning_rate: float):
        super().__init__()
        # We can pre-calculate these embeddings bc
        self.train_profile_embeddings = torch.tensor(model.train_profile_embeddings.cpu())
        self.val_profile_embeddings = torch.tensor(model.val_profile_embeddings.cpu())
        self.month_classifier = torch.nn.Sequential(
            torch.nn.Linear(model.shared_embedding_dim, 64),
            torch.nn.Linear(64, 12),
        )
        self.day_classifier = torch.nn.Sequential(
            torch.nn.Linear(model.shared_embedding_dim, 64),
            torch.nn.Linear(64, 31),
        )
        self.learning_rate = learning_rate
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy   = torchmetrics.Accuracy()
        self.loss_criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch: Tuple[int, int], batch_idx: int) -> torch.Tensor:
        profile_idxs, months, days = batch
        assert ((0 <= profile_idxs) & (profile_idxs < len(self.train_profile_embeddings))).all()
        assert ((0 <= months) & (months < 12)).all()
        assert ((0 <= days) & (days < 31)).all()
        
        clf_device = next(self.month_classifier.parameters()).device
        with torch.no_grad():
            embedding = self.train_profile_embeddings[profile_idxs].to(clf_device)
        
        
        month_logits = self.month_classifier(embedding)
        day_logits = self.day_classifier(embedding)
        
        
        month_loss = torch.nn.functional.cross_entropy(month_logits, months)
        day_loss = torch.nn.functional.cross_entropy(day_logits, days)
        
        self.log('train_acc_month', self.train_accuracy(month_logits, months))
        self.log('train_acc_day', self.train_accuracy(day_logits, days))
        
        if batch_idx == 0:
            print(
                'train_acc_month', self.train_accuracy(month_logits, months).item(), 
                '//', 
                'train_acc_day', self.train_accuracy(day_logits, days)
            )
        
        return (month_loss + day_loss)
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        profile_idxs, months, days = batch
        assert ((0 <= profile_idxs) & (profile_idxs < len(self.val_profile_embeddings))).all()
        assert ((0 <= months) & (months < 12)).all()
        assert ((0 <= days) & (days < 31)).all()
        
        clf_device = next(self.month_classifier.parameters()).device
        with torch.no_grad():
            embedding = self.val_profile_embeddings[profile_idxs].to(clf_device)
        
        
        month_logits = self.month_classifier(embedding)
        day_logits = self.day_classifier(embedding)
        
        
        month_loss = torch.nn.functional.cross_entropy(month_logits, months)
        day_loss = torch.nn.functional.cross_entropy(day_logits, days)
        
        self.log('val_acc_month', self.val_accuracy(month_logits, months))
        self.log('val_acc_day', self.val_accuracy(day_logits, days))
        
        if batch_idx == 0:
            print('val_acc_month', self.val_accuracy(month_logits, months))
            print('val_acc_day', self.val_accuracy(day_logits, days))

        return (month_loss + day_loss)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(
            list(self.parameters()), lr=self.learning_rate
        )
        return optimizer
            

def precompute_profile_embeddings(model: CoordinateAscentModel, dm: WikipediaDataModule):
    model.profile_model.cuda()
    model.profile_model.eval()
    model.profile_embed.cuda()
    model.profile_embed.eval()


    model.train_profile_embeddings = np.zeros((len(dm.train_dataset), model.shared_embedding_dim))
    for train_batch in tqdm(dm.train_dataloader(), desc="Precomputing train embeddings", colour="#008080", leave=False):
        with torch.no_grad():
            profile_embeddings = model.forward_profile(batch=train_batch)
        model.train_profile_embeddings[train_batch["text_key_id"]] = profile_embeddings.cpu()
    model.train_profile_embeddings = torch.tensor(model.train_profile_embeddings, dtype=torch.float32)
    
    model.val_profile_embeddings = np.zeros((len(dm.val_dataset), model.shared_embedding_dim))
    for val_batch in tqdm(dm.val_dataloader()[0], desc="Precomputing val embeddings", colour="#c09591", leave=False):
        with torch.no_grad():
            profile_embeddings = model.forward_profile(batch=val_batch)
        model.val_profile_embeddings[val_batch["text_key_id"]] = profile_embeddings.cpu()
    model.val_profile_embeddings = torch.tensor(model.val_profile_embeddings, dtype=torch.float32)


def main(args: argparse.Namespace):
    seed_everything(42)

    num_cpus = len(os.sched_getaffinity(0))

    checkpoint_path = model_paths_dict[args.model_key]
    print(checkpoint_path)

    model = CoordinateAscentModel.load_from_checkpoint(
        checkpoint_path
    )

    dm = WikipediaDataModule(
        document_model_name_or_path=model.document_model_name_or_path,
        profile_model_name_or_path=model.profile_model_name_or_path,
        max_seq_length=128,
        dataset_name='wiki_bio',
        dataset_train_split=args.dataset_train_split,
        dataset_val_split=args.dataset_val_split,
        dataset_version='1.2.0',
        num_workers=num_cpus,
        train_batch_size=64,
        eval_batch_size=64
    )
    dm.setup("fit")
    
    precompute_profile_embeddings(model=model, dm=dm)
    birthday_model = BirthdayModel(model, args.learning_rate)
    
    birthday_dm = BirthdayDataModule(dm)
    birthday_dm.batch_size = args.batch_size

    loggers = []
    if USE_WANDB:
        exp_name = f'{args.model_key}__{args.dataset_train_split}'
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(
            name=exp_name,
            project='deid-wikibio-probing', 
            config=vars(args),
            job_type='train',
            entity='jack-morris',
            id=args.wandb_run_id # None, or set to a str for resuming a run
        )
        wandb_logger.watch(model, log_graph=False)
        loggers.append(
            wandb_logger
        )

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    trainer = Trainer(
        default_root_dir=f"saves/jup/birthday_probing",
        val_check_interval=1/args.num_validations_per_epoch,
        max_epochs=args.epochs,
        log_every_n_steps=500,
        gpus=torch.cuda.device_count(),
        logger=loggers,
    )
    trainer.fit(birthday_model, birthday_dm)

if __name__ == '__main__':
    args = get_args()
    main(args)