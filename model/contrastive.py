from typing import Dict

import abc

import torch

from .model import Model

class ContrastiveModel(Model):
    def __init__(self, ??):
        self.scheduler = ??
        self.optimizer = ??

    def get_optimizer(self, epoch: int) -> torch.optim.Optimizer:
        return self.optimizers()
    
    def get_scheduler(self):
        return self.lr_schedulers()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(
            (
                list(self.document_model.parameters()) + 
                list(self.document_embed.parameters()) + 
                list(self.profile_model.parameters()) +
                [self.temperature]
            ), lr=self.document_learning_rate,
            eps=self.hparams.adam_epsilon
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=1e-10
        )
        scheduler = {
            "scheduler": scheduler,
            "name": "learning_rate",
        }
        return optimizer, scheduler

