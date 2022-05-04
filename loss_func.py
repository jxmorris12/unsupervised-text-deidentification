from typing import Dict

import abc

import torch

from .model import Model

class CoordinateAscentModel(Model):
    def on_train_epoch_start(self, epoch: int):
        # We only want to keep one model on GPU at a time.
        if self.document_encoder_is_training:
            self.train_document_embeddings = None
            self.val_document_embeddings = None
            self.val_document_redact_ner_embeddings = None
            self.val_document_redact_lexical_embeddings = None
            # 
            self.train_profile_embeddings = self.train_profile_embeddings.cuda()
            self.val_profile_embeddings = self.val_profile_embeddings.cuda()
            self.document_model.cuda()
            self.document_embed.cuda()
            self.profile_model.cpu()
        else:
            self.train_profile_embeddings = None
            self.val_profile_embeddings = None
            # 
            self.train_document_embeddings = self.train_document_embeddings.cuda()
            self.val_document_embeddings = self.val_document_embeddings.cuda()
            self.document_model.cpu()
            self.document_embed.cpu()
            self.profile_model.cuda()
        self.log("document_encoder_is_training", float(self.document_encoder_is_training))

    def _document_encoder_is_training(self, epoch: int) -> bool:
        """True if we're training the document encoder. If false, we are training the profile encoder.
        Should alternate during training epochs."""
        # TODO: separate loss func for pretrained prof encoder?
        if self.pretrained_profile_encoder:
            return True
        else:
            return self.current_epoch % 2 == 0

    def get_optimizer(self, epoch: int) -> torch.optim.Optimizer:
        document_optimizer, profile_optimizer = self.optimizers
        if self._document_encoder_is_training(epoch=epoch):
            return document_optimizer
        else:
            return profile_optimizer

    def get_scheduler(self):
        document_scheduler, profile_scheduler = self.lr_schedulers()
        scheduler = profile_scheduler
    
    def _training_step_document(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One step of training where training is supposed to update  `self.document_model`."""
        document_inputs, document_embeddings = self.forward_document(
            batch=batch, document_type='document', return_inputs=True
        )

        loss = self._compute_loss_exact(
            document_embeddings, self.train_profile_embeddings, batch['text_key_id'],
            metrics_key='train'
        )

        return {
            "loss": loss,
            "document_embeddings": document_embeddings.detach().cpu(),
            "text_key_id": batch['text_key_id'].cpu()
        }
    
    def _training_step_profile(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One step of training where training is supposed to update  `self.profile_model`."""
        profile_embeddings = self.forward_profile(batch=batch)

        loss = self._compute_loss_exact(
            profile_embeddings, self.train_document_embeddings, batch['text_key_id'],
            metrics_key='train'
        )

        return {
            "loss": loss,
            "profile_embeddings": profile_embeddings.detach().cpu(),
            "text_key_id": batch['text_key_id'].cpu()
        }

    def __call__(
            self, batch: Dict[str, torch.Tensor], batch_idx: int
        ) -> torch.Tensor:
        document_optimizer, profile_optimizer = self.optimizers()
        if self.document_encoder_is_training:
            optimizer = document_optimizer
            results = self._training_step_document(batch, batch_idx)
        else:
            optimizer = profile_optimizer
            results = self._training_step_profile(batch, batch_idx)

class ContrastiveModel(Model):
    def __init__(self, ??):
        self.scheduler = ??
        self.optimizer = ??

    def get_optimizer(self, epoch: int) -> torch.optim.Optimizer:
        return self.optimizer
    
    def get_scheduler(self):
        return self.scheduler
