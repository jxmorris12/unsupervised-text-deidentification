from typing import Dict

import abc
import collections

import numpy as np
import torch
import tqdm

from .model import Model

class CoordinateAscentModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_document_embeddings = None
        self.train_profile_embeddings = None
    
    def _precompute_profile_embeddings(self):
        self.profile_model.to(self.device)
        self.profile_embed.to(self.device)
        # In my experiments, train mode here seems to work as well or better.
        # self.profile_model.eval()
        self.profile_model.train()
        self.profile_embed.train()
        # print(f'Precomputing profile embeddings at epoch {self.current_epoch}...')
        self.train_profile_embeddings = np.zeros((len(self.trainer.datamodule.train_dataset), self.shared_embedding_dim))
        for train_batch in tqdm.tqdm(self.trainer.train_dataloader.loaders, desc="Precomputing profile embeddings", colour="magenta", leave=False):
            with torch.no_grad():
                profile_embeddings = self.forward_profile(batch=train_batch)
            self.train_profile_embeddings[train_batch["text_key_id"]] = profile_embeddings.cpu()
        self.train_profile_embeddings = torch.tensor(self.train_profile_embeddings, dtype=torch.float32)
    
    def _precompute_document_embeddings(self):
        self.document_model.to(self.device)
        # In my experiments, train mode here seems to work as well or better.
        # self.document_model.eval()
        self.document_model.train()
        # print(f'Precomputing document embeddings at epoch {self.current_epoch}...')
        self.train_document_embeddings = np.zeros((len(self.trainer.datamodule.train_dataset), self.shared_embedding_dim))
        for train_batch in tqdm.tqdm(self.trainer.train_dataloader.loaders, desc="Precomputing document embeddings", colour="magenta", leave=False):
            with torch.no_grad():
                document_embeddings = self.forward_document(batch=train_batch, document_type='document')
            self.train_document_embeddings[train_batch["text_key_id"]] = document_embeddings.cpu()
        self.train_document_embeddings = torch.tensor(self.train_document_embeddings, dtype=torch.float32)
        self.document_model.train()

    def on_train_epoch_start(self):
        document_optimizer, profile_optimizer = self.optimizers()
        # We only want to keep one model on GPU at a time during training.
        if self._document_encoder_is_training:
            self.train_document_embeddings = None
            self._precompute_profile_embeddings()
            self.train_profile_embeddings = self.train_profile_embeddings.to(self.device)
            # 
            self.document_model.to(self.device)
            self.document_model.train()
            self.document_embed.to(self.device)
            self.document_embed.train()
            self.profile_model.cpu()
            self.profile_embed.cpu()
            #  Reset optimizer state
            document_optimizer.state = collections.defaultdict(dict)
        else:
            self.train_profile_embeddings = None
            self._precompute_document_embeddings()
            self.train_document_embeddings = self.train_document_embeddings.to(self.device)
            # 
            self.document_model.cpu()
            self.document_embed.cpu()
            self.profile_model.to(self.device)
            self.profile_model.train()
            self.profile_embed.to(self.device)
            self.profile_embed.train()
            #  Reset optimizer state
            profile_optimizer.state = collections.defaultdict(dict)
        self.log("document_encoder_is_training", float(self._document_encoder_is_training))

    def training_epoch_end(self, training_step_outputs: Dict):
        if self._document_encoder_is_training:
            # self.train_document_embeddings = np.zeros((len(self.trainer.datamodule.train_dataset), self.shared_embedding_dim))
            # for output in training_step_outputs:
            #     self.train_document_embeddings[output["text_key_id"]] = output["document_embeddings"]
            # self.train_document_embeddings = torch.tensor(self.train_document_embeddings, requires_grad=False, dtype=torch.float32)
            self.train_profile_embeddings = None
        else:
            # self.train_profile_embeddings = np.zeros((len(self.trainer.datamodule.train_dataset), self.shared_embedding_dim))
            # for output in training_step_outputs:
            #     self.train_profile_embeddings[output["text_key_id"]] = output["profile_embeddings"]
            # self.train_profile_embeddings = torch.tensor(self.train_profile_embeddings, requires_grad=False, dtype=torch.float32)
            self.train_document_embeddings = None

    @property
    def _document_encoder_is_training(self) -> bool:
        """True if we're training the document encoder. If false, we are training the profile encoder.
        Should alternate during training epochs."""
        # if self.pretrained_profile_encoder:
        if self.pretrained_profile_encoder or not (self.current_epoch in [1, 5, 13, 45,  104]):
            return True
        else:
            return self.current_epoch % 2 == 0

    def get_optimizer(self) -> torch.optim.Optimizer:
        document_optimizer, profile_optimizer = self.optimizers()
        if self._document_encoder_is_training:
            return document_optimizer
        else:
            return profile_optimizer
    
    def _training_step_document(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One step of training where training is supposed to update  `self.document_model`."""
        document_inputs, document_embeddings = self.forward_document(
            batch=batch, document_type='document', return_inputs=True
        )

        is_correct, loss = self._compute_loss_exact(
            document_embeddings, self.train_profile_embeddings, batch['text_key_id'],
            metrics_key='train'
        )

        return {
            # can't return bools or ints if using torch DistributedDataParallel
            # "is_correct": is_correct,
            "loss": loss,
            "document_embeddings": document_embeddings.detach().cpu(),
            # "text_key_id": batch['text_key_id'].cpu()
        }
    
    def _training_step_profile(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One step of training where training is supposed to update  `self.profile_model`."""
        profile_embeddings = self.forward_profile(batch=batch)

        is_correct, loss = self._compute_loss_exact(
            profile_embeddings, self.train_document_embeddings, batch['text_key_id'],
            metrics_key='train'
        )

        return {
            # can't return bools or ints if using torch DistributedDataParallel
            # "is_correct": is_correct,
            "loss": loss,
            "profile_embeddings": profile_embeddings.detach().cpu(),
            # "text_key_id": batch['text_key_id'].cpu()
        }
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        if self._document_encoder_is_training:
            return self._training_step_document(
                batch=batch,
                batch_idx=batch_idx
            )
        else:
            return self._training_step_profile(
                batch=batch,
                batch_idx=batch_idx
            )

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        document_optimizer = torch.optim.AdamW(
            list(self.document_model.parameters()) + list(self.document_embed.parameters()) + [self.temperature], lr=self.document_learning_rate, eps=self.hparams.adam_epsilon
        )
        # document_optimizer = torch.optim.SGD(
        #     list(self.document_model.parameters()) + list(self.document_embed.parameters()) + [self.temperature],
        #     lr=self.document_learning_rate, momentum=0.9
        # )
        document_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            document_optimizer,
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            verbose=True,
            mode='min', # 'min'  for loss, 'max' for acc
            min_lr=1e-10
        )
        # see source:
        # pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html?highlight=optimizer_step#optimizer-step
        document_scheduler = {
            "scheduler": document_scheduler,
            "name": "document_learning_rate",
        }

        profile_optimizer = torch.optim.AdamW(
            list(self.profile_model.parameters()) + list(self.profile_embed.parameters()) + [self.temperature], lr=self.profile_learning_rate, eps=self.hparams.adam_epsilon
        )
        # profile_optimizer = torch.optim.SGD(
        #     list(self.document_model.parameters()) + list(self.document_embed.parameters()) + [self.temperature],
        #     lr=self.profile_learning_rate, momentum=0.9
        # )
        profile_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            profile_optimizer,
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            verbose=True,
            mode='min', # 'min'  for loss, 'max' for acc
            min_lr=1e-10
        )
        profile_scheduler = {
            "scheduler": profile_scheduler,
            "name": "profile_learning_rate",
        }
        # TODO: Consider adding a scheduler for word dropout?

        return [document_optimizer, profile_optimizer], [document_scheduler, profile_scheduler]


