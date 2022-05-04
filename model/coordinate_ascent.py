from typing import Dict

import abc

import torch

from .model import Model

class CoordinateAscentModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(args=*args, kwargs=**kwargs)
        self.train_document_embeddings = None
        self.train_profile_embeddings = None

    def setup(self, stage=None) -> None:
        super().setup(stage=stage)
        if stage != "fit":
            return
        # Precompute embeddings
        self.precompute_profile_embeddings()
    
    def _precompute_profile_embeddings(self):
        self.profile_model.cuda()
        self.profile_model.eval()
        print('Precomputing profile embeddings before first epoch...')
        self.train_profile_embeddings = np.zeros((len(self.trainer.datamodule.train_dataset), self.profile_embedding_dim))
        for train_batch in tqdm.tqdm(self.trainer.datamodule.train_dataloader(), desc="[1/2] Precomputing train embeddings", colour="magenta", leave=False):
            with torch.no_grad():
                profile_embeddings = self.forward_profile(batch=train_batch)
            self.train_profile_embeddings[train_batch["text_key_id"]] = profile_embeddings.cpu()
        self.train_profile_embeddings = torch.tensor(self.train_profile_embeddings, dtype=torch.float32)
        self.profile_model.train()

    def on_train_epoch_start(self):
        # We only want to keep one model on GPU at a time.
        if self._document_encoder_is_training:
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
        self.log("document_encoder_is_training", float(self._document_encoder_is_training))

    def training_epoch_end(self, training_step_outputs: Dict):
        if self.document_encoder_is_training:
            self.train_document_embeddings = np.zeros((len(self.trainer.datamodule.train_dataset), self.profile_embedding_dim))
            for output in training_step_outputs:
                self.train_document_embeddings[output["text_key_id"]] = output["document_embeddings"]
            self.train_document_embeddings = torch.tensor(self.train_document_embeddings, requires_grad=False, dtype=torch.float32)
        else:
            # TODO: fix this as it assumes profile and doc embeddings are the same shape.
            self.train_profile_embeddings = np.zeros((len(self.trainer.datamodule.train_dataset), self.profile_embedding_dim))
            for output in training_step_outputs:
                self.train_profile_embeddings[output["text_key_id"]] = output["profile_embeddings"]
            self.train_profile_embeddings = torch.tensor(self.train_profile_embeddings, requires_grad=False, dtype=torch.float32)
            self.train_document_embeddings = None

    def _document_encoder_is_training(self) -> bool:
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
        if self._document_encoder_is_training(epoch=epoch):
            return document_scheduler
        else:
            return profile_scheduler
    
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

    def _get_optimizer(
            self, batch: Dict[str, torch.Tensor], batch_idx: int
        ) -> torch.Tensor:
        document_optimizer, profile_optimizer = self.optimizers()
        if self._document_encoder_is_training:
            return document_optimizer
            results = self._training_step_document(batch, batch_idx)
        else:
            optimizer = profile_optimizer
            results = self._training_step_profile(batch, batch_idx)
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
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
        document_optimizer = AdamW(
            list(self.document_model.parameters()) + list(self.document_embed.parameters()) + [self.temperature], lr=self.document_learning_rate, eps=self.hparams.adam_epsilon
        )
        document_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            document_optimizer, mode='min',
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=1e-10
        )
        # see source:
        # pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html?highlight=optimizer_step#optimizer-step
        document_scheduler = {
            "scheduler": document_scheduler,
            "name": "document_learning_rate",
        }

        profile_optimizer = AdamW(
            list(self.profile_model.parameters()) + [self.temperature], lr=self.profile_learning_rate, eps=self.hparams.adam_epsilon
        )
        profile_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            profile_optimizer, mode='min',
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=1e-10
        )
        profile_scheduler = {
            "scheduler": profile_scheduler,
            "name": "profile_learning_rate",
        }
        # TODO: Consider adding a scheduler for word dropout?

        return [document_optimizer, profile_optimizer], [document_scheduler, profile_scheduler]


