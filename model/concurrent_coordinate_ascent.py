from typing import Dict

import abc

import numpy as np
import torch
import tqdm

from .coordinate_ascent import CoordinateAscentModel

class ConcurrentCoordinateAscentModel(CoordinateAscentModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_document_embeddings = None
        self.train_profile_embeddings = None

    def on_train_epoch_start(self):
        #
        self._precompute_profile_embeddings()
        self.train_profile_embeddings = self.train_profile_embeddings.cuda()
        # 
        self._precompute_document_embeddings()
        self.train_document_embeddings = self.train_document_embeddings.cuda()
        # 
        # We want both models on GPU.
        # 
        self.document_model.cuda()
        self.document_embed.cuda()
        self.profile_model.cuda()
        self.profile_embed.cuda()
        # 
        self.document_model.train()
        self.document_embed.train()
        self.profile_model.train()
        self.profile_embed.train()
        #

    def training_epoch_end(self, training_step_outputs: Dict):
        # 
        self.train_document_embeddings = np.zeros((len(self.trainer.datamodule.train_dataset), self.shared_embedding_dim))
        for output in training_step_outputs:
            self.train_document_embeddings[output["text_key_id"]] = output["document_embeddings"]
        # 
        self.train_profile_embeddings = np.zeros((len(self.trainer.datamodule.train_dataset), self.shared_embedding_dim))
        for output in training_step_outputs:
            self.train_profile_embeddings[output["text_key_id"]] = output["profile_embeddings"]
        # 
        self.train_document_embeddings = torch.tensor(self.train_document_embeddings, requires_grad=False, dtype=torch.float32)
        self.train_profile_embeddings = torch.tensor(self.train_profile_embeddings, requires_grad=False, dtype=torch.float32)

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizers()
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        doc_results = self._training_step_document(
            batch=batch,
            batch_idx=batch_idx
        )
        prof_results = self._training_step_profile(
            batch=batch,
            batch_idx=batch_idx
        )
        # Combine results into a single dict.
        results = {}
        results["text_key_id"] = batch['text_key_id'].cpu()
        results["document_embeddings"] = doc_results["document_embeddings"]
        results["profile_embeddings"] = prof_results["profile_embeddings"]
        results["loss"] = doc_results["loss"] + prof_results["loss"]
        results["is_correct"] = torch.logical_and(
            doc_results["is_correct"],
            prof_results["is_correct"]
        )
        return results

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = torch.optim.AdamW(
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
            verbose=True,
            min_lr=1e-10
        )
        scheduler = {
            "scheduler": scheduler,
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]


