from typing import Dict

import abc

import numpy as np
import torch
import tqdm

from transformers import AdamW

from .model import Model

class ContrastiveModel(Model):
    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizers()
    
    def get_scheduler(self):
        return self.lr_schedulers()
        
    def _compute_loss_infonce(self, document_embeddings: torch.Tensor, profile_embeddings: torch.Tensor,
        metrics_key: str) -> torch.Tensor:
        """InfoNCE matching loss between `document_embeddings` and `profile_embeddings`. Computes
        metrics and prints, prefixing with `metrics_key`.
         """
        assert document_embeddings.shape == profile_embeddings.shape
        assert len(document_embeddings.shape) == 2 # [batch_dim, embedding_dim]
        batch_size = len(document_embeddings)
        # Normalize embeddings before computing similarity
        document_embeddings = document_embeddings / torch.norm(document_embeddings, p=2, dim=1, keepdim=True)
        profile_embeddings = profile_embeddings / torch.norm(profile_embeddings, p=2, dim=1, keepdim=True)
        # Match documents to profiles
        document_to_profile_sim = (
            (torch.matmul(document_embeddings, profile_embeddings.T) * self.temperature.exp())
        )
        diagonal_idxs = torch.arange(batch_size).to(document_embeddings.device)
        loss = torch.nn.functional.cross_entropy(
            document_to_profile_sim, diagonal_idxs
        )
        self.log(f"{metrics_key}/loss", loss)
        # Log top-k accuracies.
        for k in [1, 5, 10, 50, 100, 500, 1000]:
            if k >= batch_size: # can't compute top-k accuracy here.
                continue
            top_k_acc = (
                document_to_profile_sim.topk(k=k, dim=1)
                    .indices
                    .eq(diagonal_idxs[:, None])
                    .any(dim=1)
                    .float()
                    .mean()
            )
            self.log(f"{metrics_key}/acc_top_k/{k}",top_k_acc)
        return loss
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        document_embeddings = self.forward_document(
            batch=batch, document_type='document'
        )
        profile_embeddings = self.forward_profile(
            batch=batch
        )
        loss = self._compute_loss_infonce(
            document_embeddings=document_embeddings,
            profile_embeddings=profile_embeddings,
            metrics_key='train'
        )
        return {
            "loss": loss,
            "document_embeddings": document_embeddings.detach().cpu(),
            "profile_embeddings": profile_embeddings.detach().cpu(),
            "text_key_id": batch['text_key_id'].cpu()
        }


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
        return [optimizer], [scheduler]

