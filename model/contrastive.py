from typing import Dict

import abc

import numpy as np
import torch
import tqdm

from .model import Model

class ContrastiveModel(Model):
    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizers()
        
    def _compute_loss_infonce(
            self,
            document_embeddings: torch.Tensor,
            profile_embeddings: torch.Tensor,
            metrics_key: str
        ) -> torch.Tensor:
        """InfoNCE matching loss between `document_embeddings` and `profile_embeddings`. Computes
        metrics and prints, prefixing with `metrics_key`.
         """
        assert len(document_embeddings.shape) == len(document_embeddings.shape) == 2 # [batch_dim, embedding_dim]
        assert document_embeddings.shape[1] == profile_embeddings.shape[1]
        batch_size = len(document_embeddings)
        # Normalize embeddings before computing similarity
        # Commented-out because the DPR paper ("Dense Passage Retrieval for Open-Domain Question Answering") reports
        # better results with dot-product than cosine similarity.
        # document_embeddings = document_embeddings / torch.norm(document_embeddings, p=2, dim=1, keepdim=True)
        # profile_embeddings = profile_embeddings / torch.norm(profile_embeddings, p=2, dim=1, keepdim=True)
        # Match documents to profiles
        document_to_profile_sim = torch.matmul(document_embeddings, profile_embeddings.T)
        document_to_profile_sim *= self.temperature.exp()
        diagonal_idxs = torch.arange(batch_size).to(document_embeddings.device)
        loss = torch.nn.functional.cross_entropy(
            document_to_profile_sim, diagonal_idxs
        )
        self.log(f"{metrics_key}/loss", loss)

        # Also track a boolean mask for which things were correct.
        is_correct = (document_to_profile_sim.argmax(dim=1) == diagonal_idxs)

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
            self.log(f"{metrics_key}/acc_top_k/{k}", top_k_acc)
        return document_to_profile_sim, is_correct, loss
    
    def compute_loss(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        document_embeddings = self.forward_document(
            batch=batch, document_type='document'
        )
        profile_embeddings = self.forward_profile(batch=batch)

        # If keys that start with 'profile_neighbor' are present in the batch,
        # include those in the loss.
        profile_neighbors_present = len(
            self._get_inputs_from_prefix(batch=batch, prefix='profile_neighbor')
        )
        if profile_neighbors_present > 0:
            extra_profile_embeddings = self.forward_profile(
                batch=batch, profile_key='profile_neighbor', collapse_axis=True
            )
            profile_embeddings = torch.cat(
                (profile_embeddings, extra_profile_embeddings), dim=0
            )
            document_to_profile_sim, is_correct, loss = self._compute_loss_infonce(
                document_embeddings=document_embeddings,
                profile_embeddings=profile_embeddings,
                metrics_key='train'
            )
            # idxs = torch.cat(
            #     (batch['text_key_id'], batch['profile_neighbor_idxs']), dim=0)
            # TODO:: somehow update nearest-neighbors here to take top-K of document2profilesim along axis 1.
            # self.trainer.datamodule.compute_new_nearest_neighbors(document_to_profile_sim, idxs)
        else:
            _, is_correct, loss = self._compute_loss_infonce(
                document_embeddings=document_embeddings,
                profile_embeddings=profile_embeddings,
                metrics_key='train'
            )
        return {
            "loss": loss,
            "document_embeddings": document_embeddings.detach().cpu(),
            "profile_embeddings": profile_embeddings.detach().cpu(),
            "is_correct": is_correct.cpu(),
            "text_key_id": batch['text_key_id'].cpu()
        }

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

