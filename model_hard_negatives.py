from typing import Dict, Optional

import os
import pickle

import numpy as np
import torch

from pytorch_lightning import LightningModule
from sentence_transformers import SentenceTransformer
from transformers import AdamW, AutoConfig, AutoModel


# TODO: make a better name for this class...
class DocumentProfileMatchingTransformerWithHardNegatives(LightningModule):
    """Encodes profiles using pre-computed encodings. Uses nearest-neighbors
    to create 'hard negatives' to get similarity.
    """
    embeddings: np.ndarray      # float ndarray, shape (train_set_len, prof_emb_dim) 
                                # example: (58266, 384)
                                # -- for wiki_bio['train:10%'] and sentence-transformers/paraphrase-MiniLM-L6-v2 encoding,

    neighbors: np.ndarray       # int ndarray, shape (train_set_len, total_num_nearest_neighbors)
                                # example: (58266, 128) 

    num_neighbors: int          # number of neighbors to use per datapoint

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.document_model = AutoModel.from_pretrained(model_name_or_path)
        self.lower_dim_embed = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=768, out_features=384), # TODO: make these numbers a feature of model/embedding type
        )
        self.temperature = torch.nn.parameter.Parameter(
            torch.tensor(5.0, dtype=torch.float32), requires_grad=True)

        # Load precomputed stuff from disk
        split = 'train[:10%]' # TODO: argparse for split
        k = 128 # TODO: argparse for k (num nearest neighbors?)
        save_folder = os.path.join('precomputed_similarities', f'{dataset_name}__{split}__{k}')
        assert os.path.exists(save_folder), f'no precomputed similarities at folder {save_folder}'
        neighbors_path = os.path.join(save_folder, 'neighbors.p')
        self.neighbors = np.array(pickle.load(open(neighbors_path, 'rb')))
        embeddings_path = os.path.join(save_folder, 'embeddings.p')
        self.embeddings = pickle.load(open(embeddings_path, 'rb'))
        self.num_neighbors = 16

        # self.profile_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

        print(f'Initialized DocumentProfileMatchingTransformer with learning_rate = {learning_rate}')

    def forward(self, **inputs):
        return self.document_model(**inputs)

    def _compute_loss(self, document_embeddings: torch.Tensor, profile_embeddings: torch.Tensor,
        metrics_key: str) -> torch.Tensor:
        """InfoNCE matching loss between `document_embeddings` and `profile_embeddings`. Computes
        metrics and prints, prefixing with `metrics_key`.

        Args:
            document_embeddings (float torch.Tensor): Embeddings for each document in batch, of shape
                (batch_size, emb_dim)
            profile_embeddings (float torch.Tensor): Embeddings for each profile in the `self.num_neighbors`
                nearest-neighbors of an element in the batch, of shape (batch_size, self.num_neighbors, emb_dim)

        Returns:
            loss (scalar float torch.Tensor)
         """
        assert (document_embeddings.shape[0], self.num_neighbors, document_embeddings.shape[1]) == profile_embeddings.shape
        assert len(document_embeddings.shape) == 2 # [batch_dim, embedding_dim]
        batch_size = len(document_embeddings)
        # Normalize embeddings before computing similarity
        document_embeddings = document_embeddings / torch.norm(document_embeddings, p=2, dim=1, keepdim=True)
        profile_embeddings = profile_embeddings / torch.norm(profile_embeddings, p=2, dim=2, keepdim=True)
        # Match documents to profiles
        document_to_profile_sim = (
            (torch.einsum('be,bke->bk', document_embeddings, profile_embeddings) * self.temperature.exp())
        )
        diagonal_idxs = torch.arange(batch_size).to(document_embeddings.device)
        # TODO: do we want loss in one direction or both...?
        loss_d2p = torch.nn.functional.cross_entropy(
            document_to_profile_sim, diagonal_idxs
        )
        loss_p2d = torch.nn.functional.cross_entropy(
            document_to_profile_sim.T, diagonal_idxs
        )
        diagonal_avg_prob = torch.diagonal(torch.nn.functional.softmax(document_to_profile_sim, dim=1)).mean()
        self.log(f"{metrics_key}/diagonal_avg_prob", diagonal_avg_prob)
        self.log(f"{metrics_key}/loss_d2p", loss_d2p)
        self.log(f"{metrics_key}/loss_p2d", loss_p2d)
        self.log(f"{metrics_key}/loss", loss_d2p + loss_p2d)
        return loss_d2p + loss_p2d


    def _compute_loss(self, document_embeddings: torch.Tensor, profile_embeddings: torch.Tensor,
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
            (torch.matmul(profile_embeddings, document_embeddings.T) * self.temperature.exp())
        )
        # We want each document to match to itself and no other
        true_idxs = torch.zeros_like(document_to_profile_sim).to(document_embeddings.device)
        true_idxs[:, 0] = 1 
        # TODO: do we want loss in one direction or both...?
        loss_d2p = torch.nn.functional.cross_entropy(
            document_to_profile_sim, true_idxs
        )
        loss_p2d = torch.tensor(0.0).to(document_embeddings.device)
        true_avg_prob = torch.nn.functional.softmax(document_to_profile_sim, dim=1)[:, 0].mean()
        self.log(f"{metrics_key}/true_avg_prob", true_avg_prob)
        self.log(f"{metrics_key}/loss_d2p", loss_d2p)
        self.log(f"{metrics_key}/loss_p2d", loss_p2d)
        self.log(f"{metrics_key}/loss", loss_d2p + loss_p2d)
        return loss_d2p + loss_p2d

    def _get_profile_embeddings(self, profile_idxs: torch.Tensor) -> torch.Tensor:
        """Gets profile embeddings from a list of indices.

        Args:
            profile_idxs (int torch.Tensor): indices of profiles to get embeddings for, of shape (batch)

        Returns:
            profile_embeddings (float torch.Tensor) embeddings of shape (batch, self.num_neighbors, prof_emb_dim)
        """
        neighbor_idxs = self.neighbors[profile_idxs][:, self.num_neighbors] # (batch, self.num_neighbors)
        return torch.tensor(self.embeddings[neighbor_idxs]).to(self.device) # (batch, self.num_neighbors, prof_emb_dim)


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return {}
        # with torch.no_grad():
        #     # Getting output['last_hidden_state'][:, 0, :] is
        #     # getting the CLS token from BERT
        #     document_embeddings = self.document_model(
        #         input_ids=batch['text1_input_ids'],
        #         attention_mask=batch['text1_attention_mask']
        #     )['last_hidden_state']
        #     document_embeddings = document_embeddings[:, 0, :]
            
        #     profile_embeddings = self._get_profile_embeddings(batch['text_key_id'].cpu())
        # return {
        #     "document_embeddings": document_embeddings,
        #     "profile_embeddings": profile_embeddings,
        #     "loss": self._compute_loss(document_embeddings, profile_embeddings, 'val_approx')

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # TODO(jxm): should we use two different models for these encodings?
        document_embeddings = self.document_model(
            input_ids=batch['text1_input_ids'],
            attention_mask=batch['text1_attention_mask']
        )
        document_embeddings = document_embeddings['last_hidden_state'][:, 0, :] # (batch, document_emb_dim)
        document_embeddings = self.lower_dim_embed(document_embeddings) # (batch, document_emb_dim) -> (batch, prof_emb_dim)
        # document_embeddings = document_embeddings['last_hidden_state'].mean(dim=1)
        # Just take last hidden state at index 0 which should be CLS. TODO(jxm): is this right?
        profile_embeddings = self._get_profile_embeddings(batch['text_key_id'].cpu())
        # self.embeddings[batch['text_key_id'].cpu()].shape -> (batch_size, prof_emb_dim)
        # self.neighbors[batch['text_key_id'].cpu()].shape -> (batch_size, k)
        loss = self._compute_loss(document_embeddings, profile_embeddings, 'train')
        return loss

    # def validation_epoch_end(self, outputs) -> torch.Tensor:
    #     document_embeddings = torch.cat([o['document_embeddings'] for o in outputs], axis=0)
    #     profile_embeddings = torch.cat([o['profile_embeddings'] for o in outputs], axis=0)
    #     assert document_embeddings.shape == profile_embeddings.shape
    #     # Normalize embeddings before computing similarity
    #     document_embeddings = document_embeddings / torch.norm(document_embeddings, p=2, dim=1, keepdim=True)
    #     profile_embeddings = profile_embeddings / torch.norm(profile_embeddings, p=2, dim=1, keepdim=True)
    #     # TODO: is there a natural way to scale temperature here?
    #     document_to_profile_sim = (
    #         (torch.matmul(profile_embeddings, document_embeddings.T) * self.temperature.exp())
    #     )
    #     batch_size = len(document_embeddings)
    #     diagonal_idxs = torch.arange(batch_size).to(document_embeddings.device)
    #     loss_d2p = torch.nn.functional.cross_entropy(
    #         document_to_profile_sim, diagonal_idxs
    #     )
    #     loss_p2d = torch.nn.functional.cross_entropy(
    #         document_to_profile_sim.T, diagonal_idxs
    #     )
    #     diagonal_avg_prob = torch.diagonal(torch.nn.functional.softmax(document_to_profile_sim, dim=1)).mean()
    #     # TODO: support batching here if val data is too big?
    #     # TODO: plot some interesting mispredictions
    #     # TODO: downscale losses
    #     self.log("val_exact/diagonal_avg_prob", diagonal_avg_prob)
    #     self.log("val_exact/loss_d2p", loss_d2p)
    #     self.log("val_exact/loss_p2d", loss_p2d)
    #     self.log("val_exact/loss", loss_d2p + loss_p2d)
    #     loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     return loss

    def setup(self, stage=None) -> None:
        """Sets stuff up. Called once before training."""
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.document_model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return optimizer
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_steps,
        #     num_training_steps=self.total_steps,
        # )
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # return [optimizer], [scheduler]