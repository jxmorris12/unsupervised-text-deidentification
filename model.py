from typing import Dict, Optional

import os
import pickle

import numpy as np
import torch

from pytorch_lightning import LightningModule
from sentence_transformers import SentenceTransformer
from transformers import AdamW, AutoConfig, AutoModel


# TODO: make a better name for this class...
class DocumentProfileMatchingTransformer(LightningModule):
    """Encodes profiles using pre-computed encodings. Uses nearest-neighbors
    to create 'hard negatives' to get similarity.
    """
    train_embeddings: np.ndarray      # float ndarray, shape (train_set_len, prof_emb_dim) 
                                # example: (58266, 384)
                                # -- for wiki_bio['train:10%'] and sentence-transformers/paraphrase-MiniLM-L6-v2 encoding,

    train_neighbors: np.ndarray       # int ndarray, shape (train_set_len, total_num_nearest_neighbors)
                                # example: (58266, 128) 

    num_neighbors: int          # number of neighbors to use per datapoint

    loss_fn: string             # one of ['hard_negatives', 'infonce', 'exact']

    def __init__(
        self,
        model_name_or_path: str,
        dataset_name: str,
        loss_fn: str = 'exact',
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
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=768, out_features=384), # TODO: make these numbers a feature of model/embedding type
        )
        self.temperature = torch.nn.parameter.Parameter(
            torch.tensor(5.0, dtype=torch.float32), requires_grad=True)
        
        assert loss_fn in ['hard_negatives', 'infonce', 'exact'], f'invalid loss function {loss_fn}'
        self.loss_fn = loss_fn

        # Load precomputed stuff from disk
        train_split = 'train[:10%]' # TODO: argparse for split/dataset?
        k = 128 # TODO: argparse for k (num nearest neighbors?)
        train_save_folder = os.path.join('precomputed_similarities', f'{dataset_name}__{train_split}__{k}')
        assert os.path.exists(train_save_folder), f'no precomputed similarities at folder {train_save_folder}'
        train_neighbors_path = os.path.join(train_save_folder, 'neighbors.p')
        self.train_neighbors = np.array(pickle.load(open(train_neighbors_path, 'rb')))
        train_embeddings_path = os.path.join(train_save_folder, 'embeddings.p')
        self.train_embeddings = pickle.load(open(train_embeddings_path, 'rb'))
        self.num_neighbors = 128

        val_split = 'val[:20%]'
        val_save_folder = os.path.join('precomputed_similarities', f'{dataset_name}__{val_split}__{k}')
        assert os.path.exists(val_save_folder), f'no precomputed similarities at folder {val_save_folder}'
        val_embeddings_path = os.path.join(val_save_folder, 'embeddings.p')
        self.val_embeddings = pickle.load(open(val_embeddings_path, 'rb'))

        # self.profile_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

        print(f'Initialized DocumentProfileMatchingTransformer with learning_rate = {learning_rate}')

    def forward(self, **inputs):
        return self.document_model(**inputs)

    def _compute_loss_nn(self, document_embeddings: torch.Tensor, profile_embeddings: torch.Tensor,
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
        assert document_to_profile_sim.shape == (batch_size, self.num_neighbors)
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
            (torch.matmul(profile_embeddings, document_embeddings.T) * self.temperature.exp())
        )
        diagonal_idxs = torch.arange(batch_size).to(document_embeddings.device)
        # TODO: do we want loss in one direction or both...?
        loss_d2p = torch.nn.functional.cross_entropy(
            document_to_profile_sim, diagonal_idxs
        )
        loss_p2d = 0.0
        diagonal_avg_prob = torch.diagonal(torch.nn.functional.softmax(document_to_profile_sim, dim=1)).mean()
        pct_correct = (document_to_profile_sim.argmax(1) == diagonal_idxs).to(float).mean()
        self.log(f"{metrics_key}/pct_correct", pct_correct)
        self.log(f"{metrics_key}/diagonal_avg_prob", diagonal_avg_prob)
        self.log(f"{metrics_key}/loss_d2p", loss_d2p)
        self.log(f"{metrics_key}/loss_p2d", loss_p2d)
        self.log(f"{metrics_key}/loss", loss_d2p + loss_p2d)
        return loss_d2p + loss_p2d

    
    def _compute_loss_exact(self,
            document_embeddings: torch.Tensor, profile_embeddings: torch.Tensor, document_idxs: torch.Tensor,
        metrics_key: str) -> torch.Tensor:
        """TODONOW write docstring/
         """
        assert document_embeddings.shape == profile_embeddings.shape
        assert len(document_embeddings.shape) == 2 # [batch_dim, embedding_dim]
        batch_size = len(document_embeddings)
        # Normalize embeddings before computing similarity
        document_embeddings = document_embeddings / torch.norm(document_embeddings, p=2, dim=1, keepdim=True)
        profile_embeddings = profile_embeddings / torch.norm(profile_embeddings, p=2, dim=1, keepdim=True)
        # TODONOW compute loss using profile_embeddings and document_Idxs
        # Match documents to profiles
        document_to_profile_sim = (
            (torch.matmul(profile_embeddings, document_embeddings.T) * self.temperature.exp())
        )
        diagonal_idxs = torch.arange(batch_size).to(document_embeddings.device)
        # TODO: do we want loss in one direction or both...?
        loss_d2p = torch.nn.functional.cross_entropy(
            document_to_profile_sim, diagonal_idxs
        )
        loss_p2d = 0.0
        diagonal_avg_prob = torch.diagonal(torch.nn.functional.softmax(document_to_profile_sim, dim=1)).mean()
        pct_correct = (document_to_profile_sim.argmax(1) == diagonal_idxs).to(float).mean()
        self.log(f"{metrics_key}/pct_correct", pct_correct)
        self.log(f"{metrics_key}/diagonal_avg_prob", diagonal_avg_prob)
        self.log(f"{metrics_key}/loss_d2p", loss_d2p)
        self.log(f"{metrics_key}/loss_p2d", loss_p2d)
        self.log(f"{metrics_key}/loss", loss_d2p + loss_p2d)
        return loss_d2p + loss_p2d

    def _get_nn_profile_embeddings(self, profile_idxs: torch.Tensor) -> torch.Tensor:
        """Gets profile embeddings from a list of indices.

        Args:
            profile_idxs (int torch.Tensor): indices of profiles to get embeddings for, of shape (batch)

        Returns:
            profile_embeddings (float torch.Tensor) embeddings of shape (batch, self.num_neighbors, prof_emb_dim)
        """
        assert len(profile_idxs.shape) == 1
        neighbor_idxs = self.train_neighbors[profile_idxs][:, :self.num_neighbors] # (batch, self.num_neighbors)
        profile_embeddings = torch.tensor(self.train_embeddings[neighbor_idxs]).to(self.device) # (batch, self.num_neighbors, prof_emb_dim)
        assert len(profile_embeddings.shape) == 3
        return profile_embeddings

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        document_embeddings = self.document_model(
            input_ids=batch['text1_input_ids'],
            attention_mask=batch['text1_attention_mask']
        )
        document_embeddings = document_embeddings['last_hidden_state'][:, 0, :] # (batch, document_emb_dim)
        document_embeddings = self.lower_dim_embed(document_embeddings) # (batch, document_emb_dim) -> (batch, prof_emb_dim)

        if self.loss_fn == 'hard_negatives':
            profile_embeddings = self._get_nn_profile_embeddings(batch['text_key_id'].cpu())
            loss = self._compute_loss_nn(document_embeddings, profile_embeddings, 'train')
        elif self.loss_fn == 'infonce':
            profile_embeddings = torch.tensor(self.train_embeddings[profile_idxs]).to(self.device) # (batch, self.num_neighbors, prof_emb_dim)
            assert len(profile_embeddings.shape) == 2
            loss = self._compute_loss_infonce(document_embeddings, profile_embeddings, 'train')
        elif self.loss_fn == 'exact':
            profile_embeddings = self.val_embeddings[:len(document_embeddings), :]
            profile_embeddings = torch.tensor(profile_embeddings).to(self.device)
            loss = self._compute_loss_infonce(document_embeddings)
        else:
            raise ValueError(f'Unsupported loss function {self.loss_fn}')
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        document_embeddings = self.document_model(
            input_ids=batch['text1_input_ids'],
            attention_mask=batch['text1_attention_mask']
        )
        document_embeddings = document_embeddings['last_hidden_state'][:, 0, :] # (batch, document_emb_dim)
        document_embeddings = self.lower_dim_embed(document_embeddings) # (batch, document_emb_dim) -> (batch, prof_emb_dim)
        breakpoint()
        return {
            "document_embeddings": document_embeddings,
            "??": [] # TODONOW: pass indices here
        }

    def validation_epoch_end(self, outputs) -> torch.Tensor:
        document_embeddings = torch.cat([o['document_embeddings'] for o in outputs], axis=0)
        print('validating with first', len(document_embeddings), 'val embeddings')
        profile_embeddings = self.val_embeddings[:len(document_embeddings), :] # TODONOW: don't truncate but use indices here
        profile_embeddings = torch.tensor(profile_embeddings).to(self.device)
        # self.train_embeddings[batch['text_key_id'].cpu()].shape -> (batch_size, prof_emb_dim)
        # self.train_neighbors[batch['text_key_id'].cpu()].shape -> (batch_size, k)
        loss = self._compute_loss_infonce(document_embeddings, profile_embeddings, 'val_exact')
        assert document_embeddings.shape == profile_embeddings.shape
        # Normalize embeddings before computing similarity
        document_embeddings = document_embeddings / torch.norm(document_embeddings, p=2, dim=1, keepdim=True)
        profile_embeddings = profile_embeddings / torch.norm(profile_embeddings, p=2, dim=1, keepdim=True)
        # TODO: is there a natural way to scale temperature here?
        document_to_profile_sim = (
            (torch.matmul(profile_embeddings, document_embeddings.T) * self.temperature.exp())
        )
        batch_size = len(document_embeddings)
        diagonal_idxs = torch.arange(batch_size).to(document_embeddings.device)
        loss_d2p = torch.nn.functional.cross_entropy(
            document_to_profile_sim, diagonal_idxs
        )
        loss_p2d = torch.tensor(0.0).to(document_embeddings.device)
        diagonal_avg_prob = torch.diagonal(torch.nn.functional.softmax(document_to_profile_sim, dim=1)).mean()
        # TODO: support batching here if val data is too big?
        # TODO: plot some interesting mispredictions
        # TODO: downscale losses
        self.log("val_exact/diagonal_avg_prob", diagonal_avg_prob)
        self.log("val_exact/loss_d2p", loss_d2p)
        self.log("val_exact/loss_p2d", loss_p2d)
        self.log("val_exact/loss", loss_d2p + loss_p2d)
        return loss_d2p + loss_p2d

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