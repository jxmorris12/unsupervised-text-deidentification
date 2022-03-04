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

    num_neighbors: int          # number of neighbors to use per datapoint (only for 'hard_negatives' loss)

    loss_fn: str                # one of ['hard_negatives', 'infonce', 'exact']

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
        num_neighbors: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.document_model = AutoModel.from_pretrained(model_name_or_path)
        self.lower_dim_embed = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=768, out_features=384),
            # 769 * 384 = 295296
            # TODO: different options for this, or less dropout?
            # TODO: make these numbers a feature of model/embedding type
        )
        self.temperature = torch.nn.parameter.Parameter(
            torch.tensor(5.0, dtype=torch.float32), requires_grad=True)
        
        assert loss_fn in ['hard_negatives', 'infonce', 'exact'], f'invalid loss function {loss_fn}'
        self.loss_fn = loss_fn

        # Load precomputed stuff from disk
        train_split = 'train[:10%]' # TODO: argparse for split/dataset?
        self.num_neighbors = num_neighbors
        train_save_folder = os.path.join('precomputed_similarities', f'{dataset_name}__{train_split}__{self.num_neighbors}')
        assert os.path.exists(train_save_folder), f'no precomputed similarities at folder {train_save_folder}'
        train_neighbors_path = os.path.join(train_save_folder, 'neighbors.p')
        self.train_neighbors = np.array(pickle.load(open(train_neighbors_path, 'rb')))
        train_embeddings_path = os.path.join(train_save_folder, 'embeddings.p')
        self.train_embeddings = pickle.load(open(train_embeddings_path, 'rb'))

        val_split = 'val[:20%]'
        val_save_folder = os.path.join('precomputed_similarities', f'{dataset_name}__{val_split}__{self.num_neighbors}')
        assert os.path.exists(val_save_folder), f'no precomputed similarities at folder {val_save_folder}'
        val_embeddings_path = os.path.join(val_save_folder, 'embeddings.p')
        self.val_embeddings = pickle.load(open(val_embeddings_path, 'rb'))
        print(f'Initialized DocumentProfileMatchingTransformer with learning_rate = {learning_rate}')
        
        # TODO make these things show up in W&B.
        self.hparams["train_split"] = train_split
        self.hparams["val_split"] = val_split
        self.hparams["len_train_embeddings"] = len(self.train_embeddings)
        self.hparams["len_val_embeddings"] = len(self.val_embeddings)
        self.hparams["num_neighbors"] = self.num_neighbors

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
        loss = torch.nn.functional.cross_entropy(document_to_profile_sim, true_idxs)
        true_avg_prob = torch.nn.functional.softmax(document_to_profile_sim, dim=1)[:, 0].mean()
        self.log(f"{metrics_key}/true_avg_prob", true_avg_prob)
        pct_correct = (document_to_profile_sim.argmax(1) == 0).to(float).mean()
        self.log(f"{metrics_key}/pct_correct", pct_correct)
        self.log(f"{metrics_key}/loss", loss)
        return loss
    
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
        diagonal_avg_prob = torch.diagonal(torch.nn.functional.softmax(document_to_profile_sim, dim=1)).mean()
        pct_correct = (document_to_profile_sim.argmax(1) == diagonal_idxs).to(float).mean()
        self.log(f"{metrics_key}/pct_correct", pct_correct)
        self.log(f"{metrics_key}/diagonal_avg_prob", diagonal_avg_prob)
        self.log(f"{metrics_key}/loss", loss)
        return loss

    
    def _compute_loss_exact(self,
            document_embeddings: torch.Tensor, profile_embeddings: torch.Tensor, document_idxs: torch.Tensor,
        metrics_key: str) -> torch.Tensor:
        """Computes classification loss from document embeddings to profile embeddings. 
        
        There are typically many more profiles than documents.

        Args:
            document_embeddings (float torch.Tensor) - document embeddings for batch, of shape (batch, emb_dim)
            profile_embeddings (float torch.Tensor) - all profile embeddings in dataset, of shape (num_profiles, emb_dim)
            document_idxs (int torch.Tensor) - integer indices of documents in profile_embeddings, of shape (batch,)

        Returns:
            loss (float torch.Tensor) - the loss, a scalar
        """
        # print('document_embeddings.shape:', document_embeddings.shape, '//', 'profile_embeddings.shape:', profile_embeddings.shape, '//', 'document_idxs.shape', document_idxs.shape)
        assert len(document_embeddings.shape) == len(profile_embeddings.shape) == 2 # [batch_dim, embedding_dim]
        assert document_embeddings.shape[1] == profile_embeddings.shape[1] # embedding dims must match
        assert len(document_idxs.shape) == 1
        assert document_embeddings.shape[0] == document_embeddings.shape[0] # batch dims must match
        batch_size = len(document_embeddings)
        # Normalize embeddings before computing similarity
        document_embeddings = document_embeddings / torch.norm(document_embeddings, p=2, dim=1, keepdim=True)
        profile_embeddings = profile_embeddings / torch.norm(profile_embeddings, p=2, dim=1, keepdim=True)
        # Match documents to profiles
        document_to_profile_sim = (
            (torch.matmul(document_embeddings, profile_embeddings.T) * self.temperature.exp())
        )
        loss = torch.nn.functional.cross_entropy(
            document_to_profile_sim, document_idxs
        )
        diagonal_avg_prob = torch.diagonal(torch.nn.functional.softmax(document_to_profile_sim, dim=1)).mean()
        pct_correct = (document_to_profile_sim.argmax(1) == document_idxs).to(float).mean()
        self.log(f"{metrics_key}/pct_correct", pct_correct)
        self.log(f"{metrics_key}/diagonal_avg_prob", diagonal_avg_prob)
        self.log(f"{metrics_key}/loss", loss)
        return loss

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
            profile_embeddings = torch.tensor(
                self.train_embeddings[batch['text_key_id'].cpu()]
            ).to(self.device) # (batch, prof_emb_dim)
            assert len(profile_embeddings.shape) == 2
            loss = self._compute_loss_infonce(document_embeddings, profile_embeddings, 'train')
        elif self.loss_fn == 'exact':
            profile_embeddings = torch.tensor(self.train_embeddings).to(self.device)
            loss = self._compute_loss_exact(document_embeddings, profile_embeddings, batch['text_key_id'], metrics_key='train')
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
        return {
            "document_embeddings": document_embeddings,
            "text_key_id": batch['text_key_id']
        }

    def validation_epoch_end(self, outputs) -> torch.Tensor:
        document_embeddings = torch.cat([o['document_embeddings'] for o in outputs], axis=0)
        text_key_id = torch.cat([o['text_key_id'] for o in outputs], axis=0)
        profile_embeddings = torch.tensor(self.val_embeddings).to(self.device)
        loss = self._compute_loss_exact(document_embeddings, profile_embeddings, text_key_id.to(self.device), metrics_key='val_exact')
        return loss

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