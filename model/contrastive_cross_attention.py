from typing import Any, Dict, List

import abc

import numpy as np
import torch
import tqdm

from transformers import AutoModel

from .model import Model

class ContrastiveCrossAttentionModel(Model):

    def __init__(
        self,
        document_model_name_or_path: str,
        profile_model_name_or_path: str,
        learning_rate: float = 2e-5,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 3,
        grad_norm_clip: float = 5.0,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        train_batch_size: int = 32,
        shared_embedding_dim: int = 768,
        warmup_epochs: float = 0.2,
        label_smoothing: float = 0.0,
        pretrained_profile_encoder: bool = False,
        **kwargs,
    ):
        super(Model, self).__init__() # this syntax calls super().super()
        self.save_hyperparameters()
        self.document_model_name_or_path = document_model_name_or_path
        self.document_model = AutoModel.from_pretrained(document_model_name_or_path)

        self.bottleneck_embedding_dim = 768

        print("Initializing contrastive cross-encoder as document model (not creating profile model)")
        assert document_model_name_or_path == profile_model_name_or_path
        # Experiments show that the extra layer isn't helpful for either the document embedding or the profile embedding.
        # But it's useful to have a profile embedding (we didn't use to have one at all).
        self.document_embed = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.01),
            # torch.nn.Linear(in_features=self.bottleneck_embedding_dim, out_features=self.bottleneck_embedding_dim, dtype=torch.float32),
            # torch.nn.ReLU(),
            torch.nn.Dropout(p=0.01),
            torch.nn.Linear(in_features=self.bottleneck_embedding_dim, out_features=1, dtype=torch.float32),
        )
        self.label_smoothing = label_smoothing
        
        self.document_learning_rate = learning_rate
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self._optim_steps = {}
        self.warmup_epochs = warmup_epochs

        print(f'Initialized ContrastiveCrossAttention model with learning_rate = {learning_rate} and patience {self.lr_scheduler_patience}')
        
        self.grad_norm_clip = grad_norm_clip

        # Important: This property activates manual optimization,
        # but we have to do loss.backward() ourselves below.
        self.automatic_optimization = False

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizers()

    def forward_document_and_profile_inputs(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Inputs both document and profile to joint encoder.
        
        Args:
            batch: Dict[str, torch.Tensor] containing 'document__input_ids' and 'profile__input_ids',
                all of shape (batch_size, max_seq_length/2)

        Returns:
            float32 torch.Tensor of shape (batch_size,) with scores for each thing.
        """
        document_profile_outputs = self.document_model(**inputs)
        # Mean-pooling over sequence length to produce embedding.
        document_profile_cross_embeddings = document_profile_outputs['last_hidden_state'].mean(dim=1)
        # Then map output to a score.
        return self.document_embed(document_profile_cross_embeddings)
    
    def _compute_cross_encoder_loss(
            self, score_matrix: torch.Tensor, metrics_key: str
        ) -> torch.Tensor:
        """Computes loss for cross-encoder. Cross-encoder outputs a score for every (document, profile) pair.
        We softmax over them and compute the loss. The pairs along the first column are the correct ones, which
        is why the true idxs are zeros.
        
        TODO: docstring."""
        batch_size = len(score_matrix)
        zero_idxs = torch.zeros(batch_size, dtype=torch.long).to(score_matrix.device)
        loss = torch.nn.functional.cross_entropy(
            score_matrix, zero_idxs,
            label_smoothing=self.label_smoothing
        )
        self.log(f"{metrics_key}/loss", loss, sync_dist=True)

        # Also track a boolean mask for which things were correct.
        is_correct = (score_matrix.argmax(dim=1) == zero_idxs)

        # Log top-k accuracies.
        for k in [1, 5, 10, 50, 100, 500, 1000]:
            if (k >= score_matrix.shape[1]): # can't compute top-k accuracy here.
                continue
            top_k_acc = (
                score_matrix.topk(k=k, dim=1)
                    .indices
                    .eq(zero_idxs[:, None])
                    .any(dim=1)
                    .float()
                    .mean()
            )
            self.log(f"{metrics_key}/acc_top_k/{k}", top_k_acc, sync_dist=True)
        return is_correct, loss
    
    def compute_loss(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int,
            document_type: str = 'document',
            metrics_key: str = 'train',
        ) -> Dict[str, torch.Tensor]:
        batch_size, sequence_length = batch[f'{document_type}__input_ids'].shape

        # If keys that start with 'profile_neighbor' are present in the batch,
        # include those in the loss.
        assert "profile_neighbor__input_ids" in batch.keys(), f"profile neighbors not found in batch (keys: {batch.keys()})"
        prof_neighbor_inputs = self._get_inputs_from_prefix(batch=batch, prefix='profile_neighbor')

        assert len(prof_neighbor_inputs['input_ids'].shape) == 3, "profile neighbors should have inputs of shape (bs, nn, sl)"
        assert prof_neighbor_inputs['input_ids'].shape[0] == batch_size
        assert prof_neighbor_inputs['input_ids'].shape[2] == sequence_length

        num_neighbors = prof_neighbor_inputs['input_ids'].shape[1]
        assert num_neighbors > 0

        prof_inputs = self._get_inputs_from_prefix(batch=batch, prefix='profile')
        all_prof_inputs = {
            key: torch.cat((prof_inputs[key][:, None, :], prof_neighbor_inputs[key]), dim=1)
            for key in prof_neighbor_inputs.keys()
        }
        assert all_prof_inputs['input_ids'].shape == (batch_size, 1 + num_neighbors, sequence_length)

        # TODO: Change last token in profile to be SEP instead of END?? or is it already SEP?

        doc_inputs = self._get_inputs_from_prefix(batch=batch, prefix=document_type)
        if 'word_ids' in doc_inputs: del doc_inputs['word_ids']
        assert doc_inputs['input_ids'].shape == (batch_size, sequence_length)

        # Repeat document IDs along num_neighbors dimension.
        all_doc_inputs = {
            key: doc_inputs[key][:, None, :].repeat((1, 1 + num_neighbors, 1))
            for key in doc_inputs.keys()
        }
        assert all_doc_inputs['input_ids'].shape == all_prof_inputs['input_ids'].shape

        assert all_doc_inputs.keys() == all_prof_inputs.keys()

        # Concatenate everything along *sequence_length* dimension.
        all_inputs = {
            key: torch.cat((all_doc_inputs[key], all_prof_inputs[key]), dim=2)
            for key in doc_inputs
        }
        assert all_inputs['input_ids'].shape == (batch_size, 1 + num_neighbors, sequence_length * 2)

        # Input into model
        all_inputs = {
            key: all_inputs[key].view((batch_size * (1 + num_neighbors), sequence_length * 2))
            for key in all_inputs
        }
        outputs = self.forward_document_and_profile_inputs(inputs=all_inputs)
        assert outputs.shape == (batch_size * (1 + num_neighbors), 1)
        outputs = outputs.view((batch_size, 1 + num_neighbors))

        # Compute loss
        is_correct, loss = self._compute_cross_encoder_loss(
            score_matrix=outputs, metrics_key=metrics_key
        )

        return {
            "loss": loss,
            "is_correct": is_correct.cpu(),
            "text_key_id": batch['text_key_id'].cpu()
        }
    
    def assert_models_are_training(self):
        assert self.document_model.training
        assert self.document_embed.training
    
    def on_validation_start(self):
        # 
        # self.document_model.cuda()
        # self.document_embed.cuda()
        # 
        pass
    
    def _process_validation_batch(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        for document_type in ['document', 'document_redact_ner', 'document_redact_lexical']:
            self.compute_loss(
                batch=batch, batch_idx=batch_idx, document_type=document_type, metrics_key=f'val/{document_type}'
            )
        
        idf_total_loss = 0.0
        for idf_n in [20, 40, 60, 80]:
            results = self.compute_loss(
                batch=batch, batch_idx=batch_idx, document_type=f'document_redact_idf_{idf_n}', metrics_key=f'val/document_redact_idf_{idf_n}'
            )
            idf_total_loss += results["loss"].item()
        self.log(f"val/document_redact_idf_total/loss", idf_total_loss, sync_dist=True)
        

    def _process_adv_validation_batch(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        return
        # TODO: process adversarial batches. Need profiles for it though.
        for k in [1, 10, 100, 1000]:
            self.compute_loss(
                batch=batch, batch_idx=batch_idx, document_type=f'adv_document_{k}', metrics_key=f'val/document_redact_adversarial_{k}'
            )

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx: int=0) -> Dict[str, torch.Tensor]:
        assert not self.document_model.training
        assert not self.document_embed.training

        assert dataloader_idx in [0, 1]
        with torch.no_grad():
            if dataloader_idx == 0:
                output = self._process_validation_batch(batch=batch, batch_idx=batch_idx)
            else:
                output = self._process_adv_validation_batch(batch=batch, batch_idx=batch_idx)
        return output

    def validation_epoch_end(self, output_list: List[List[Dict[str, torch.Tensor]]]) -> torch.Tensor:
        pass

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = torch.optim.AdamW(
            (
                list(self.document_model.parameters()) + 
                list(self.document_embed.parameters())
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

