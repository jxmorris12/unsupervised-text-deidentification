from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
import tqdm

from pytorch_lightning import LightningModule
from transformers import AdamW, AutoModel

# hide warning:
# TAPAS is a question answering model but you have not passed a query. Please be aware that the model will probably not behave correctly.
from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_error()

# TODO: make a better name for this class...
class Model(LightningModule):
    """Encodes profiles using pre-computed encodings. Uses nearest-neighbors
    to create 'hard negatives' to get similarity.
    """
    profile_embedding_dim: int
    
    adversarial_mask_k_tokens: int

    base_folder: str             # base folder for precomputed_similarities/. defaults to ''.

    learning_rate: float
    lr_scheduler_factor: float
    lr_scheduler_patience: int

    pretrained_profile_encoder: bool

    def __init__(
        self,
        document_model_name_or_path: str,
        profile_model_name_or_path: str,
        learning_rate: float = 2e-5,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 3,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        adversarial_mask_k_tokens: int = 0,
        train_batch_size: int = 32,
        pretrained_profile_encoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.document_model = AutoModel.from_pretrained(document_model_name_or_path)
        self.profile_model = AutoModel.from_pretrained(profile_model_name_or_path)

        self.profile_embedding_dim = 768
        self.document_embed = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=768, out_features=self.profile_embedding_dim),
            # TODO: make these numbers a feature of model/embedding type
        )
        self.temperature = torch.nn.parameter.Parameter(
            torch.tensor(3.5, dtype=torch.float32), requires_grad=True
        )
        
        self.adversarial_mask_k_tokens = adversarial_mask_k_tokens
        self.pretrained_profile_encoder = pretrained_profile_encoder

        print(f'Initialized DocumentProfileMatchingTransformer with learning_rate = {learning_rate}')

        self.document_learning_rate = learning_rate
        self.profile_learning_rate = learning_rate
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience

        # Important: This property activates manual optimization,
        # but we have to do loss.backward() ourselves below.
        self.automatic_optimization = False

    @property
    def document_model_device(self) -> torch.device:
        return next(self.document_model.parameters()).device

    @property
    def profile_model_device(self) -> torch.device:
        return next(self.profile_model.parameters()).device
    
    def _get_profile_for_training(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["profile"]

    def on_train_epoch_start(self):
        self.loss_function.on_train_epoch_start(epoch=self.current_epoch)
    
    def _get_inputs_from_prefix(self, batch: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        prefix += '__'
        return {k.replace(prefix, ''): v for k,v in batch.items() if k.startswith(prefix)}
    
    def forward_document_inputs(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {k: v.to(self.document_model_device) for k,v in inputs.items()}
        document_outputs = self.document_model(**inputs)                        # (batch,  sequence_length) -> (batch, sequence_length, document_emb_dim)
        document_embeddings = document_outputs['last_hidden_state'][:, 0, :]    # (batch, document_emb_dim)
        document_embeddings = self.document_embed(document_embeddings)          # (batch, document_emb_dim) -> (batch, prof_emb_dim)
        return document_embeddings
        
    def forward_document(self, batch: List[str], document_type: str, return_inputs: bool = False) -> torch.Tensor:
        """Tokenizes text and inputs to document encoder."""
        if torch.cuda.is_available(): assert self.document_model_device.type == 'cuda'
        inputs = self._get_inputs_from_prefix(batch=batch, prefix=document_type)
        doc_embeddings = self.forward_document_inputs(inputs=inputs)
        if return_inputs:
            return inputs, doc_embeddings
        else:
            return doc_embeddings

    def forward_profile(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Tokenizes text and inputs to profile encoder."""
        if torch.cuda.is_available(): assert self.profile_model_device.type == 'cuda'
        inputs = self._get_inputs_from_prefix(batch=batch, prefix='profile')
        inputs = {k: v.to(self.profile_model_device) for k,v in inputs.items()}
        output = self.profile_model(**inputs)
        return output['last_hidden_state'][:, 0, :]

    def get_optimizer(self) -> torch.optim.Optimizer:
        # TODO: make abcmethod
        raise NotImplementedError()

    def get_scheduler(self) -> torch.optim.Optimizer:
        # TODO: make abcmethod
        raise NotImplementedError()
    
    def compute_loss(self) -> Dict[str, tocrh.Tensor]:
        # TODO: make abcmethod
        raise NotImplementedError()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        dict_keys(['document', 'profile', 'document_redact_lexical', 'document_redact_ner', 'text_key_id'])
        """
        # Alternate between training phases per epoch.
        assert self.document_model.training
        assert self.document_embed.training
        assert self.profile_model.training

        optimizer = self.get_optimizer()
        results = self.compute_loss(batch=batch, batch_idx=batch_idx)
        self.log("temperature", self.temperature.exp())

        optimizer.zero_grad()
        loss = results["loss"]
        loss.backward()
        optimizer.step()

        return results
    
    def _process_validation_batch(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        document_embeddings = self.forward_document(
            batch=batch, document_type='document'
        )
        document_redact_ner_embeddings = self.forward_document(
            batch=batch, document_type='document_redact_ner'
        )
        document_redact_lexical_embeddings = self.forward_document(
            batch=batch, document_type='document_redact_lexical'
        )

        profile_embeddings = self.forward_profile(batch=batch)

        return {
            "document_embeddings": document_embeddings,
            "document_redact_ner_embeddings": document_redact_ner_embeddings,
            "document_redact_lexical_embeddings": document_redact_lexical_embeddings,
            "profile_embeddings": profile_embeddings,
            "text_key_id": batch['text_key_id']
        }

    def _process_adv_validation_batch(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        # Document embeddings (redacted document - adversarial)
        out_embeddings = {
            "adv_text_key_id": batch['text_key_id']
        }
        for k in [1, 10, 100, 1000]:
            document_adv_embeddings = self.forward_document(
                batch=batch, document_type=f'adv_document_{k}'
            )
            out_embeddings[f"adv_document_{k}"] = document_adv_embeddings
        return out_embeddings

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx: int=0) -> Dict[str, torch.Tensor]:
        assert not self.document_model.training
        assert not self.document_embed.training
        assert not self.profile_model.training

        assert dataloader_idx in [0, 1]
        with torch.no_grad():
            if dataloader_idx == 0:
                output = self._process_validation_batch(batch=batch, batch_idx=batch_idx)
            else:
                output = self._process_adv_validation_batch(batch=batch, batch_idx=batch_idx)
        return output

    def validation_epoch_end(self, output_list: List[List[Dict[str, torch.Tensor]]]) -> torch.Tensor:
        val_outputs, adv_val_outputs = output_list
        text_key_id = torch.cat(
            [o['text_key_id'] for o in val_outputs], axis=0)
        
        # Get embeddings.
        document_embeddings = torch.cat(
            [o['document_embeddings'] for o in val_outputs], axis=0)
        document_redact_ner_embeddings = torch.cat(
            [o['document_redact_ner_embeddings'] for o in val_outputs], axis=0)
        document_redact_lexical_embeddings = torch.cat(
            [o['document_redact_lexical_embeddings'] for o in val_outputs], axis=0)
        profile_embeddings = self.val_profile_embeddings

        # Compute loss on adversarial documents.
        for k in [1, 10, 100, 1000]:
            document_redact_adversarial_embeddings = torch.cat(
                [o[f"adv_document_{k}"] for o in adv_val_outputs], axis=0)
            # There may be far fewer adversarial documents than total profiles, so we only want to compare
            # for the 'text_key_id' that we have adversarial profiles for.
            adv_text_key_id = torch.cat([o['adv_text_key_id'] for o in adv_val_outputs], axis=0)
            self._compute_loss_exact(
                document_redact_adversarial_embeddings.cuda(), profile_embeddings.cuda(), adv_text_key_id.cuda(),
                metrics_key=f'val/document_redact_adversarial_{k}'
            )

        scheduler = document_scheduler

        document_embeddings = self.val_document_embeddings
        document_redact_ner_embeddings = self.val_document_redact_ner_embeddings
        document_redact_lexical_embeddings = self.val_document_redact_lexical_embeddings
        profile_embeddings = torch.cat(
            [o['profile_embeddings'] for o in val_outputs], axis=0)

        # Compute losses.
        # TODO: is `text_key_id` still correct when training profile encoder?
        #       i.e., will this still work if the validation data is shuffled?
        doc_loss = self._compute_loss_exact(
            document_embeddings.cuda(), profile_embeddings.cuda(), text_key_id.cuda(),
            metrics_key='val/document'
        )
        doc_redact_ner_loss = self._compute_loss_exact(
            document_redact_ner_embeddings.cuda(), profile_embeddings.cuda(), text_key_id.cuda(),
            metrics_key='val/document_redact_ner'
        )
        doc_redact_lexical_loss = self._compute_loss_exact(
            document_redact_lexical_embeddings.cuda(), profile_embeddings.cuda(), text_key_id.cuda(),
            metrics_key='val/document_redact_lexical'
        )
        scheduler = self.get_scheduler()
        # scheduler.step(doc_loss)
        scheduler.step(doc_redact_ner_loss)
        # scheduler.step(doc_redact_lexical_loss)
        return doc_loss

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