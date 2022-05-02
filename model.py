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
class DocumentProfileMatchingTransformer(LightningModule):
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
            # TODO: consider a nonlinearity here?
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=768, out_features=self.profile_embedding_dim),
            # TODO: make these numbers a feature of model/embedding type
        )
        self.temperature = torch.nn.parameter.Parameter(
            torch.tensor(3.5, dtype=torch.float32), requires_grad=True
        )
        
        self.adversarial_mask_k_tokens = adversarial_mask_k_tokens
        self.pretrained_profile_encoder = pretrained_profile_encoder

        print(f'Initialized DocumentProfileMatchingTransformer with learning_rate = {learning_rate}')

        self.train_document_embeddings = None
        self.train_profile_embeddings = None

        self.val_document_embeddings = None
        self.val_document_redact_ner_embeddings = None
        self.val_document_redact_lexical_embeddings = None
        self.val_profile_embeddings = None

        self.document_learning_rate = learning_rate
        self.profile_learning_rate = learning_rate
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience

        # Important: This property activates manual optimization,
        # but we have to do loss.backward() ourselves below.
        self.automatic_optimization = False

    @property
    def document_encoder_is_training(self) -> bool:
        """True if we're training the document encoder. If false, we are training the profile encoder.
        Should alternate during training epochs."""
        if self.pretrained_profile_encoder:
            return True
        else:
            return self.current_epoch % 2 == 0

    @property
    def document_model_device(self) -> torch.device:
        return next(self.document_model.parameters()).device

    @property
    def profile_model_device(self) -> torch.device:
        return next(self.profile_model.parameters()).device
    
    def _get_profile_for_training(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["profile"]

    def on_train_epoch_start(self):
        # We only want to keep one model on GPU at a time.
        # TODO: Make sure model.train() and model.eval() are properly set
        # when models are alternating
        if self.document_encoder_is_training:
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
            # self.profile_embed.cpu()
        else:
            self.train_profile_embeddings = None
            self.val_profile_embeddings = None
            # 
            self.train_document_embeddings = self.train_document_embeddings.cuda()
            self.val_document_embeddings = self.val_document_embeddings.cuda()
            self.document_model.cpu()
            self.document_embed.cpu()
            self.profile_model.cuda()
            # self.profile_embed.cuda()
        self.log("document_encoder_is_training", float(self.document_encoder_is_training))
    
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
    
    def _compute_loss_exact(
            self,
            document_embeddings: torch.Tensor,
            profile_embeddings: torch.Tensor,
            document_idxs: torch.Tensor,
            metrics_key: str
        ) -> torch.Tensor:
        """Computes classification loss from document embeddings to profile embeddings. 
        
        There are typically many more profiles than documents.

        Args:
            document_embeddings (float torch.Tensor) - document embeddings for batch, of shape (batch, emb_dim)
            profile_embeddings (float torch.Tensor) - all profile embeddings in dataset, of shape (num_profiles, emb_dim)
            document_idxs (int torch.Tensor) - integer indices of documents in profile_embeddings, of shape (batch,)

        Returns:
            loss (float torch.Tensor) - the loss, a scalar
        """
        assert len(document_embeddings.shape) == len(profile_embeddings.shape) == 2 # [batch_dim, embedding_dim]
        assert document_embeddings.shape[1] == profile_embeddings.shape[1] # embedding dims must match
        assert len(document_idxs.shape) == 1
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
        self.log(f"{metrics_key}/loss", loss)
        # Log top-k accuracies.
        for k in [1, 5, 10, 50, 100, 500, 1000]:
            if k >= batch_size: # can't compute top-k accuracy here.
                continue
            top_k_acc = (
                document_to_profile_sim.topk(k=k, dim=1)
                    .indices
                    .eq(document_idxs[:, None])
                    .any(dim=1)
                    .float()
                    .mean()
            )
            self.log(f"{metrics_key}/acc_top_k/{k}",top_k_acc)
        return loss
    
    def _training_step_document(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One step of training where training is supposed to update  `self.document_model`."""
        document_inputs, document_embeddings = self.forward_document(
            batch=batch, document_type='document', return_inputs=True
        )

        loss = self._compute_loss_exact(
            document_embeddings, self.train_profile_embeddings, batch['text_key_id'],
            metrics_key='train'
        )
        
        # if self.adversarial_mask_k_tokens:
        #     loss.backward()
        #     topk_tokens, document_inputs["input_ids"] = (
        #         self.masking_tokenizer.redact_and_tokenize_ids_from_grad(
        #             input_ids=document_inputs["input_ids"],
        #             model=self.document_model,
        #             k=self.adversarial_mask_k_tokens,
        #             mask_token_id=self.document_tokenizer.mask_token_id
        #         )
        #     )
        #     adv_document_embeddings = self.forward_document_inputs(document_inputs)
        #     adv_loss = self._compute_loss_exact(
        #         adv_document_embeddings, self.train_profile_embeddings, batch['text_key_id'],
        #         metrics_key='train'
        #     )
        #     loss = adv_loss

        self.log("temperature", self.temperature.exp())
        return {
            "loss": loss,
            "document_embeddings": document_embeddings.detach().cpu(),
            "text_key_id": batch['text_key_id'].cpu()
        }
    
    def _training_step_profile(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One step of training where training is supposed to update  `self.profile_model`."""
        profile_embeddings = self.forward_profile(batch=batch)
        self.log("temperature", self.temperature.exp())

        loss = self._compute_loss_exact(
            profile_embeddings, self.train_document_embeddings, batch['text_key_id'],
            metrics_key='train'
        )

        return {
            "loss": loss,
            "profile_embeddings": profile_embeddings.detach().cpu(),
            "text_key_id": batch['text_key_id'].cpu()
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        dict_keys(['document', 'profile', 'document_redact_lexical', 'document_redact_ner', 'text_key_id'])
        """
        # Alternate between training phases per epoch.
        assert self.document_model.training
        assert self.document_embed.training
        assert self.profile_model.training

        document_optimizer, profile_optimizer = self.optimizers()
        if self.document_encoder_is_training:
            optimizer = document_optimizer
            results = self._training_step_document(batch, batch_idx)
        else:
            optimizer = profile_optimizer
            results = self._training_step_profile(batch, batch_idx)

        optimizer.zero_grad()
        loss = results["loss"]
        loss.backward()
        optimizer.step()

        return results
        
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
        
    
    def _process_validation_batch(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        if self.document_encoder_is_training:
            # Document embeddings (original document)
            document_embeddings = self.forward_document(
                batch=batch, document_type='document'
            )

            # Document embeddings (redacted document - NER)
            document_redact_ner_embeddings = self.forward_document(
                batch=batch, document_type='document_redact_ner'
            )

            # Document embeddings (redacted document - lexical overlap)
            document_redact_lexical_embeddings = self.forward_document(
                batch=batch, document_type='document_redact_lexical'
            )

            return {
                "document_embeddings": document_embeddings,
                "document_redact_ner_embeddings": document_redact_ner_embeddings,
                "document_redact_lexical_embeddings": document_redact_lexical_embeddings,
                "text_key_id": batch['text_key_id']
            }
        else:
            # Profile embeddings
            profile_embeddings = self.forward_profile(batch=batch)
            return {
                "profile_embeddings": profile_embeddings,
                "text_key_id": batch['text_key_id']
            }

    def _process_adv_validation_batch(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        if self.document_encoder_is_training:
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
        else:
            # Don't recompute this loss if the document encoder isn't training.
            return {}

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
        document_scheduler, profile_scheduler = self.lr_schedulers()
        # Get embeddings.
        if self.document_encoder_is_training:
            document_embeddings = torch.cat(
                [o['document_embeddings'] for o in val_outputs], axis=0)
            document_redact_ner_embeddings = torch.cat(
                [o['document_redact_ner_embeddings'] for o in val_outputs], axis=0)
            document_redact_lexical_embeddings = torch.cat(
                [o['document_redact_lexical_embeddings'] for o in val_outputs], axis=0)
            profile_embeddings = self.val_profile_embeddings

            self.val_document_embeddings = document_embeddings
            self.val_document_redact_ner_embeddings = document_redact_ner_embeddings
            self.val_document_redact_lexical_embeddings = document_redact_lexical_embeddings

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

        else:
            document_embeddings = self.val_document_embeddings
            document_redact_ner_embeddings = self.val_document_redact_ner_embeddings
            document_redact_lexical_embeddings = self.val_document_redact_lexical_embeddings
            profile_embeddings = torch.cat(
                [o['profile_embeddings'] for o in val_outputs], axis=0)
            
            self.val_profile_embeddings = profile_embeddings

            scheduler = profile_scheduler
        
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
        # scheduler.step(doc_loss)
        scheduler.step(doc_redact_ner_loss)
        # scheduler.step(doc_redact_lexical_loss)
        return doc_loss
    
    def precompute_profile_embeddings(self):
        self.profile_model.cuda()
        self.profile_model.eval()
        # self.profile_embed.cuda()
        print('Precomputing profile embeddings before first epoch...')
        self.train_profile_embeddings = np.zeros((len(self.trainer.datamodule.train_dataset), self.profile_embedding_dim))
        for train_batch in tqdm.tqdm(self.trainer.datamodule.train_dataloader(), desc="[1/2] Precomputing train embeddings", colour="magenta", leave=False):
            with torch.no_grad():
                profile_embeddings = self.forward_profile(batch=train_batch)
            self.train_profile_embeddings[train_batch["text_key_id"]] = profile_embeddings.cpu()
        self.train_profile_embeddings = torch.tensor(self.train_profile_embeddings, dtype=torch.float32)

        self.val_profile_embeddings = np.zeros((len(self.trainer.datamodule.val_dataset), self.profile_embedding_dim))

        val_dataloader, adv_val_dataloader = self.trainer.datamodule.val_dataloader()
        for val_batch in tqdm.tqdm(val_dataloader, desc="[2/2] Precomputing val embeddings", colour="green", leave=False):
            with torch.no_grad():
                profile_embeddings = self.forward_profile(batch=val_batch)
                # TODO: remove name for TAPAS?
            self.val_profile_embeddings[val_batch["text_key_id"]] = profile_embeddings.cpu()
        self.val_profile_embeddings = torch.tensor(self.val_profile_embeddings, dtype=torch.float32)
        self.profile_model.train()

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
        # Precompute embeddings
        self.precompute_profile_embeddings()

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
