from typing import Dict, List, Optional

import os
import pickle
import random
import re

import numpy as np
import torch

from pytorch_lightning import LightningModule
from sentence_transformers import SentenceTransformer
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer

from utils import words_from_text


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

    max_seq_length: int

    word_dropout_ratio: float    # Percentage of the time to do word dropout
    word_dropout_perc: float     # Percentage of words to replace with mask token
    word_dropout_mask_token: str # mask token

    profile_model_name_or_path: str
    profile_encoder_name: str    # like ['tapas', 'st-paraphrase']
    redaction_strategy: str      # one of ['', 'spacy_ner', 'lexical']

    base_folder: str             # base folder for precomputed_similarities/. defaults to ''.

    learning_rate: float
    lr_scheduler_factor: float
    lr_scheduler_patience: int

    def __init__(
        self,
        document_model_name_or_path: str,
        profile_model_name_or_path: str,
        dataset_name: str,
        learning_rate: float = 2e-5,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 3,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        max_seq_length: int = 128,
        word_dropout_ratio: float = 0.0,
        word_dropout_perc: float = 0.0,
        redaction_strategy = "",
        base_folder = "",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.document_model = AutoModel.from_pretrained(document_model_name_or_path)
        self.document_tokenizer = AutoTokenizer.from_pretrained(document_model_name_or_path, use_fast=True)
        self.profile_model = AutoModel.from_pretrained(profile_model_name_or_path)
        self.profile_tokenizer = AutoTokenizer.from_pretrained(profile_model_name_or_path, use_fast=True)

        profile_emb_dim = 768 # TODO: set dynamically based on model
        self.document_embed = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=768, out_features=profile_emb_dim),
            # (769 + 1) * 384 = 295,680 parameters
            # TODO: different options for this, or less dropout?
            # TODO: make these numbers a feature of model/embedding type
        )
        self.temperature = torch.nn.parameter.Parameter(
            torch.tensor(5.0, dtype=torch.float32), requires_grad=True
        )
        
        assert redaction_strategy in ["", "spacy_ner", "lexical"]
        self.redaction_strategy = redaction_strategy
        self.max_seq_length = max_seq_length

        print(f'Initialized DocumentProfileMatchingTransformer with learning_rate = {learning_rate}')

        self.word_dropout_ratio = word_dropout_ratio
        self.word_dropout_perc = word_dropout_perc
        self.word_dropout_mask_token = self.document_tokenizer.mask_token

        if self.word_dropout_ratio:
            print('[*] Word dropout hyperparameters:', 
                'ratio:', self.word_dropout_ratio, '/',
                'percentage:', self.word_dropout_perc, '/',
                'token:', self.word_dropout_mask_token
            )

        self.train_document_embeddings = None
        self.train_profile_embeddings = None

        self.val_document_embeddings = None
        self.val_document_redact_ner_embeddings = None
        self.val_document_redact_lexical_embeddings = None
        self.val_profile_embeddings = None
        
        # TODO make these things show up in W&B?
        self.learning_rate = learning_rate
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.hparams["train_split"] = train_split
        self.hparams["val_split"] = val_split
        self.hparams["num_neighbors"] = self.num_neighbors

    @property
    def document_encoder_is_training(self) -> bool:
        """True if we're training the document encoder. If false, we are training the profile encoder.
        Should alternate during training epochs."""
        return self.current_epoch % 2 == 0

    def on_train_epoch_start(self):
        # We only want to keep one model on GPU at a time.
        if self.document_encoder_is_training:
            self.document_model.cuda()
            self.train_profile_embeddings.cuda()
            self.val_profile_embeddings.cuda()
            self.profile_model.cpu()
        else:
            self.document_model.cpu()
            self.train_document_embeddings.cuda()
            self.val_document_embeddings.cuda()
            self.profile_model.cuda()

    def word_dropout_text(self, text: List[str]) -> List[str]:
        """Apply word dropout to list of text inputs."""
        # TODO: implement this in dataloader to take advantage of multiprocessing
        for i in range(len(text)):
            if random.random() > self.word_dropout_ratio:
                # Don't do dropout this % of the time
                continue
            for w in words_from_text(text[i]):
                if random.random() < self.word_dropout_perc:
                    text[i] = re.sub(
                        (r'\b{}\b').format(w),
                        self.word_dropout_mask_token, text[i], 1
                    )
        return text
        
    def forward_document_text(self, text: List[str]) -> torch.Tensor:
        """Tokenizes text and inputs to document encoder."""
        if self.training:
            text = self.word_dropout_text(text)
        inputs = self.document_tokenizer.batch_encode_plus(
            text,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        document_embeddings = self.document_model(**inputs)                     # (batch,  sequence_length) -> (batch, document_emb_dim)
        document_embeddings = document_embeddings['last_hidden_state'][:, 0, :] # (batch, document_emb_dim)
        document_embeddings = self.document_embed(document_embeddings)         # (batch, document_emb_dim) -> (batch, prof_emb_dim)
        return document_embeddings
    
    def forward_profile_text(self, text: List[str]) -> torch.Tensor:
        """Tokenizes text and inputs to profile encoder."""
        # if self.training:
        #     text = self.word_dropout_text(text)
        inputs = self.profile_tokenizer.batch_encode_plus(
            text,
            # TODO: permit a different max seq length for profile?
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        profile_embeddings = self.profile_model(**inputs)                     # (batch,  sequence_length) -> (batch, document_emb_dim)
        return profile_embeddings
    
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
    
    def _training_step_document(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One step of training where training is supposed to update  `self.document_model`."""
        if self.redaction_strategy == "":
            document_embeddings = self.forward_document_text(
                text=batch['document'],
            )
        elif self.redaction_strategy == "spacy_ner":
            document_embeddings = self.forward_document_text(
                text=batch['document_redact_ner'],
            )
        elif self.redaction_strategy == "lexical":
            document_embeddings = self.forward_document_text(
                text=batch['document_redact_lexical'],
            )
        else:
            raise Exception(f"unknown redaction strategy {self.redaction_strategy}")
        self.log("temperature", self.temperature.exp())

        return {
            "loss": self._compute_loss_exact(document_embeddings, self.train_profile_embeddings, batch['text_key_id'], metrics_key='train')
            "document_embeddings": document_embeddings.detach().cpu(),
            "text_key_id": batch['text_key_id'].cpu()
        }
    
    def _training_step_profile(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One step of training where training is supposed to update  `self.profile_model`."""
        profile_embeddings = self.forward_profile_text(
            text=batch['profile'],
        )
        self.log("temperature", self.temperature.exp())

        return {
            "loss": self._compute_loss_exact(self.train_document_embeddings, profile_embeddings, batch['text_key_id'], metrics_key='train'),
            "profile_embeddings": profile_embeddings.detach().cpu(),
            "text_key_id": batch['text_key_id'].cpu()
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        dict_keys(['document', 'profile', 'document_redact_lexical', 'document_redact_ner', 'text_key_id'])
        """
        # Alternate between training phases per epoch.
        if self.document_encoder_is_training:
            return self._training_step_document(batch, batch_idx)
        else:
            return self._training_step_profile(batch, batch_idx)
        
    def training_epoch_end(self, training_step_outputs: Dict):
        if self.document_encoder_is_training:
            # TODO: fix this as it assumes profile and doc embeddings are the same shape.
            self.train_document_embeddings = np.zeros_like(train_profile_embeddings)
            for document_embeddings, text_key_id in zip(training_step_outputs["document_embeddings"], training_step_outputs["text_key_id"]):
                self.train_document_embeddings[text_key_id] = document_embeddings
            self.train_document_embeddings = torch.tensor(self.document_profile_embeddings)
            self.train_profile_embeddings = None
        else:
            # TODO: fix this as it assumes profile and doc embeddings are the same shape.
            self.train_profile_embeddings = np.zeros_like(document_profile_embeddings)
            for profile_embeddings, text_key_id in zip(training_step_outputs["profile_embeddings"], training_step_outputs["text_key_id"]):
                self.train_profile_embeddings[text_key_id] = profile_embeddings
            self.train_profile_embeddings = torch.tensor(self.train_profile_embeddings)
            self.train_document_embeddings = None

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict[str, torch.Tensor]:
        if self.document_encoder_training:
            # Document embeddings (original document)
            document_embeddings = self.forward_document_text(
                text=batch['document'],
            )

            # Document embeddings (redacted document - NER)
            document_redact_ner_embeddings = self.forward_document_text(
                text=batch['document_redact_ner']
            )

            # Document embeddings (redacted document - lexical overlap)
            document_redact_lexical_embeddings = self.forward_document_text(
                text=batch['document_redact_lexical'],
            )
            return {
                "document_embeddings": document_embeddings,
                "document_redact_ner_embeddings": document_redact_ner_embeddings,
                "document_redact_lexical_embeddings": document_redact_lexical_embeddings,
                "text_key_id": batch['text_key_id']
            }
        else:
            # Profile embeddings
            profile_embeddings = self.forward_profile_text(
                text=batch['profile']
            )
            return {
                "profile_embeddings": profile_embeddings,
                "text_key_id": batch['text_key_id']
            }

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        text_key_id = torch.cat(
            [o['text_key_id'] for o in outputs], axis=0)
        
        # Get embeddings.
        if self.document_encoder_training:
            document_embeddings = torch.cat(
                [o['document_embeddings'] for o in outputs], axis=0)
            document_redact_ner_embeddings = torch.cat(
                [o['document_redact_ner_embeddings'] for o in outputs], axis=0)
            document_redact_lexical_embeddings = torch.cat(
                [o['document_redact_lexical_embeddings'] for o in outputs], axis=0)
            profile_embeddings = self.val_profile_embeddings.to(self.device)

            self.val_profile_embeddings = None
        else:
            document_embeddings = self.val_document_embeddings.to(device)
            document_redact_ner_embeddings = self.val_document_redact_ner_embeddings.to(device)
            document_redact_lexical_embeddings = self.val_document_redact_lexical_embeddings.to(device)
            profile_embeddings = torch.cat(
                [o['profile_embeddings'] for o in outputs], axis=0)
            
            self.val_document_embeddings = None
            self.val_document_redact_ner_embeddings = None
            self.val_document_redact_lexical_embeddings = None
        
        # Compute losses.
        # TODO: is `text_key_id` still correct when training profile encoder?
        #       i.e., will this still work if the validation data is shuffled?
        doc_loss = self._compute_loss_exact(
            document_embeddings, profile_embeddings, text_key_id.to(self.device),
            metrics_key='val_exact/document'
        )
        doc_redact_ner_loss = self._compute_loss_exact(
            document_redact_ner_embeddings, profile_embeddings, text_key_id.to(self.device),
            metrics_key='val_exact/document_redact_ner'
        )
        doc_redact_lexical_loss = self._compute_loss_exact(
            document_redact_lexical_embeddings, profile_embeddings, text_key_id.to(self.device),
            metrics_key='val_exact/document_redact_lexical'
        )
        return doc_loss
    
    def _precompute_initial_embeddings(self):
        self.profile_model.cuda()
        self.train_profile_embeddings = np.zeros(document_profile_embeddings)
        for train_batch in self.trainer.datamodule.train_dataloader():
            with torch.no_grad():
                profile_embeddings = self.forward_profile_text(text=train_batch["profile"])
            self.train_profile_embeddings[train_batch["text_key_id"]] = profile_embeddings
        self.train_profile_embeddings = torch.tensor(self.train_profile_embeddings)

        self.val_profile_embeddings = np.zeros(??)
        for val_batch in self.trainer.datamodule.val_dataloader():
            with torch.no_grad():
                profile_embeddings = self.forward_profile_text(text=val_batch["profile"])
            self.val_profile_embeddings[train_batch["text_key_id"]] = profile_embeddings
        self.val_profile_embeddings = torch.tensor(self.val_profile_embeddings)
            

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
        self._precompute_initial_embeddings()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(
            self.document_model.parameters(), lr=self.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=1e-8
        )
        # see source:
        # pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html?highlight=optimizer_step#optimizer-step
        scheduler = {
            "scheduler": scheduler,
            "monitor": "val_exact/document/loss",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]