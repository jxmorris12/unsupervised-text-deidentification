from typing import Any, Dict, List, Tuple

import abc

import numpy as np
import torch
import transformers

from pytorch_lightning import LightningModule
from transformers import AutoModel


class Model(LightningModule, abc.ABC):
    """Encodes profiles using pre-computed encodings. Uses nearest-neighbors
    to create 'hard negatives' to get similarity.
    """
    profile_embedding_dim: int

    base_folder: str             # base folder for precomputed_similarities/. defaults to ''.

    learning_rate: float
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    grad_norm_clip: float

    pretrained_profile_encoder: bool

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
            torch.nn.Linear(in_features=768, out_features=self.profile_embedding_dim, dtype=torch.float32),
        )
        self.temperature = torch.nn.parameter.Parameter(
            torch.tensor(1.0, dtype=torch.float32), requires_grad=True
        )
        
        self.pretrained_profile_encoder = pretrained_profile_encoder

        self.document_learning_rate = learning_rate
        self.profile_learning_rate = learning_rate
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience

        print(f'Initialized model with learning_rate = {learning_rate} and patience {self.lr_scheduler_patience}')
        
        self.grad_norm_clip = grad_norm_clip

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
        return

    def _log_adv_masking_table(self):
        # If we're doing adversarial masking, log a table to W&B.
        from main import USE_WANDB
        if not USE_WANDB:
            return
        
        import wandb
        if not wandb.run:
            return

        train_dataset = self.trainer.train_dataloader.loaders.dataset

        if train_dataset.adversarial_masking:
            rows = []
            for idx in range(32):
                doc_name = train_dataset.dataset[idx]["name"]
                doc_text = train_dataset.dataset[idx]["document"]
                masked_words = list(train_dataset.adv_word_mask_map[idx])
                num_words_to_mask_next = train_dataset.adv_word_mask_num[idx]
                rows.append((doc_name, doc_text, masked_words, num_words_to_mask_next))
            my_table = wandb.Table(
                columns=["name", "document", "masked words", "num_words_to_mask_next"],
                data=rows
            )
            wandb.run.log({"adversarial_mask_table": my_table})

    def on_train_epoch_end(self):
        self._log_adv_masking_table()

    
    def _get_inputs_from_prefix(self, batch: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        prefix += '__'
        return {k.replace(prefix, ''): v for k,v in batch.items() if k.startswith(prefix)}

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
        # Commented-out because the DPR paper ("Dense Passage Retrieval for Open-Domain Question Answering") reports
        # better results with dot-product than cosine similarity.
        # document_embeddings = document_embeddings / torch.norm(document_embeddings, p=2, dim=1, keepdim=True)
        # profile_embeddings = profile_embeddings / torch.norm(profile_embeddings, p=2, dim=1, keepdim=True)
        # Match documents to profiles
        document_to_profile_sim = torch.matmul(document_embeddings, profile_embeddings.T)
        document_to_profile_sim *= self.temperature.exp()
        loss = torch.nn.functional.cross_entropy(
            document_to_profile_sim, document_idxs
        )
        self.log(f"{metrics_key}/loss", loss)

        # Also track a boolean mask for which things were correct.
        is_correct = (document_to_profile_sim.argmax(dim=1) == document_idxs)

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
            self.log(f"{metrics_key}/acc_top_k/{k}", top_k_acc)
        return is_correct, loss
    
    def forward_document_inputs(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # We track the word IDs for documents in case we need them to do
        # subword pooling or anything like that. But the document doesn't take
        # word_ids as input to the forward pass.
        if "word_ids" in inputs: del inputs["word_ids"]

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

    def forward_profile(self, batch: Dict[str, torch.Tensor], profile_key: str = 'profile', collapse_axis: bool = False) -> torch.Tensor:
        """Tokenizes text and inputs to profile encoder."""
        if torch.cuda.is_available(): assert self.profile_model_device.type == 'cuda'
        inputs = self._get_inputs_from_prefix(batch=batch, prefix=profile_key)
        if collapse_axis: # Collapse shape (b, n, s) -> (b * n, s)
            inputs = {
                k: v.flatten(end_dim=1)
                for k,v in inputs.items()
            }
        inputs = {k: v.to(self.profile_model_device) for k,v in inputs.items()}
        output = self.profile_model(**inputs)
        return output['last_hidden_state'][:, 0, :]

    @abc.abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def compute_loss(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Alternate between training phases per epoch.
        assert self.document_model.training
        assert self.document_embed.training
        assert self.profile_model.training

        optimizer = self.get_optimizer()
        optimizer.zero_grad()
        results = self.compute_loss(batch=batch, batch_idx=batch_idx)
        self.log("temperature", self.temperature.exp())

        self.manual_backward(results["loss"])
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm_clip)
        optimizer.step()

        # Log number of masks in training inputs.
        mask_token = self.trainer.train_dataloader.loaders.dataset.document_tokenizer.mask_token_id
        avg_num_masks = (batch['document__input_ids'] == mask_token).float().sum(dim=1).mean()
        self.log("num_masks", avg_num_masks)

        # Use document model embedding gradient for adversarial masking (if enabled).
        emb_grad = self.document_model.embeddings.word_embeddings.weight.grad
        if not (emb_grad is None) and (emb_grad.sum() > 0):
            # We call process_grad() on the train dataset because the
            # train dataset may use this gradient to perform masking.
            self.trainer.train_dataloader.loaders.dataset.process_grad(
                input_ids=batch['document__input_ids'],
                word_ids=batch['document__word_ids'],
                emb_grad=emb_grad.norm(p=2, dim=1),
                is_correct=results["is_correct"],
                text_key_id=batch['text_key_id']
            )
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
    
    def on_validation_start(self):
        self.profile_model.cuda()
        self.document_model.cuda()
        self.document_embed.cuda()

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
        profile_embeddings = torch.cat(
            [o['profile_embeddings'] for o in val_outputs], axis=0)

        # Compute loss on adversarial documents.
        for k in [1, 10, 100, 1000]:
            document_redact_adversarial_embeddings = torch.cat(
                [o[f"adv_document_{k}"] for o in adv_val_outputs], axis=0)
            # There may be far fewer adversarial documents than total profiles, so we only want to compare
            # for the 'text_key_id' that we have adversarial profiles for.
            adv_text_key_id = torch.cat([o['adv_text_key_id'] for o in adv_val_outputs], axis=0)
            if adv_text_key_id.max().item() > len(profile_embeddings):
                # If the validation set is too small, this will throw an error bc
                # the corresponding profiles for the adversarially-masked documents
                # aren't in the val set. In this case, just skip to the next run of
                # the loop.
                continue
            self._compute_loss_exact(
                document_redact_adversarial_embeddings.cuda(), profile_embeddings.cuda(), adv_text_key_id.cuda(),
                metrics_key=f'val/document_redact_adversarial_{k}'
            )

        # Compute losses on regular documents.
        _, doc_loss = self._compute_loss_exact(
            document_embeddings.cuda(), profile_embeddings.cuda(), text_key_id.cuda(),
            metrics_key='val/document'
        )
        _, doc_redact_ner_loss = self._compute_loss_exact(
            document_redact_ner_embeddings.cuda(), profile_embeddings.cuda(), text_key_id.cuda(),
            metrics_key='val/document_redact_ner'
        )
        _, doc_redact_lexical_loss = self._compute_loss_exact(
            document_redact_lexical_embeddings.cuda(), profile_embeddings.cuda(), text_key_id.cuda(),
            metrics_key='val/document_redact_lexical'
        )

        # If there are multiple LR schedulers, call step() on all of them.
        lr_schedulers = self.lr_schedulers()
        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]

        for scheduler in lr_schedulers:
            # scheduler.step(doc_loss)
            # scheduler.step(doc_redact_ner_loss)
            # scheduler.step(doc_redact_lexical_loss)
            scheduler.step(
                self.trainer.logged_metrics.get('val/document_redact_adversarial_1/loss', 100.0)
            )
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