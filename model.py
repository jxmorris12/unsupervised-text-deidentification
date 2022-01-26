from typing import Optional

import torch

from pytorch_lightning import LightningModule
from transformers import AdamW, AutoConfig, AutoModel


class DocumentProfileMatchingTransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # TODO(jxm): use AutoModel here just to get vectors..?
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.temperature = torch.nn.parameter.Parameter(
            torch.tensor(5.0, dtype=torch.float32), requires_grad=True)
        # self.metric = datasets.load_metric(
        #     "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        # )
        print(f'Initialized DocumentProfileMatchingTransformer with learning_rate = {learning_rate}')

    def forward(self, **inputs):
        return self.model(**inputs)

    def _compute_loss(self, profile_embeddings: torch.Tensor, document_embeddings: torch.Tensor) -> torch.Tensor:
        """ TODO(jxm): document/explain """
        assert profile_embeddings.shape == document_embeddings.shape
        assert len(profile_embeddings.shape) == 2 # [batch_dim, embedding_dim]
        batch_size = len(profile_embeddings)
        # Normalize embeddings before computing similarity
        profile_embeddings = profile_embeddings / torch.norm(profile_embeddings, p=2, dim=1, keepdim=True)
        document_embeddings = document_embeddings / torch.norm(document_embeddings, p=2, dim=1, keepdim=True)
        # Match documents to profiles
        document_to_profile_sim = torch.nn.functional.softmax(
            (torch.matmul(document_embeddings, profile_embeddings.T) * self.temperature.exp()), dim=-1
        )
        diagonal_idxs = torch.arange(batch_size).to(profile_embeddings.device)
        return torch.nn.functional.cross_entropy(
            document_to_profile_sim, diagonal_idxs
        )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # TODO(jxm): should we use two different models for these encodings?
        profile_embeddings = self.model(
            input_ids=batch['text1_input_ids'],
            attention_mask=batch['text1_attention_mask']
        )
        profile_embeddings = profile_embeddings['last_hidden_state'][:, 0, :]
        # Just take last hidden state at index 0 which should be CLS. TODO(jxm): is this right?

        document_embeddings = self.model(
            input_ids=batch['text2_input_ids'],
            attention_mask=batch['text2_attention_mask']
        )
        document_embeddings = document_embeddings['last_hidden_state'][:, 0, :]

        # profile_embeddings = self({ 
        #     'input_ids':        batch['text1_input_ids'],
        #     'attention_mask':   batch['text1_attention_mask']
        # })
        # document_embeddings = self({ 
        #     'input_ids':        batch['text2_input_ids'],
        #     'attention_mask':   batch['text2_attention_mask']
        # })
        return self._compute_loss(profile_embeddings, document_embeddings)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        profile_embeddings = self.model(batch['text1_input_ids'])
        document_embeddings = self.model(batch['text2_input_ids'])
        # TODO(jxm): return predictions or labels?
        return {
            "loss": self._compute_loss(profile_embeddings, document_embeddings)
        }

    def validation_epoch_end(self, outputs) -> torch.Tensor:
        # TODO(jxm): compute metrics properly
        # preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        # labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        # self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
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
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return optimizer
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_steps,
        #     num_training_steps=self.total_steps,
        # )
        # scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # return [optimizer], [scheduler]