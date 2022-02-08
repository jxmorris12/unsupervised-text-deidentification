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

    def _compute_loss(self, profile_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
        metrics_key: str) -> torch.Tensor:
        """InfoNCE matching loss between `profile_embeddings` and `document_embeddings`. Computes
        metrics and prints, prefixing with `metrics_key`.
         """
        assert profile_embeddings.shape == document_embeddings.shape
        assert len(profile_embeddings.shape) == 2 # [batch_dim, embedding_dim]
        batch_size = len(profile_embeddings)
        # Normalize embeddings before computing similarity
        profile_embeddings = profile_embeddings / torch.norm(profile_embeddings, p=2, dim=1, keepdim=True)
        document_embeddings = document_embeddings / torch.norm(document_embeddings, p=2, dim=1, keepdim=True)
        # Match documents to profiles
        document_to_profile_sim = (
            (torch.matmul(document_embeddings, profile_embeddings.T) * self.temperature.exp())
        )
        diagonal_idxs = torch.arange(batch_size).to(profile_embeddings.device)
        loss_d2p = torch.nn.functional.cross_entropy(
            document_to_profile_sim, diagonal_idxs
        )
        loss_p2d = torch.nn.functional.cross_entropy(
            document_to_profile_sim.T, diagonal_idxs
        )
        diagonal_avg_prob = torch.diagonal(torch.nn.functional.softmax(document_to_profile_sim, dim=-1)).mean()
        self.log(f"{metrics_key}/diagonal_avg_prob", diagonal_avg_prob)
        self.log(f"{metrics_key}/loss_d2p", loss_d2p)
        self.log(f"{metrics_key}/loss_p2d", loss_p2d)
        self.log(f"{metrics_key}/loss", loss_d2p + loss_p2d)
        return loss_d2p + loss_p2d

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # TODO(jxm): should we use two different models for these encodings?
        profile_embeddings = self.model(
            input_ids=batch['text1_input_ids'],
            attention_mask=batch['text1_attention_mask']
        )
        profile_embeddings = profile_embeddings['last_hidden_state'][:, 0, :]
        # profile_embeddings = profile_embeddings['last_hidden_state'].mean(dim=1)
        # Just take last hidden state at index 0 which should be CLS. TODO(jxm): is this right?

        document_embeddings = self.model(
            input_ids=batch['text2_input_ids'],
            attention_mask=batch['text2_attention_mask']
        )
        document_embeddings = document_embeddings['last_hidden_state'][:, 0, :]
        # document_embeddings = document_embeddings['last_hidden_state'].mean(dim=1)

        # document_embeddings = profile_embeddings # TMP
        loss = self._compute_loss(profile_embeddings, document_embeddings, 'train')
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        profile_embeddings = self.model(
            input_ids=batch['text1_input_ids'],
            attention_mask=batch['text1_attention_mask']
        )
        document_embeddings = self.model(
            input_ids=batch['text2_input_ids'],
            attention_mask=batch['text2_attention_mask']
        )
        # TODO(jxm): return predictions or labels?
        return {
            "loss": self._compute_loss(profile_embeddings, document_embeddings, 'val')
        }

    def validation_epoch_end(self, outputs) -> torch.Tensor:
        # TODO(jxm): compute metrics properly
        # preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        # labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        # self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        loss = torch.stack([x["loss"] for x in outputs]).mean()
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