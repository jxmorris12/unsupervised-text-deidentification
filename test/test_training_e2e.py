from typing import Tuple

import pytest

import torch

from pytorch_lightning import Trainer, seed_everything

from dataloader import WikipediaDataModule
from model import ContrastiveModel, CoordinateAscentModel

class TestEnd2End:
    @pytest.fixture
    def setup(self):
        seed_everything(88)
    
    def _get_dm_and_model(self, loss_fn: str, document_model: str, profile_model: str, max_seq_length: 32) -> Tuple[WikipediaDataModule, CoordinateAscentModel]:
        dm = WikipediaDataModule(
            document_model_name_or_path=document_model,
            profile_model_name_or_path=profile_model,
            max_seq_length=max_seq_length,
            dataset_name='wiki_bio',
            dataset_train_split='train[:50]',
            dataset_val_split='val[:50]',
            dataset_version='1.2.0',
            word_dropout_ratio=0.2,
            word_dropout_perc=0.3,
            sample_spans=True,
            train_batch_size=16,
            eval_batch_size=3,
            num_workers=4,
        )
        dm.setup("fit")
        
        model_cls_dict = {
            'coordinate_ascent': CoordinateAscentModel,
            'contrastive': ContrastiveModel,
        }
        model = model_cls_dict[loss_fn](
            document_model_name_or_path=document_model,
            profile_model_name_or_path=profile_model,
            train_batch_size=16,
            learning_rate=1e-6,
            pretrained_profile_encoder=False,
            lr_scheduler_factor=0.5,
            lr_scheduler_patience=1000,
            adversarial_mask_k_tokens=0,
            num_workers=4,
        )
        return (dm, model)

    def _run_e2e_test(
            self,
            tmpdir: str,
            loss_fn: str,
            document_model: str,
            profile_model: str, 
            max_seq_length: int = 32,
            precision: int = 32,
        ):
        dm, model = self._get_dm_and_model(
            loss_fn=loss_fn,
            document_model=document_model,
            profile_model=profile_model,
            max_seq_length=max_seq_length
        )
        trainer = Trainer(
            default_root_dir=tmpdir,
            enable_checkpointing=False,
            val_check_interval=0.5,
            callbacks=[],
            max_epochs=2,
            precision=precision,
            log_every_n_steps=min(len(dm.train_dataloader()), 50),
            limit_train_batches=1.0, # change this to make training faster (1.0 = full train set)
            limit_val_batches=1.0,
            gpus=torch.cuda.device_count(),
            logger=[],
        )
        trainer.fit(model, dm)

    @pytest.mark.parametrize("loss_fn,precision", [("coordinate_ascent", 32), ("contrastive", 32)]) #, ("contrastive", 16)
    def test_end_to_end_distilbert(self, tmpdir: str, loss_fn: str, precision: int):
        self._run_e2e_test(
            tmpdir,
            loss_fn,
            'distilbert-base-uncased',
            'distilbert-base-uncased',
            max_seq_length=32,
            precision=precision
        )

    @pytest.mark.parametrize("loss_fn", ["coordinate_ascent", "contrastive"])
    def test_e2e_tapas(self, tmpdir: str, loss_fn: str):
        self._run_e2e_test(tmpdir, loss_fn, 'distilbert-base-uncased',  'google/tapas-base', 32)
    
    @pytest.mark.parametrize("loss_fn", ["coordinate_ascent", "contrastive"])
    def test_inference_roberta_distilbert(self, loss_fn):
        dm, model = self._get_dm_and_model(
            loss_fn=loss_fn,
            document_model='roberta-base',
            profile_model='distilbert-base-uncased',
            max_seq_length=16
        )
        train_batch = next(iter(dm.train_dataloader()))
        val_batch = next(iter(dm.val_dataloader()[0]))
        if torch.cuda.is_available(): 
            model.document_model.cuda()
            model.document_embed.cuda()
        train_doc_outputs = model.forward_document(batch=train_batch, document_type='document')
        val_doc_outputs = model.forward_document(batch=val_batch, document_type='document')
        assert not (train_doc_outputs.isnan().any())
        assert not (val_doc_outputs.isnan().any())