import pytest

import torch

from pytorch_lightning import Trainer, seed_everything

from dataloader import WikipediaDataModule
from model import ContrastiveModel, CoordinateAscentModel

class TestContrastiveLoss:
    @pytest.fixture
    def setup(self):
        seed_everything(88)
    
    def test_loss_with_adv_masking(self, tmpdir):
        document_model = 'distilbert-base-uncased'
        profile_model = 'distilbert-base-uncased'
        max_seq_length = 32
        dm = WikipediaDataModule(
            document_model_name_or_path=document_model,
            profile_model_name_or_path=profile_model,
            max_seq_length=max_seq_length,
            dataset_name='wiki_bio',
            dataset_train_split='train[:10%]',
            dataset_val_split='val[:1%]',
            dataset_version='1.2.0',
            word_dropout_ratio=0.0,
            word_dropout_perc=0.0,
            sample_spans=True,
            adversarial_masking=True,
            train_batch_size=10,
            eval_batch_size=3,
            num_nearest_neighbors=0, # no nearest-neighbors
        )
        dm.setup("fit")
        
        model = ContrastiveModel(
            document_model_name_or_path=document_model,
            profile_model_name_or_path=profile_model,
            train_batch_size=10,
            learning_rate=1e-6,
            pretrained_profile_encoder=False,
            lr_scheduler_factor=0.5,
            lr_scheduler_patience=1000,
            adversarial_mask_k_tokens=0,
        )
        trainer = Trainer(
            default_root_dir=tmpdir,
            enable_checkpointing=False,
            val_check_interval=1,
            callbacks=[],
            max_epochs=5,
            log_every_n_steps=min(len(dm.train_dataloader()), 50),
            limit_val_batches=0.0,
            gpus=torch.cuda.device_count(),
            logger=[],
        )
        trainer.fit(model, dm)
    
    def test_loss_with_nearest_neighbors(self, tmpdir):
        document_model = 'distilbert-base-uncased'
        profile_model = 'distilbert-base-uncased'
        max_seq_length = 24
        dm = WikipediaDataModule(
            document_model_name_or_path=document_model,
            profile_model_name_or_path=profile_model,
            max_seq_length=max_seq_length,
            dataset_name='wiki_bio',
            dataset_train_split='train[:10%]',
            dataset_val_split='val[:1%]',
            dataset_version='1.2.0',
            word_dropout_ratio=0.2,
            word_dropout_perc=0.3,
            sample_spans=True,
            train_batch_size=16,
            eval_batch_size=3,
            num_nearest_neighbors=4,
        )
        dm.setup("fit")
        
        model = ContrastiveModel(
            document_model_name_or_path=document_model,
            profile_model_name_or_path=profile_model,
            train_batch_size=16,
            learning_rate=1e-6,
            pretrained_profile_encoder=False,
            lr_scheduler_factor=0.5,
            lr_scheduler_patience=1000,
            adversarial_mask_k_tokens=0,
        )
        trainer = Trainer(
            default_root_dir=tmpdir,
            enable_checkpointing=False,
            val_check_interval=0.5,
            callbacks=[],
            max_epochs=2,
            log_every_n_steps=min(len(dm.train_dataloader()), 50),
            limit_train_batches=4,
            limit_val_batches=4,
            gpus=torch.cuda.device_count(),
            logger=[],
        )
        trainer.fit(model, dm)