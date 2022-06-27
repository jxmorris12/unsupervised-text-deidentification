import os
import pytest

import torch

from datamodule import WikipediaDataModule


num_cpus = len(os.sched_getaffinity(0))

class TestWikiDataset:
    def test_dataset(self):
        eval_batch_size = 9
        max_seq_length = 32
        dm = WikipediaDataModule(
            document_model_name_or_path='distilbert-base-uncased',
            profile_model_name_or_path='distilbert-base-uncased',
            max_seq_length=max_seq_length,
            mask_token='<mask>',
            dataset_name='wiki_bio',
            dataset_train_split='train[:64]',
            dataset_val_split='val[:64]',
            dataset_version='1.2.0',
            num_workers=num_cpus,
            train_batch_size=16,
            eval_batch_size=eval_batch_size,
        )
        dm.setup("fit")
        train_dataloader = dm.train_dataloader()
        train_batch = next(iter(train_dataloader))
        
        val_dataloader, adv_val_dataloader = dm.val_dataloader()

        val_batch = next(iter(val_dataloader))
        adv_val_batch = next(iter(adv_val_dataloader))

        # Check pretokenized profiles.
        assert 'profile__input_ids' in val_batch
        assert isinstance(val_batch['profile__input_ids'], torch.Tensor)
        assert val_batch['profile__input_ids'].shape == (eval_batch_size, max_seq_length)

        assert 'profile__attention_mask' in val_batch
        assert isinstance(val_batch['profile__attention_mask'], torch.Tensor)
        assert val_batch['profile__attention_mask'].shape == (eval_batch_size, max_seq_length)

        # Check redacted stuff.
        assert "document__input_ids" in val_batch.keys()
        assert "document__word_ids" in val_batch.keys()
        assert "document_redact_ner__input_ids" in val_batch.keys()
        assert "document_redact_lexical__input_ids" in val_batch.keys()

        # sanity-check number of redacted tokens for bm25.
        mask_id = dm.document_tokenizer.mask_token_id
        assert "document_redact_idf_20__input_ids" in val_batch.keys()
        assert (
            (val_batch["document_redact_idf_20__input_ids"] == mask_id).sum()
            <
            (val_batch["document_redact_idf_40__input_ids"] == mask_id).sum()
        )
        assert (
            (val_batch["document_redact_idf_40__input_ids"] == mask_id).sum()
            <
            (val_batch["document_redact_idf_60__input_ids"] == mask_id).sum()
        )
        assert (
            (val_batch["document_redact_idf_60__input_ids"] == mask_id).sum()
            <
            (val_batch["document_redact_idf_80__input_ids"] == mask_id).sum()
        )
    
    def test_dataset_nearest_neighbors(self):
        train_batch_size = 16
        num_nearest_neighbors = 2
        max_seq_length = 32
        dm = WikipediaDataModule(
            document_model_name_or_path='distilbert-base-uncased',
            profile_model_name_or_path='distilbert-base-uncased',
            max_seq_length=max_seq_length,
            mask_token='<mask>',
            dataset_name='wiki_bio',
            dataset_train_split='train[:1%]',
            dataset_val_split='val[:256]',
            dataset_version='1.2.0',
            num_workers=num_cpus,
            num_nearest_neighbors=num_nearest_neighbors,
            train_batch_size=train_batch_size,
            eval_batch_size=8,
        )
        dm.setup("fit")
        # Check nearest-neighbors got loaded into dataset.
        assert "nearest_neighbor_idxs" in dm.train_dataset[0].keys()
        # Check nearest-neighbors are tokenized and returned in dataloader.
        train_dataloader = dm.train_dataloader()
        train_batch = next(iter(train_dataloader))
        assert "profile_neighbor__input_ids" in train_batch
        assert (
            train_batch["profile_neighbor__input_ids"].shape
            ==
            (train_batch_size, num_nearest_neighbors, max_seq_length)
        )   

