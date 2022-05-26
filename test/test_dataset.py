import pytest

import torch

from dataloader import WikipediaDataModule


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
            num_workers=0,
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
            dataset_train_split='train[:10%]',
            dataset_val_split='val[:64]',
            dataset_version='1.2.0',
            num_workers=0,
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

