import pytest

from dataloader import WikipediaDataModule


class TestWikiDataset:
    def test_dataset(self):
        dm = WikipediaDataModule(
            document_model_name_or_path='distilbert-base-uncased',
            profile_model_name_or_path='distilbert-base-uncased',
            max_seq_length=32,
            mask_token='<mask>',
            dataset_name='wiki_bio',
            dataset_train_split='train[:64]',
            dataset_val_split='val[:64]',
            dataset_version='1.2.0',
            num_workers=0,
            train_batch_size=16,
            eval_batch_size=8,
        )
        dm.setup("fit")
        train_dataloader = dm.train_dataloader()
        train_batch = next(iter(train_dataloader))
        
        val_dataloader, adv_val_dataloader = dm.val_dataloader()

        val_batch = next(iter(val_dataloader))
        adv_val_batch = next(iter(adv_val_dataloader))