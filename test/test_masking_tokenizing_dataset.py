import datasets
import pytest

import transformers

from masking_tokenizing_dataset import MaskingTokenizingDataset
from utils import create_document_and_profile_from_wikibio

class TestMaskingTokenizingDataset:

    def test_create_document_and_profile(self):
        train_dataset = datasets.load_dataset('wiki_bio', split='train[:1024]', version='1.2.0')
        ex = create_document_and_profile_from_wikibio(train_dataset[0])
        assert len(ex["profile_keys"].strip())
        assert len(ex["profile_values"].strip())
        assert len(ex["document"].strip())
        assert len(ex["profile"].strip())

    def test_train_data(self):
        split = "train[:10%]"
        datasets.utils.logging.set_verbosity_debug()
        train_dataset = datasets.load_dataset(
            "wiki_bio", split=split, version="1.2.0"
        )
        train_dataset = train_dataset.map(
            create_document_and_profile_from_wikibio
        )
        train_dataset = train_dataset.add_column(
            "text_key_id", 
            list(range(len(train_dataset)))
        )
        train_dataset.cleanup_cache_files()

        document_tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base')
        profile_tokenizer = transformers.AutoTokenizer.from_pretrained('google/tapas-base')
        train_tokenizing_dataset = MaskingTokenizingDataset(
            train_dataset,
            document_tokenizer=document_tokenizer,
            profile_tokenizer=profile_tokenizer,
            max_seq_length=32,
            word_dropout_ratio=0.2,
            word_dropout_perc=0.2,
            profile_row_dropout_perc=0.5,
            sample_spans=True,
            num_nearest_neighbors=0,
            document_types=["document"],
            is_train_dataset=True
        )
        import tqdm
        for epoch in tqdm.trange(3):
            for idx in tqdm.trange(256):
                ex = train_tokenizing_dataset[idx]