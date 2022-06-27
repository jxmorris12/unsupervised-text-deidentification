import datasets
import os
import pytest
import transformers
import tqdm

from torch.utils.data import DataLoader

from masking_tokenizing_dataset import MaskingTokenizingDataset
from utils import create_document_and_profile_from_wikibio

num_cpus = len(os.sched_getaffinity(0))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TestMaskingTokenizingDataset:

    def test_create_document_and_profile(self):
        train_dataset = datasets.load_dataset('wiki_bio', split='train[:1024]', version='1.2.0')
        ex = create_document_and_profile_from_wikibio(train_dataset[0])
        assert len(ex["profile_keys"].strip())
        assert len(ex["profile_values"].strip())
        assert len(ex["document"].strip())
        assert len(ex["profile"].strip())
        assert ex["profile_keys"].split("||") == ['name', 'nationality', 'birth_date', 'article_title', 'occupation']

    def test_create_document_and_profile_redacted(self):
        train_dataset = datasets.load_dataset('wiki_bio', split='train[:1024]', version='1.2.0')
        ex = create_document_and_profile_from_wikibio(train_dataset[0], redact_profile=True)
        assert len(ex["profile_keys"].strip())
        assert len(ex["profile_values"].strip())
        assert len(ex["document"].strip())
        assert len(ex["profile"].strip())
        # name, birth_date, article_title should be redacte.d
        assert ex["profile_keys"].split("||") == ['nationality', 'occupation']

    def test_train_data(self):
        split = "train[:1%]"
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

        for epoch in tqdm.trange(3):
            for idx in tqdm.trange(256):
                ex = train_tokenizing_dataset[idx]

    def test_train_data_uniform_idf(self):
        split = "train[:100%]"
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
            word_dropout_ratio=1.0,
            word_dropout_perc=1.0,
            profile_row_dropout_perc=0.1,
            sample_spans=False,
            adversarial_masking=False,
            idf_masking=True,
            num_nearest_neighbors=0,
            document_types=["document"],
            is_train_dataset=True
        )
        
        for ex in tqdm.tqdm(train_tokenizing_dataset, desc='iterating training data'):
            pass
    

if __name__ == '__main__':
    # for profiling

    d = TestMaskingTokenizingDataset()
    d.test_train_data_uniform_idf()
