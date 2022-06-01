import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')
    
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

    def test_train_data(self):
        split = "train[:1%]"
        datasets.utils.logging.set_verbosity_debug()
        train_dataset = datasets.load_dataset(
            "wiki_bio", split=split, version="1.2.0"
        )
        train_dataset = train_dataset.map(
            create_document_and_profile_from_wikibio,
            n_proc=num_cpus
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

"""
Profiler output
         927590291 function calls (926402870 primitive calls) in 342.589 seconds

   Ordered by: internal time
   List reduced from 8337 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  1422249   13.489    0.000   35.460    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:2128(_clean_text)
  3194682   12.575    0.000   32.334    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:2069(_run_split_on_punc)
 17102266   12.465    0.000   12.465    0.000 {method 'translate' of 'str' objects}
  4270215   12.061    0.000   33.553    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/tokenization_utils_base.py:1233(all_special_tokens_extended)
  4158795   11.622    0.000   17.305    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:2151(tokenize)
  1423978    9.586    0.000  242.473    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/tokenization_utils.py:481(tokenize)
  3553936    9.416    0.000   23.153    0.000 {method 'sub' of 're.Pattern' objects}
 17306343    9.414    0.000   14.012    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/tokenization_utils.py:292(_is_punctuation)
 19082486    9.053    0.000   13.686    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/tokenization_utils.py:514(<lambda>)
  1423978    8.796    0.000   12.331    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/tokenization_utils.py:91(split)
    40320    8.466    0.000  253.373    0.006 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:1315(_tokenize_table)
39885710/39879843    8.027    0.000    8.071    0.000 {built-in method builtins.getattr}
  4270215    7.926    0.000   15.378    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/tokenization_utils_base.py:1207(special_tokens_map_extended)
 19076928    7.839    0.000   11.864    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/tokenization_utils.py:280(_is_control)
  1422249    7.750    0.000   14.188    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:2091(_tokenize_chinese_chars)
"""