import os

import pandas as pd
import datasets
import transformers

from dataloader import WikipediaDataModule
from utils import get_profile_df, tokenize_profile, words_from_text

class TestUtils:

    def test_get_profile_df(self):
        keys = ['name', 'date', 'hometown']
        values = ['jack', '2021-05-25', 'arlington, virginia']
        df = get_profile_df(keys=keys, values=values)
        sample_df = pd.DataFrame(
            data=[['jack', '2021-05-25', 'arlington, virginia']],
            columns=['name', 'date', 'hometown']
        )
        assert (df == sample_df).to_numpy().all()
    
    def test_val_df_tapas_thaila_ayala(self):
        num_cpus = len(os.sched_getaffinity(0))

        dm = WikipediaDataModule(
            document_model_name_or_path="roberta-base",
            profile_model_name_or_path="google/tapas-base",
            max_seq_length=128,
            dataset_name='wiki_bio',
            dataset_train_split='train[:1]',
            dataset_val_split='val[:1024]',
            dataset_version='1.2.0',
            word_dropout_ratio=0.0,
            word_dropout_perc=0.0,
            num_workers=1,
            train_batch_size=64,
            eval_batch_size=64
        )
        dm.setup("fit")

        ex = dm.val_dataset[10]

        prof_keys = ex["profile_keys"].split("||")
        prof_values = ex["profile_values"].split("||")
        if not len(prof_keys):
            raise ValueError("empty profile_keys")
        if not len(prof_values):
            raise ValueError("empty prof_values")
        df = get_profile_df(
            keys=prof_keys, values=prof_values
        )
        df_first_entry = dict(df.iloc[0])
        assert df_first_entry == {
            'alt': '200px',
            'name': 'thaila ayala',
            'birth_name': 'thaila ayala sales',
            'spouse': 'paulo vilhena -lrb- 2011 -- 2013 -rrb-',
            'image': 'thaila ayala 02-2 . jpg',
            'birth_place': 'presidente prudente , brazil',
            'birth_date': '14 april 1986',
            'article_title': 'thaila ayala',
            'occupation': 'actress , model'
        }
        assert set(df_first_entry.keys()) == set(prof_keys)
        assert set(df_first_entry.values()) == set(prof_values)

        tokenizer = transformers.AutoTokenizer.from_pretrained("google/tapas-base")
        encoded = tokenize_profile(
            ex=ex,
            tokenizer=tokenizer,
            max_seq_length=128
        )

        decoded = ''
        for _id in encoded['input_ids']: 
            decoded += tokenizer.decode(_id)
        assert decoded == '[CLS] who is this? [SEP] alt name birth _ name spouse image birth _ place birth _ date article _ title occupation 200px thaila ayala thaila ayala sales paulo vilhena - lrb - 2011 - - 2013 - rrb - thaila ayala 02 - 2. jpg presidente prudente, brazil 14 april 1986 thaila ayala actress, model [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

    def test_val_df_tapas_jim_bob(self):
        num_cpus = len(os.sched_getaffinity(0))

        dm = WikipediaDataModule(
            document_model_name_or_path="roberta-base",
            profile_model_name_or_path="google/tapas-base",
            max_seq_length=128,
            dataset_name='wiki_bio',
            dataset_train_split='train[:1]',
            dataset_val_split='val[:1024]',
            dataset_version='1.2.0',
            word_dropout_ratio=0.0,
            word_dropout_perc=0.0,
            num_workers=1,
            train_batch_size=64,
            eval_batch_size=64
        )
        dm.setup("fit")

        ex = dm.val_dataset[7]
        val_dataset = datasets.load_dataset(
            'wiki_bio', split='val[:1024]', version='1.2.0')

        prof_keys = ex["profile_keys"].split("||")
        prof_values = ex["profile_values"].split("||")
        if not len(prof_keys):
            raise ValueError("empty profile_keys")
        if not len(prof_values):
            raise ValueError("empty prof_values")
        df = get_profile_df(
            keys=prof_keys, values=prof_values
        )
        df_first_entry = dict(df.iloc[0])
        
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/tapas-base")
        encoded = tokenize_profile(
            ex=ex,
            tokenizer=tokenizer,
            max_seq_length=128
        )

        decoded = ''
        for _id in encoded['input_ids']: 
            decoded += tokenizer.decode(_id)
        assert decoded == "[CLS] who is this? [SEP] caption name background image label origin years _ active associated _ acts birth _ date article _ title genre jim bob performing at the garage, 2010 jim bob solo _ singer jim bob at relentless garage. jpg the rough trade fierce panda ten forty sound cherry red emi big cat london, england 1985 - - carter abdoujaparov idou chris t - t usm jim's super stereoworld james robert morrison 22 november 1960 jim bob punk rock, acoustic [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]"
    
    def test_words_from_text_empty(self):
        assert words_from_text('') == []
    
    def test_words_from_text_empty_but_punc(self):
        assert words_from_text('. /; -') == []

    def test_words_from_text_simple(self):
        s = 'sally sells seashells to make money for grad school'
        assert words_from_text(s) == s.split()
