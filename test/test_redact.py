import os
import pytest

from dataloader import WikipediaDataModule

from redact import (
    remove_named_entities_spacy, remove_named_entities_spacy_batch,
    remove_named_entities_bert_batch,
    remove_overlapping_words
)

@pytest.fixture
def val_dataset():
    num_cpus = len(os.sched_getaffinity(0))

    dm = WikipediaDataModule(
        document_model_name_or_path="roberta-base",
        profile_model_name_or_path="google/tapas-base",
        max_seq_length=128,
        dataset_name='wiki_bio',
        dataset_train_split='train[:1]',
        dataset_val_split='val[:36]',
        dataset_version='1.2.0',
        word_dropout_ratio=0.0,
        word_dropout_perc=0.0,
        num_workers=1,
        train_batch_size=64,
        eval_batch_size=64
    )
    dm.setup("fit")
    return dm.val_dataset

class TestRedact:

    def test_redact_ner(self, val_dataset):
        doc = val_dataset[0]['document']
        redacted_doc = remove_named_entities_spacy(doc)

        # Won't get *all* named entities since data is pre-lowercased.
        assert redacted_doc == "pope [MASK] [MASK] of alexandria ( also known as khail iii ) was the coptic pope of alexandria and patriarch of the see of st. mark ( [MASK] -- [MASK] ) .\nin [MASK] , the governor of [MASK] , [MASK] [MASK] tulun , forced khail to pay heavy contributions , forcing him to sell a church and some attached properties to the local jewish community .\nthis building was at one time believed to have later become the site of the [MASK] geniza .\n"

    def test_redact_ner_batch(self):
        docs = [
            'Jack went to Pike\'s Place Market',
            'Julius Caesar crossed the Rubicon'
        ]
        redacted_docs = remove_named_entities_spacy_batch(docs)
        # Works better with case restored
        assert redacted_docs == [
            '[MASK] went to [MASK][MASK] [MASK] [MASK]',
            '[MASK] [MASK] crossed the [MASK]'
        ]
    
    def test_redact_ner_bert(self):
        docs = [
            'Jack went to Pike\'s Place Market',
            'Julius Caesar crossed the Rubicon'
        ]
        docs = [doc.lower() for doc in docs]
        redacted_docs = remove_named_entities_bert_batch(docs)
        print(redacted_docs)
        assert redacted_docs == [
            '[MASK] went to [MASK]\'[MASK] [MASK] [MASK]',
            '[MASK] [MASK] crossed [MASK] [MASK]'
        ]
    
    def test_redact_ner_bert_val_dataset(self, val_dataset):
        take_20_words = lambda s: ' '.join(s.split(' ')[:20])
        docs = [
            take_20_words(val_dataset[i]['document']) for i in range(10)
        ]
        redacted_docs = remove_named_entities_bert_batch(docs)
        assert redacted_docs == [
            'pope [MASK] [MASK] of [MASK] ( also known as [MASK] [MASK] ) was the [MASK] pope of [MASK] and patriarch',
            '[MASK] [MASK] is a male former table tennis player from [MASK] .\n',
            '[MASK] [MASK] ( born 30 november 1977 ) is a [MASK] professional footballer .\nhe currently plays as a striker for',
            '[MASK] [MASK] , ( born march 14 , 1996 ) is a professional squash player who represents [MASK] .\nshe reached',
            '[MASK] [MASK]. [MASK] is a former [MASK] member of the [MASK] [MASK] [MASK] [MASK] .\nhe was born in [MASK] to',
            '[MASK] [MASK] ( born may 8 , 1935 ) is an [MASK] stage , film and television actress .\nshe is',
            "[MASK] [MASK] [MASK] ( born august 5 , 1981 ) , nicknamed `` [MASK] [MASK] [MASK] '' , is an",
            '[MASK] [MASK] ( born [MASK] [MASK] [MASK] on 22 november 1960 ) is a [MASK] musician and author , best',
            '[MASK] [MASK] ( born november 20 , 1972 in [MASK] , [MASK] ) is a former professional [MASK] football defensive',
            'blessed [MASK] [MASK] [MASK] t.[MASK].[MASK].[MASK]. ( ) was a [MASK] visionary and anchoress from [MASK] ( [MASK] ) .\nshe was'
        ]

    
    def test_redact_lexical(self, val_dataset):
        doc = val_dataset[0]['document']
        prof = val_dataset[0]['profile']
        redacted_doc = remove_overlapping_words(doc, prof)

        # Won't get *all* named entities since data is pre-lowercased.
        assert redacted_doc == "[MASK] [MASK] [MASK] [MASK] [MASK] ( also known as khail [MASK] ) was [MASK] [MASK] [MASK] [MASK] [MASK] and [MASK] [MASK] [MASK] [MASK] [MASK] st. [MASK] ( [MASK] -- [MASK] ) .\n[MASK] 882 , [MASK] governor [MASK] [MASK] , ahmad ibn tulun , forced khail to pay heavy contributions , forcing him to sell a [MASK] and some attached properties to [MASK] local jewish community .\nthis building was at one time believed to have later become [MASK] site [MASK] [MASK] cairo geniza .\n"