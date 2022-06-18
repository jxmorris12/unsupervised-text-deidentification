from typing import Any, Dict, List

import os
import pickle
import re

import spacy
import torch

from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

from utils import words_from_text


nlp = spacy.load("en_core_web_sm")

eng_stopwords = stopwords.words('english')

num_cpus = os.cpu_count()

def remove_named_entities_spacy(x: str, mask_token: str = "[MASK]") -> str:
    """
    Replaces named entities in `x` with `mask_token`.
    
    From spacy.io/usage/rule-based-matching/#regex-text:
        nsubj: Nominal subject.
        prep: Preposition.
        pobj: Object of preposition.
        NNP: Proper noun, singular.
        VBD: Verb, past tense.
        IN: Conjunction, subordinating or preposition.
    """
    doc = nlp(x)
    new_tokens = [t.text_with_ws if not t.ent_type_ else (mask_token + t.whitespace_) for t in doc]
    return "".join(new_tokens)

def remove_named_entities_spacy_batch(x_list: List[str], mask_token: str = "[MASK]") -> List[str]:
    """
    Replaces named entities in each `x` from `x_list` with `mask_token`.
        Utilizes batching from spacy library via `nlp.pipe()`.
    """
    docs = nlp.pipe(x_list, n_process=num_cpus)
    new_tokens_list = [
        [t.text_with_ws if not t.ent_type_ else (mask_token + t.whitespace_) for t in doc]
        for doc in docs
    ]
    return ["".join(new_tokens) for new_tokens in new_tokens_list]


bert_ner_pipeline = None
def remove_named_entities_bert_batch(x_list: List[str], mask_token: str = "[MASK]") -> List[str]:
    """
    Replaces named entities in each `x` from `x_list` with `mask_token`.
        Utilizes BERT-based NER model from HuggingFace: https://huggingface.co/dslim/bert-base-NER-uncased
    
        Example entities:
            {'end': 4,
            'entity': 'B-PER',
            'index': 1,
            'score': 0.96453863,
            'start': 0,
            'word': 'jack'}
            {'end': 17,
            'entity': 'B-LOC',
            'index': 4,
            'score': 0.7856458,
            'start': 13,
            'word': 'pike'}
            {'end': 18,
            'entity': 'I-LOC',
            'index': 5,
            'score': 0.9546801,
            'start': 17,
            'word': "'"}
            {'end': 19,
            'entity': 'I-LOC',
            'index': 6,
            'score': 0.9678726,
            'start': 18,
            'word': 's'}
    """
    global bert_ner_pipeline
    
    if bert_ner_pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER-uncased")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER-uncased")
        bert_ner_pipeline = pipeline(
            "ner", model=model, tokenizer=tokenizer, device=(0 if torch.cuda.is_available() else -1)
        )
    
    entities = bert_ner_pipeline(x_list)
    redacted_docs = []
    for i in range(len(x_list)):
        doc = x_list[i]
        for entity in entities[i][::-1]:
            word_to_mask = entity['word'].replace('##', '')
            if not word_to_mask.isalnum():
                continue
            entity_start_idx = entity['start']
            entity_end_idx = entity['end']
            doc = doc[:entity_start_idx] + mask_token + doc[entity_end_idx:]
        
        # collapse subwords to single-word, i.e. <mask><mask> for 'rubicon' becomes just <mask> here.
        while (mask_token + mask_token) in doc:
            doc = doc.replace(mask_token + mask_token, mask_token)
        redacted_docs.append(doc)
    return redacted_docs


def remove_overlapping_words(t1: str, t2: str, mask_token: str = "[MASK]", ignore_stopwords=False) -> str:
    """Replaces words in `t1` that occur in `t2` with `mask_token`.

    Also known as "lexical redaction".

    Optionally ignores english stopwords.
    
    """
    words_to_mask = set(t2.split())
    if ignore_stopwords:
        words_to_mask -= eng_stopwords
    
    return fixed_redact_str(t1, words_to_mask=words_to_mask, mask_token=mask_token)

def fixed_redact_str(text: str, words_to_mask: List[str], mask_token: str) -> str:
    for w in words_to_mask:
        text = re.sub(
            (r'\b{}\b').format(re.escape(w)),
            mask_token, text, count=0
        )
    return text

def redact(document: str, p: float, idf: Dict[str, float], mask_token: str):
    words = list(set(words_from_text(document)))
    words.sort(key=lambda w: (-idf.get(w, 0.0)))
    n = round(len(words) * p)
    return fixed_redact_str(text=document, words_to_mask=words[:n], mask_token=mask_token)

val_idf = None
def remove_words_val_idf(document: str, p: float, mask_token: str) -> str:
    global val_idf
    if val_idf is None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        val_idf_file_path = os.path.join(current_folder, 'val_100_idf.p')
        val_idf = pickle.load(open(val_idf_file_path, 'rb'))
    return redact(document=document, p=p, idf=val_idf, mask_token=mask_token)
    

if __name__ == '__main__':
    # print(remove_named_entities_spacy("Apple is looking. And looking. And looking at buying U.K. startup for $1 billion!"))
    data = {'input_text': {'table': {'column_header': ['nationality', 'name', 'article_title', 'occupation', 'birth_date'], 'row_number': [1, 1, 1, 1, 1], 'content': ['german', 'walter extra', 'walter extra\n', 'aircraft designer and manufacturer', '1954']}, 'context': 'walter extra\n'}, 'target_text': 'walter extra is a german award-winning aerobatic pilot , chief aircraft designer and founder of extra flugzeugbau -lrb- extra aircraft construction -rrb- , a manufacturer of aerobatic aircraft .\nextra was trained as a mechanical engineer .\nhe began his flight training in gliders , transitioning to powered aircraft to perform aerobatics .\nhe built and flew a pitts special aircraft and later built his own extra ea-230 .\nextra began designing aircraft after competing in the 1982 world aerobatic championships .\nhis aircraft constructions revolutionized the aerobatics flying scene and still dominate world competitions .\nthe german pilot klaus schrodt won his world championship title flying an aircraft made by the extra firm .\nwalter extra has designed a series of performance aircraft which include unlimited aerobatic aircraft and turboprop transports .\n'}
    from utils import create_document_and_profile_from_wikibio
    ex = create_document_and_profile_from_wikibio(data)
    print(ex['text1'])
    print()
    print(ex['text2'])
    print()
    print(remove_overlapping_words(ex['text1'], ex['text2']))
