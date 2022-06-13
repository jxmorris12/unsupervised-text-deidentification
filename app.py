from typing import Callable, Dict, List, Tuple

import functools
import os

import datasets
import streamlit as st
import pandas as pd
import numpy as np
import torch
import tqdm

from redact import remove_named_entities_spacy, remove_overlapping_words
from utils import create_document_and_profile_from_wikibio, name_from_table_rows


# num_cpus = 1 # Multiprocessing will break with streamlit!
num_cpus = len(os.sched_getaffinity(0))

def redact_example(
    redact_func: Callable,
    example: Dict,
    suffix: str,
    include_profile: bool = True
):
    if include_profile:
        example[f'document_{suffix}'] = redact_func(example['document'], example['profile'])
    else:
        example[f'document_{suffix}'] = redact_func(example['document'])
    return example

# docs.streamlit.io/library/advanced-features/experimental-cache-primitives
# @st.experimental_singleton
def load_val_dataset(mask_token: str) -> datasets.Dataset:
    val_dataset = datasets.load_dataset(
        'wiki_bio', split='val[:20%]', version='1.2.0'
    )
    val_dataset = val_dataset.map(create_document_and_profile_from_wikibio)

    # Lexical (word overlap) redaction
    lexical_redact_func = functools.partial(
        remove_overlapping_words, mask_token=mask_token)
    val_dataset = val_dataset.map(
        lambda ex: redact_example(
            redact_func=lexical_redact_func, example=ex, suffix='redact_lexical', include_profile=True),
        num_proc=1
    )

    #  NER redaction
    ner_redact_func = functools.partial(
        remove_named_entities_spacy, mask_token=mask_token
    )
    val_dataset = val_dataset.map(
        lambda ex: redact_example(redact_func=ner_redact_func, example=ex, suffix='redact_ner', include_profile=False),
        num_proc=1
    )

    return val_dataset

def make_infobox_html(table: List[Tuple[str, str]]) -> str:
    s = '<table><tbody>'
    # print('table:', table)
    for rkey, rval in table:
        s += '<tr>'
        s += f'<th><b>{rkey}</b></th>'
        s += f'<td>{rval}</td>'
        s += '</tr>'
    s += '</tbody></table>'
    return s

def table_from_table_rows(rows_str: str) -> List[Tuple[str, str]]:
    return [[el.strip() for el in r.split('||')] for r in rows_str.split('\n')]

def main():
    val_dataset = load_val_dataset('<mask>')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print('getting names...')
    names = [
        name_from_table_rows(table_from_table_rows(prof))
        for prof in val_dataset['profile']
    ]
    print('...got names...')

    with st.sidebar:
        choice_idx = st.selectbox(
            label='select a person',
            options=list(range(len(names))),
            format_func=lambda idx: names[idx]
        )
    print('choice_idx:', choice_idx)

    # st.subheader('Document')
    st.header(f'{prof_or_doc.capitalize()} search')


    doc_text = dm.val_dataset['document'][choice_idx]
    doc = st.text_area(
        label='Document',
        value=doc_text
    )
    k = 20
    topk_probs = np.arange(k)
    topk_profiles = np.array(dm.val_dataset['profile'])[
        topk_probs
    ]
    st.write('<hr>', unsafe_allow_html=True)
    st.subheader('Matching profiles')
    for prof_idx, prof in zip(topk_probs, topk_profiles):
        prob = 0
        color = '#008b00' if prof_idx == choice_idx else '#6b0000'
        prof_table = table_from_table_rows(prof)
        prof_name = name_from_table_rows(prof_table)
        st.write(f'<b style="font-size:22px;color:{color}">{prob*100:.2f}% · {prof_name}</b>', unsafe_allow_html=True)
        st.write(make_infobox_html(prof_table), unsafe_allow_html=True)
        st.write('<br>', unsafe_allow_html=True)


if __name__ == '__main__': main()