from typing import Callable, Dict, List, Tuple

import functools
import glob
import os
import random
import re

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

def highlight_masked_spans_html(text: str, mask_token: str) -> str:
    text = text.replace(mask_token, '<span style="background-color: black; color: black">XXXXX</span>')
    return f'<p>{text}</p>'

def redact_example(
    redact_func: Callable,
    example: Dict,
    suffix: str,
    include_profile: bool = True
) -> Dict:
    if include_profile:
        example[f'document_{suffix}'] = redact_func(example['document'], example['profile'])
    else:
        example[f'document_{suffix}'] = redact_func(example['document'])
    return example

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

# docs.streamlit.io/library/advanced-features/experimental-cache-primitives
@st.experimental_singleton
def load_adv_df():
    adv_df = None
    for model_name in ['model_4', 'model_5', 'model_6', 'model_7', 'model_8', 'model_9']:
        csv_filenames = glob.glob(f'./adv_csvs/{model_name}*/results_1_*0.csv')
        print(model_name, csv_filenames)
        for filename in csv_filenames:
            df = pd.read_csv(filename)
            df['model_name'] = re.search(r'adv_csvs/(model_\d.*)/.+.csv', filename).group(1)
            df['i'] = df.index
            mini_df = df[['perturbed_text', 'model_name', 'i']]
            
            mini_df = mini_df.iloc[:100]
            
            if adv_df is None:
                adv_df = mini_df
            else:
                adv_df = pd.concat((adv_df, mini_df), axis=0)
    
    # restore newlines
    adv_df['perturbed_text'] = adv_df['perturbed_text'].map(
        lambda text: text.replace('<SPLIT>', '\n')
    )

    # standardize to roberta-style masks
    adv_df['perturbed_text'] = adv_df['perturbed_text'].map(
        lambda text: text.replace('[MASK]', '<mask>')
    )
    # take first 100 words!
    def truncate_words(text: str, max_words: int = 100) -> str:
        words = text.split(' ')
        if len(words) > max_words:
            words = words[:max_words]
            text = ' '.join(words) + ' …'
        return text
    adv_df['perturbed_text'] = adv_df['perturbed_text'].map(
        truncate_words
    )
    
    return adv_df

def load_baseline_adv_df(val_dataset: datasets.Dataset) -> pd.DataFrame:
    mini_val_dataset = val_dataset[:1000]
    ner_df = pd.DataFrame(
        columns=['perturbed_text'],
        data=mini_val_dataset['document_redact_ner']
    )
    ner_df['model_name'] = 'named_entity'
    ner_df['i'] = ner_df.index
        
    lex_df = pd.DataFrame(
        columns=['perturbed_text'],
        data=mini_val_dataset['document_redact_lexical']
    )
    lex_df['model_name'] = 'lexical'
    lex_df['i'] = lex_df.index

    return pd.concat((lex_df, ner_df), axis=0)


def main():
    mask_token = '<mask>'
    val_dataset = load_val_dataset(mask_token)
    adv_df = load_adv_df()
    baseline_df = load_baseline_adv_df(val_dataset=val_dataset)
    adv_df = pd.concat((baseline_df, adv_df), axis=0)

    # fix weird ` quotes by replacing with '
    adv_df['perturbed_text'] = adv_df['perturbed_text'].map(
        lambda text: text.replace('`', '\'')
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    names = val_dataset['name']

    with st.sidebar:
        chosen_model_name = st.selectbox(
            label='Select a model',
            options=adv_df['model_name'].unique(),
        )

    st.header(f'Redacted document')

    this_adv_df = adv_df[adv_df['model_name'] == chosen_model_name]

    choice_idx = random.choice(range(len(this_adv_df)))
    ex = this_adv_df.iloc[choice_idx]
    print('choice_idx:', choice_idx)

    doc_text = ex['perturbed_text']
    st.subheader(ex['model_name'])
    st.write(
       f'{ex["i"]}  ' + highlight_masked_spans_html(doc_text, mask_token), unsafe_allow_html=True
    )
    k = 20
    topk_probs = np.arange(k)
    topk_profiles = np.array(val_dataset['profile'])[topk_probs]
    st.write('<hr>', unsafe_allow_html=True)
    st.header('Matching profiles')
    for prof_idx, prof in zip(topk_probs, topk_profiles):
        prob = 0
        color = '#008b00' if prof_idx == choice_idx else '#6b0000'
        prof_table = table_from_table_rows(prof)
        prof_name = name_from_table_rows(prof_table)
        st.write(f'<b style="font-size:22px;color:{color}">{prob*100:.2f}% · {prof_name}</b>', unsafe_allow_html=True)
        st.write(make_infobox_html(prof_table), unsafe_allow_html=True)
        st.write('<br>', unsafe_allow_html=True)


if __name__ == '__main__': main()