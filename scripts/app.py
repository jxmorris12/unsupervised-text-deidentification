from typing import List, Tuple

import os

import streamlit as st
import pandas as pd
import numpy as np
import torch
import tqdm

from dataloader import WikipediaDataModule
from model import DocumentProfileMatchingTransformer
from utils import name_from_table_rows


# num_cpus = 1 # Multiprocessing will break with streamlit!
num_cpus = os.cpu_count()

def precompute_profile_embeddings(model: DocumentProfileMatchingTransformer, datamodule: WikipediaDataModule):
    model.document_model.cuda()
    model.document_embed.cuda()
    model.document_model.eval()
    model.document_embed.eval()
    print('Precomputing document embeddings before first epoch...')

    model.val_document_embeddings = np.zeros((len(datamodule.val_dataset), model.shared_embedding_dim))
    for val_batch in tqdm.tqdm(datamodule.val_dataloader(), desc="[1/2] Precomputing val embeddings - document", colour="cyan", leave=False):
        with torch.no_grad():
            document_embeddings = model.forward_document_text(text=val_batch["document"])
        model.val_document_embeddings[val_batch["text_key_id"]] = document_embeddings.cpu()
    model.val_document_embeddings = torch.tensor(model.val_document_embeddings, dtype=torch.float32)
    
    model.profile_model.cuda()
    model.profile_model.eval()
    print('Precomputing profile embeddings before first epoch...')
    model.val_profile_embeddings = np.zeros((len(datamodule.val_dataset), model.shared_embedding_dim))
    for val_batch in tqdm.tqdm(datamodule.val_dataloader(), desc="[2/2] Precomputing val embeddings - profile", colour="green", leave=False):
        with torch.no_grad():
            profile_embeddings = model.forward_profile_text(text=val_batch["profile"])
        model.val_profile_embeddings[val_batch["text_key_id"]] = profile_embeddings.cpu()
    model.val_profile_embeddings = torch.tensor(model.val_profile_embeddings, dtype=torch.float32)
    model.profile_model.train()


# docs.streamlit.io/library/advanced-features/experimental-cache-primitives
@st.experimental_singleton
def load_everything() -> Tuple[WikipediaDataModule, DocumentProfileMatchingTransformer]:
    with st.spinner('Loading model...'):
        model = DocumentProfileMatchingTransformer.load_from_checkpoint(
            # distilbert-distilbert model
            #    '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/distilbert-base-uncased__dropout_0.8_0.8/deid-wikibio_default/1irhznnp_130/checkpoints/epoch=25-step=118376.ckpt',
            # roberta-distilbert model
            # '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/roberta__distilbert-base-uncased__dropout_0.8_0.8/deid-wikibio_default/1f7mlhxn_162/checkpoints/epoch=16-step=309551.ckpt',
            # roberta-distilbert model trained for longer
            '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/roberta__distilbert-base-uncased__dropout_0.8_0.8/deid-wikibio_default/3nbt75gp_171/checkpoints/epoch=20-step=382387.ckpt',
            document_model_name_or_path='roberta-base',
            profile_model_name_or_path='distilbert-base-uncased',
            num_workers=min(8, num_cpus),
            train_batch_size=64,
            eval_batch_size=64,
            learning_rate=1e-6,
            max_seq_length=256,
            pretrained_profile_encoder=False,
            word_dropout_ratio=0.0,
            word_dropout_perc=0.0,
            lr_scheduler_factor=0.5,
            lr_scheduler_patience=3,
            adversarial_mask_k_tokens=0,
            train_without_names=False,
        )
    doc_mask_token = model.document_tokenizer.mask_token

    with st.spinner('Loading dataset...'):
        dm = WikipediaDataModule(
            mask_token=doc_mask_token,
            dataset_name='wiki_bio',
            dataset_train_split='train[:1%]', # unused for now
            dataset_val_split='val[:20%]',
            dataset_version='1.2.0',
            num_workers=min(8, num_cpus),
            train_batch_size=64,
            eval_batch_size=64,
        )
        dm.setup("fit")

    
    with st.spinner('Precomputing embeddings...'):
        precompute_profile_embeddings(model, dm)

    return dm, model

def get_top_profiles_probs(model: DocumentProfileMatchingTransformer, doc: str) -> torch.Tensor:
    model_device = next(model.profile_model.parameters()).device
    with torch.no_grad():
        document_embeddings = model.forward_document_text(text=[doc])
        # TODO: fix temperature value :(
        document_to_profile_logits = document_embeddings @ model.val_profile_embeddings.T.to(model_device) * (1/10.)
        document_to_profile_probs = torch.nn.functional.softmax(
            document_to_profile_logits, dim=-1
        )
    assert document_to_profile_probs.shape == (1, len(model.val_profile_embeddings))
    return document_to_profile_probs.squeeze(0).cpu()

def get_top_documents_probs(model: DocumentProfileMatchingTransformer, prof: str) -> torch.Tensor:
    model_device = next(model.document_model.parameters()).device
    with torch.no_grad():
        profile_embeddings = model.forward_profile_text(text=[prof])
        # TODO: fix temperature value :(
        profile_to_document_logits = profile_embeddings @ model.val_document_embeddings.T.to(model_device) * (1/10.)
        profile_to_document_probs = torch.nn.functional.softmax(
            profile_to_document_logits, dim=-1
        )
    assert profile_to_document_probs.shape == (1, len(model.val_document_embeddings))
    return profile_to_document_probs.squeeze(0).cpu()

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
    return [[el.strip() for el in r.split('|')] for r in rows_str.split('\n')]

def main():
    dm, model = load_everything()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    st.write(f'mask token: **{model.document_tokenizer.mask_token}**')

    val_batch = next(iter(dm.val_dataloader()))
    #
    # val_batch keys: 
    #    [
    #       'document', 'profile', 'document_redact_lexical',
    #       'document_redact_ner', 'text_key_id'
    #    ]
    #

    print('getting names...')
    names = [
        name_from_table_rows(table_from_table_rows(prof))
        for prof in dm.val_dataset['profile']
    ]
    print('...got names...')

    with st.sidebar:
        choice_idx = st.selectbox(
            label='select a person',
            options=list(range(len(names))),
            format_func=lambda idx: names[idx]
        )
        prof_or_doc = st.radio(
            label='search by:',
            options=('document', 'profile')
        )
    print('choice_idx:', choice_idx)

    # st.subheader('Document')
    st.header(f'{prof_or_doc.capitalize()} search')


    if prof_or_doc == 'profile':
        prof_text = dm.val_dataset['profile'][choice_idx]
        prof = st.text_area(
            label='Profile',
            value=prof_text
        )
        probs = get_top_documents_probs(model, prof)
        k = 20
        topk_probs = (-probs).argsort()[:k]
        topk_documents = np.array(dm.val_dataset['document'])[
            topk_probs
        ]
        st.write('<hr>', unsafe_allow_html=True)
        st.subheader('Matching documents')
        for doc_idx, doc in zip(topk_probs, topk_documents):
            prob = probs[doc_idx]
            color = '#008b00' if choice_idx == doc_idx else '#6b0000'
            prof_name = name_from_table_rows(table_from_table_rows(dm.val_dataset['profile'][doc_idx]))
            st.write(f'<b style="font-size:22px;color:{color}">{prob*100:.2f}% · {prof_name}</b>', unsafe_allow_html=True)
            st.write(doc, unsafe_allow_html=False)
            st.write('<br>', unsafe_allow_html=True)
    else:
        doc_text = dm.val_dataset['document'][choice_idx]
        doc = st.text_area(
            label='Document',
            value=doc_text
        )
        probs = get_top_profiles_probs(model, doc)
        k = 20
        topk_probs = (-probs).argsort()[:k]
        topk_profiles = np.array(dm.val_dataset['profile'])[
            topk_probs
        ]
        st.write('<hr>', unsafe_allow_html=True)
        st.subheader('Matching profiles')
        for prof_idx, prof in zip(topk_probs, topk_profiles):
            prob = probs[prof_idx]
            color = '#008b00' if prof_idx == choice_idx else '#6b0000'
            prof_table = table_from_table_rows(prof)
            prof_name = name_from_table_rows(prof_table)
            st.write(f'<b style="font-size:22px;color:{color}">{prob*100:.2f}% · {prof_name}</b>', unsafe_allow_html=True)
            st.write(make_infobox_html(prof_table), unsafe_allow_html=True)
            st.write('<br>', unsafe_allow_html=True)

    # doc_col, prof_col = st.columns(2)
    
    # Document column and profile column
    # st.title('data')
    # df = pd.DataFrame(val_batch)
    # st.table(df)

if __name__ == '__main__': main()