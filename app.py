from typing import Tuple

import os

import streamlit as st
import pandas as pd
import numpy as np

from dataloader import WikipediaDataModule
from model import DocumentProfileMatchingTransformer


num_cpus = os.cpu_count()

def load_stuff() -> Tuple[WikipediaDataModule, DocumentProfileMatchingTransformer]:
    # model = DocumentProfileMatchingTransformer(
    model = DocumentProfileMatchingTransformer.load_from_checkpoint(
        # distilbert-distilbert model
        #    '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/distilbert-base-uncased__dropout_0.8_0.8/deid-wikibio_default/1irhznnp_130/checkpoints/epoch=25-step=118376.ckpt',
        # roberta-distilbert model
        '/home/jxm3/research/deidentification/unsupervised-deidentification/saves/roberta__distilbert-base-uncased__dropout_0.8_0.8/deid-wikibio_default/1f7mlhxn_162/checkpoints/epoch=16-step=309551.ckpt',
        document_model_name_or_path=args.document_model_name,
        profile_model_name_or_path=args.profile_model_name,
        num_workers=min(8, num_cpus),
        train_batch_size=64,
        eval_batch_size=64,
        learning_rate=1e-6,
        max_seq_length=256,
        pretrained_profile_encoder=False,
        word_dropout_ratio=0.0,
        word_dropout_perc=0.0,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        adversarial_mask_k_tokens=args.adversarial_mask_k_tokens,
        train_without_names=args.train_without_names,
    )
    doc_mask_token = model.document_tokenizer.mask_token
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

    return dm, model


def main():
    data_load_state = st.text('Loading data...')

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write('<Writing raw data>')

    st.subheader('Number of pickups by hour')

if __name__ == '__main__': main()