from typing import Dict, List

import os
import pickle

import datasets
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi


eng_stopwords = stopwords.words('english')

from tqdm.auto import tqdm
tqdm.pandas()


def get_words_from_doc(s: List[str]) -> List[str]:
    words = s.split()
    return [w for w in words if not w in eng_stopwords]

def combine_dicts(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    all_keys = a.keys() | b.keys()
    return {k: a.get(k, 0) + b.get(k, 0) for k in all_keys}

def get_top_matches_adv_data_bm25():
    split = 'val[:20%]'
    train_data = datasets.load_dataset('wiki_bio', split=split, version='1.2.0')
    k = 256

    def make_table_str(ex):
        ex['table_str'] = (
            ' '.join(ex['input_text']['table']['column_header'] + ex['input_text']['table']['content'])
        )
        return ex

    train_data = train_data.map(make_table_str)
    profile_corpus = train_data['table_str']
    document_corpus = train_data['target_text']

    print("tokenizing corpii")
    tokenized_document_corpus = [
        get_words_from_doc(doc) for doc in document_corpus
    ]
    tokenized_profile_corpus = [
        get_words_from_doc(prof) for prof in profile_corpus
    ]

    print("creating search index")

    bm25 = BM25Okapi(tokenized_profile_corpus)

    ######################################################################
    # combine corpuses for idf calculation
    # doc_bm25 = BM25Okapi(tokenized_document_corpus)
    # ndd = doc_bm25._initialize(tokenized_document_corpus)
    
    # prof_bm25 = BM25Okapi(tokenized_profile_corpus)
    # ndp = prof_bm25._initialize(tokenized_profile_corpus)

    # bm25 = BM25Okapi(tokenized_profile_corpus)
    # bm25._calc_idf(combine_dicts(ndd, ndp))
    # bm25.avgdl = (bm25.avgdl + doc_bm25.avgdl) / 2.0
    # bm25.doc_freqs = combine_dicts(bm25.doc_freqs, doc_bm25.doc_freqs)
    ######################################################################

    # adv_csv_filename = 'adv_csvs/model_1/results_1_1000.csv' # 0.670 (prof corpus, removed stopwords)
    # adv_csv_filename = 'adv_csvs/model_1/results_10_1000.csv' # 0.397 (prof corpus, removed stopwords)
    adv_csv_filename = 'adv_csvs/model_1/results_100_1000.csv' # 0.397 (prof corpus, removed stopwords)
    # adv_csv_filename = 'adv_csvs/model_2/results_1_1000.csv' # 0.469 (prof corpus, removed stopwords)
    # adv_csv_filename = 'adv_csvs/model_2/results_10_1000.csv' # 0.284 (prof corpus, removed stopwords)

    print(f"loading adversarial data from {adv_csv_filename}")
    
    adv_df = pd.read_csv(adv_csv_filename)
    adv_df["adv_idx"] = pickle.load(open('nearest_neighbors/nn__idxs.p', 'rb'))
    adv_df = adv_df[adv_df["result_type"] == "Successful"]

    # ######################################################################
    # idx = 14
    # ex = adv_df.iloc[idx]
    # query = ex["perturbed_text"].split()
    # top_k = bm25.get_scores(query).argsort()[::-1][:1+k]
    # breakpoint()
    ########################################################################

    print("getting top-k matches")
    top_matches = []
    def get_top_k(ex):
        query = ex["perturbed_text"].split()
        top_k = bm25.get_scores(query).argsort()[::-1]
        ex["correct_idx"] = top_k.tolist().index(ex["adv_idx"])
        ex["is_correct"] = 1 if top_k[0] == ex["adv_idx"] else 0
        return ex
    
    num_proc = min(8, len(os.sched_getaffinity(0)))
    out_df = adv_df.progress_apply(get_top_k, axis=1)
    print(out_df["is_correct"].mean())
    breakpoint()


if __name__ == '__main__':
    get_top_matches_adv_data_bm25()
