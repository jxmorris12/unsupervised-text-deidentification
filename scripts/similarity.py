from typing import List

import os
import pickle

import datasets
import numpy as np
import tqdm

from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi


eng_stopwords = stopwords.words('english')

def get_words_from_doc(s: List[str]) -> List[str]:
    words = s.split()
    return [w for w in words if not w in eng_stopwords]


def get_top_matches_bm25():
    # split = 'train[:1%]'
    # split = 'train[:10%]'
    split = 'val[:100%]'
    # split = 'val[:20%]'
    train_data = datasets.load_dataset('wiki_bio', split=split, version='1.2.0')
    k = 256

    def make_table_str(ex):
        ex['table_str'] = (
            ' '.join(ex['input_text']['table']['column_header'] + ex['input_text']['table']['content'])
        )
        return ex

    train_data = train_data.map(make_table_str)
    corpus = train_data['table_str']

    print("tokenizing corpus")
    tokenized_corpus = [
        get_words_from_doc(doc) for doc in corpus
    ]
    # (Pdb) p tokenized_corpus[-1]
    # ['nfl', 'position', 'highschool', 'height_in', 'statvalue', 'debutteam', 'birth_date', 'article_title', 
    # 'draftyear', 'number', 'finalteam', 'weight', 'debutyear', 'statlabel', 'draftpick', 'name', 'college',
    # 'height_ft', 'finalyear', 'birth_place', 'draftroundgle378870', 'offensive', 'tackle', 'oakland',
    # '-lrb-', 'ca', '-rrb-', "o'dowd", '5', '154', '154', '1', 'indianapolis', 'colts', '25', 'may', '1976',
    # 'tarik', 'glenn\n', '1997', '78', 'indianapolis', 'colts', '332', '1997', 'games', 'played', 'games',
    # 'started', 'fumbles', 'recovered', '19', 'tarik', 'glenn', 'california', '6', '2006', 'cleveland', ',',
    # 'ohio', '1']

    print("creating search index")
    bm25 = BM25Okapi(tokenized_corpus)

    #####################################################################
    # for i in range(10):
    #     ex = train_data[i]
    #     query = ex["target_text"].split()
    #     qscores = bm25.get_scores(query)
    #     qtop_k = qscores.argsort()[::-1][:1+k]
    #     breakpoint()
    ####################################################################

    print("getting top-k matches")
    top_matches = []
    def get_top_k(ex):
        query = ex["target_text"].split()
        ex["top_k"] = bm25.get_scores(query).argsort()[::-1][:1+k]
        return ex
    
    num_proc = min(8, len(os.sched_getaffinity(0)))
    train_data = train_data.map(get_top_k, num_proc=num_proc)
    # train_data = train_data.map(get_top_k)

    top_matches = np.array(train_data['top_k'])
    
    top_matches = np.array(top_matches)
    outfile = f"nearest_neighbors/nn__{split}__{k}.p"
    pickle.dump(top_matches, open(outfile, "wb"))
    print(f"wrote {len(top_matches)} top matches to file {outfile}")

if __name__ == '__main__':
    get_top_matches_bm25()
