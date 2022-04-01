## compute embeddings and similarities between wiki_bio training examples

from typing import Dict, List

import argparse
import os
import pathlib
import pickle

import datasets
import numpy as np
import pandas as pd
import scipy
import scipy.spatial
import tqdm
import torch
import transformers

from sentence_transformers import SentenceTransformer

# hide warning:
# TAPAS is a question answering model but you have not passed a query. Please be aware that the model will probably not behave correctly.
from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def map_ex(ex):
    """
    transforms wiki_bio example into a single str

    >>> ex['target_text']
    'walter extra is a german award-winning aerobatic pilot , chief aircraft designer and founder of extra....
    >>> ex['input_text']
    {'table': {
            'column_header': 
                ['nationality', 'name', 'article_title', 'occupation', 'birth_date'], 
            'row_number':
                [1, 1, 1, 1, 1], 
            'content': 
                ['german', 'walter extra', 'walter extra\n', 'aircraft designer and manufacturer', '1954']
            }, 
    'context':
        'walter extra\n'
    }
    """
    # transform table to str
    table_info = ex['input_text']['table']
    table_rows = list(zip(
        map(lambda s: s.strip(), table_info['column_header']),
        map(lambda s: s.strip(), table_info['content']))
    )
    table_text = '\n'.join([' | '.join(row) for row in table_rows])

    # <First paragraph of biography> + ' ' + <Table re-printed as a string>
    return { 'text': ex['target_text'] + ' ' + table_text }

def tapas_embeddings_from_dataset(dataset: datasets.Dataset) -> np.ndarray:
    """
    ex['input_text']: {
        'context': 'walter extra\n',
        'table': {'column_header': ['name',
                             'nationality',
                             'birth_date',
                             'article_title',
                             'occupation'],
        'content': ['walter extra',
                       'german',
                       '1954',
                       'walter extra\n',
                       'aircraft designer and manufacturer'],
        'row_number': [1, 1, 1, 1, 1]}
    }
    """
    def map_ex_to_df(ex: Dict) -> pd.DataFrame:
        table = ex['input_text']['table']
        return pd.DataFrame(columns=table['column_header'], data=[table['content']])
    tokenizer = transformers.TapasTokenizer.from_pretrained("google/tapas-base")
    model = transformers.TapasModel.from_pretrained("google/tapas-base")
    model.to(device)
    print('[1/2] generating dataframes')
    dataframes = [map_ex_to_df(ex) for ex in tqdm.tqdm(dataset, desc='creating dataframes', colour='#32cd32')]
    encodings = []
    print('[2/2] embedding dataframes')
    for df in tqdm.tqdm(dataframes, desc='embedding dataframes with tapas', colour='#ffc0cb'):
        inputs = tokenizer(
            table=df,
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        
        encodings.append(output.pooler_output.squeeze(dim=0).cpu())
    return torch.stack(encodings, dim=0).numpy()



def parse_args() -> argparse.Namespace():
    parser = argparse.ArgumentParser(
        description='precompute similarity matrix.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--k', type=int, default=2048, help='number of nearest-neighbors to use')
    parser.add_argument('--dataset_name', '--dataset', type=str, default='wiki_bio', help='dataset to use')
    parser.add_argument('--encoder', '--profile_encoder', type=str, default='tapas', choices=('tapas', 'st-paraphrase'), help='profile encoder to use')
    parser.add_argument('--split', type=str, default='train[:10%]', help='split to use, from dataset')
    parser.add_argument('--compute_neighbors', default=False,
        action='store_true', help='compute nearest-neighbors')
    return parser.parse_args()


def main(args: argparse.Namespace):
    dataset_name = args.dataset_name
    split = args.split
    k = args.k
    save_folder = os.path.join('precomputed_similarities', args.encoder, f'{dataset_name}__{split}__{k}')
    pathlib.Path(save_folder).mkdir(exist_ok=True, parents=True)
    # get data
    data = datasets.load_dataset(dataset_name, split=split)
    sentence_keys = data.map(map_ex)['text']

    # embed data
    print(f'Getting embeddings for {len(sentence_keys)} wikipedia article infoboxes...')


    if args.encoder == 'tapas':
        embeddings = tapas_embeddings_from_dataset(data)
    elif args.encoder == 'st-paraphrase':
        model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2').to(device)
        embeddings = model.encode(sentence_keys)
        # dimensionality: 768
    else:
        raise ValueError(f'unknown encoder: {args.encoder}')

    print('embeddings shape:', embeddings.shape)
    embeddings_path = os.path.join(save_folder, 'embeddings.p')
    pickle.dump(embeddings, open(embeddings_path, 'wb'))

    # process k neighbors for each thing
    print('creating str_to_idx')
    str_to_idx = {}
    for idx in range(len(embeddings)):
        str_to_idx[sentence_keys[idx]] = idx

    str_to_idx_path = os.path.join(save_folder, 'str_to_idx.p') 
    pickle.dump(str_to_idx, open(str_to_idx_path, 'wb'))

    if not args.compute_neighbors:
        print('Not computing neighbors, exiting early')

    # put data in tree
    print('Building KDTree...')
    tree = scipy.spatial.cKDTree(embeddings)

    neighbors = []
    print('Getting nearest neighbors...')
    # process k neighbors for each thing
    for idx in tqdm.trange(len(embeddings), desc='Getting nearest neighbors', colour='#008080'):
        # example result of query(embeddings[0], k=3): 
        #       (array([0.        , 3.30052274, 3.31010842]), array([  0, 655, 617]))
        dists, neighbor_idxs = tree.query(embeddings[idx], k=k)
        neighbors.append(neighbor_idxs)

    # store hashed example indices + nearest neighbor matrix
    neighbors_path = os.path.join(save_folder, 'neighbors.p')
    pickle.dump(neighbors, open(neighbors_path, 'wb'))

if __name__ == '__main__': 
    args = parse_args()
    main(args)