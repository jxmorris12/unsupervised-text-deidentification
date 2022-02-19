## compute similarity between wiki_bio training examples
## we will use for a q

import argparse
import os
import pathlib
import pickle

import datasets
import scipy
import scipy.spatial
import tqdm

from sentence_transformers import SentenceTransformer

def map_ex(ex):
    """
    transforms wiki_bio example into a single str

    >>> ex['target_text']
    'walter extra is a german award-winning aerobatic pilot , chief aircraft designer and founder of extra....
    >>> ex['input_text']
    {'table': {'column_header': ['nationality', 'name', 'article_title', 'occupation', 'birth_date'], 'row_number': [1, 1, 1, 1, 1], 'content': ['german', 'walter extra', 'walter extra\n', 'aircraft designer and manufacturer', '1954']}, 'context': 'walter extra\n'}
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

def parse_args() -> argparse.Namespace():
    parser = argparse.ArgumentParser(
        description='precompute similarity matrix.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--k', type=int, default=128, help='number of nearest-neighbors to use')
    parser.add_argument('--dataset_name', '--dataset', type=str, default='wiki_bio', help='dataset to use')
    parser.add_argument('--split', type=str, default='train', help='split to use, from dataset')
    return parser.parse_args()

def main(args: argparse.Namespace):
    dataset_name = args.dataset_name
    split = args.split
    k = args.k
    model_folder = os.path.join('precomputed_similarities', f'{dataset_name}__{split}__{k}')
    pathlib.Path(model_folder).mkdir(exist_ok=True, parents=True)
    # get data
    data = datasets.load_dataset(dataset_name, split=split)
    sentences = data.map(map_ex)['text']

    # embed data
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    print('embeddings shape:', embeddings.shape)

    # put data in tree
    print('Building KDTree...')
    tree = scipy.spatial.cKDTree(embeddings)

    str_to_idx = {}
    neighbors = []
    print('Getting nearest neighbors...')
    # process k neighbors for each thing
    for idx in tqdm.trange(len(embeddings), desc='Getting nearest neighbors'):
        # example result of query(embeddings[0], k=3): 
        #       (array([0.        , 3.30052274, 3.31010842]), array([  0, 655, 617]))
        dists, neighbor_idxs = tree.query(embeddings[0], k=k)
        neighbors.append(neighbor_idxs)
        str_to_idx[sentences[idx]] = idx

    # store hashed example indices + nearest neighbor matrix
    str_to_idx_path = os.path.join(model_folder, 'str_to_idx.p') 
    pickle.dump(str_to_idx, open(str_to_idx_path, 'wb'))
    neighbors_path = os.path.join(model_folder, 'neighbors.p')
    pickle.dump(neighbors, open(neighbors_path, 'wb'))

if __name__ == '__main__': 
    args = parse_args()
    main(args)