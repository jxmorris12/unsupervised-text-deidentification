import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

import pickle

import datasets
import numpy as np

from utils import create_document_and_profile_from_wikibio


def main():
    val_dataset = datasets.load_dataset('wiki_bio', split='val[:20%]', version='1.2.0')
    val_dataset= val_dataset.map(
            create_document_and_profile_from_wikibio
        )
    val_dataset = val_dataset.add_column(
        "text_key_id", 
        list(range(len(val_dataset)))
    )
    dataset = [ex for ex in val_dataset if ex['document'].count(' ') < 100]
    idxs = [e['text_key_id'] for e in dataset[:1000]]
    pickle.dump(np.array(idxs), open('nearest_neighbors/nn__idxs.p', 'wb'))

if __name__ == '__main__': main()