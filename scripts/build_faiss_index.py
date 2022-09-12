import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

from typing import Tuple
import os
import pickle

import numpy as np
import tqdm

from datamodule import WikipediaDataModule
from model import CoordinateAscentModel
from model_cfg import model_paths_dict
from utils import get_profile_embeddings_by_model_key, get_profile_embeddings_dir_by_model_key

num_cpus = len(os.sched_getaffinity(0))

def load_dm(model_key: str) -> Tuple[CoordinateAscentModel, WikipediaDataModule]:
    checkpoint_path = model_paths_dict[model_key]
    assert isinstance(checkpoint_path, str), f"invalid checkpoint_path {checkpoint_path} for {model_key}"
    print(f"running eval on {model_key} loaded from {checkpoint_path}")
    model = CoordinateAscentModel.load_from_checkpoint(
        checkpoint_path
    )

    print(f"loading data with {num_cpus} CPUs")
    dm = WikipediaDataModule(
        document_model_name_or_path=model.document_model_name_or_path,
        profile_model_name_or_path=model.profile_model_name_or_path,
        dataset_name='wiki_bio',
        dataset_train_split='train[:100%]',
        dataset_val_split='val[:100%]',
        dataset_test_split='test[:100%]',
        dataset_version='1.2.0',
        num_workers=num_cpus,
        train_batch_size=64,
        eval_batch_size=64,
        max_seq_length=128,
        sample_spans=False,
    )
    dm.setup("fit")

    return dm

def dump_embedding_nearest_neighbors(prefix: str, out_dir: str, embeddings: np.ndarray, k: int = 16) -> None:
    print(f"computing nn for {prefix} embeddings")
    neighbors = []
    embeddings = embeddings.cuda()
    i = 0
    batch_size = 256
    pbar = tqdm.tqdm(total=len(embeddings), desc="finding nearest neighbors", colour="red")
    while i < len(embeddings):
        emb_batch = embeddings[i:i+batch_size].cuda()
        emb_scores = emb_batch @ embeddings.T
        nn_i = (-emb_scores).argsort(dim=1)[:, :k+1].cpu().numpy().tolist()
        neighbors.extend(nn_i)
        #
        i += batch_size
        pbar.update(batch_size)
    pickle.dump(neighbors, open(os.path.join(out_dir, f'{prefix}_nn.p'), 'wb'))
    print(f"done computing {prefix} nn!")


def main(model_key: str):
    print(f"loading datasets for model_key {model_key}")
    dm = load_dm(model_key=model_key)
    
    print(f"getting profile embeddings for model_key {model_key}")
    profile_embeddings = get_profile_embeddings_by_model_key(model_key=model_key)
    
    out_dir = get_profile_embeddings_dir_by_model_key(model_key=model_key)

    dump_embedding_nearest_neighbors(prefix="test", out_dir=out_dir, embeddings=profile_embeddings['test'])
    dump_embedding_nearest_neighbors(prefix="val", out_dir=out_dir, embeddings=profile_embeddings['val'])
    dump_embedding_nearest_neighbors(prefix="train", out_dir=out_dir, embeddings=profile_embeddings['train'])

    # dm.test_dataset = dm.test_dataset.add_column('embeddings', profile_embeddings['test'].tolist())
    # dm.test_dataset.add_faiss_index(column='embeddings')
    # dm.test_dataset.save_faiss_index(
    #     'embeddings', os.path.join(out_dir, 'test_index.faiss')
    # )

    # print("building index for val dataset . . .")
    # dm.val_dataset = dm.val_dataset.add_column(
    #     'embeddings', profile_embeddings['val'].numpy().tolist()
    # )
    # dm.val_dataset.add_faiss_index(column='embeddings')
    # dm.val_dataset.save_faiss_index(
    #     'embeddings', os.path.join(out_dir, 'val_index.faiss')
    # )

    # print("building index for train dataset . . .")
    # dm.train_dataset = dm.train_dataset.add_column(
    #     'embeddings', profile_embeddings['train'].numpy().tolist()
    # )
    # dm.train_dataset.add_faiss_index(column='embeddings')
    # dm.train_dataset.save_faiss_index(
    #     'embeddings', os.path.join(out_dir, 'train_index.faiss')
    # )

if __name__ == '__main__':
    MODEL_KEY = 'model_3_3'
    main(model_key=MODEL_KEY)
