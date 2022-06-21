from typing import Any, Tuple
import pickle

import numpy as np
import os
import torch
from tqdm import tqdm

from model_cfg import model_paths_dict


num_cpus = len(os.sched_getaffinity(0))


def get_profile_embedding_dir_by_model_key(model_key: str) -> str:
    current_folder = os.path.dirname(os.path.abspath(__file__))
    base_folder = os.path.join(current_folder, os.pardir)
    return os.path.normpath(
        os.path.join(base_folder, 'embeddings', 'profile', model_key)
    )

def precompute_profile_embeddings(
        model: Any, dm: Any
        # For some reason have to annotate WikipediaDataModule as `Any` type
        # to avoid a circular import error.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.profile_model.cuda()
    model.profile_embed.cuda()
    model.profile_model.eval()
    model.profile_embed.eval()

    train_profile_embeddings = np.zeros((len(dm.train_dataset), model.shared_embedding_dim))
    for val_batch in tqdm(dm.train_dataloader(), desc="Precomputing train profile embeddings", colour="blue", leave=False):
        with torch.no_grad():
            profile_embeddings = model.forward_profile(batch=val_batch)
        train_profile_embeddings[val_batch["text_key_id"]] = profile_embeddings.cpu()
    train_profile_embeddings = torch.tensor(train_profile_embeddings, dtype=torch.float32)

    val_profile_embeddings = np.zeros((len(dm.val_dataset), model.shared_embedding_dim))
    for val_batch in tqdm(dm.val_dataloader()[0], desc="Precomputing val profile embeddings", colour="green", leave=False):
        with torch.no_grad():
            profile_embeddings = model.forward_profile(batch=val_batch)
        val_profile_embeddings[val_batch["text_key_id"]] = profile_embeddings.cpu()
    val_profile_embeddings = torch.tensor(val_profile_embeddings, dtype=torch.float32)

    test_profile_embeddings = np.zeros((len(dm.test_dataset), model.shared_embedding_dim))
    for test_batch in tqdm(dm.test_dataloader(), desc="Precomputing test profile embeddings", colour="magenta", leave=False):
        with torch.no_grad():
            profile_embeddings = model.forward_profile(batch=test_batch)
        test_profile_embeddings[test_batch["text_key_id"]] = profile_embeddings.cpu()
    test_profile_embeddings = torch.tensor(test_profile_embeddings, dtype=torch.float32)
    
    return (
        train_profile_embeddings, val_profile_embeddings, test_profile_embeddings
    )


def precompute_profile_embeddings_for_model_key(model_key: str):
    from dataloader import WikipediaDataModule
    from model import CoordinateAscentModel

    checkpoint_path = model_paths_dict[model_key]
    model = CoordinateAscentModel.load_from_checkpoint(checkpoint_path)
    dm = WikipediaDataModule(
        document_model_name_or_path=model.document_model_name_or_path,
        profile_model_name_or_path=model.profile_model_name_or_path,
        dataset_name='wiki_bio',
        dataset_train_split='train[:100%]',
        dataset_val_split='val[:100%]',
        dataset_test_split='test[:100%]',
        dataset_version='1.2.0',
        num_workers=num_cpus,
        train_batch_size=256,
        eval_batch_size=256,
        max_seq_length=128,
        sample_spans=False,
    )
    dm.setup("fit")

    model_embeddings_path = get_profile_embedding_dir_by_model_key(
        model_key=model_key
    )
    os.makedirs(model_embeddings_path, exist_ok=True)

    train_profile_embeddings, val_profile_embeddings, test_profile_embeddings = (
        precompute_profile_embeddings(model=model, dm=dm)
    )
    pickle.dump(
        train_profile_embeddings,
        open(os.path.join(model_embeddings_path, 'train.pkl'), 'wb')
    )
    pickle.dump(
        val_profile_embeddings,
        open(os.path.join(model_embeddings_path, 'val.pkl'), 'wb')
    )
    pickle.dump(
        test_profile_embeddings,
        open(os.path.join(model_embeddings_path, 'test.pkl'), 'wb')
    )

    return {
        'train': train_profile_embeddings,
        'val': val_profile_embeddings,
        'test': test_profile_embeddings
    }


def get_profile_embeddings_by_model_key(model_key: str):
    model_embeddings_path = get_profile_embedding_dir_by_model_key(
        model_key=model_key
    )
    if (not os.path.exists(model_embeddings_path)) or (not os.path.exists(os.path.join(model_embeddings_path, 'test.pkl'))):
        return precompute_profile_embeddings_for_model_key(model_key=model_key)

    else:
        train_embeddings_path = os.path.join(model_embeddings_path, 'train.pkl')
        train_embeddings = pickle.load(open(train_embeddings_path, 'rb'))
        print(f">> loaded {len(train_embeddings)} train embeddings from", train_embeddings_path)

        val_embeddings_path = os.path.join(model_embeddings_path, 'val.pkl')
        val_embeddings = pickle.load(open(val_embeddings_path, 'rb'))
        print(f">> loaded {len(val_embeddings)} val embeddings from", val_embeddings_path)

        test_embeddings_path = os.path.join(model_embeddings_path, 'test.pkl')
        test_embeddings = pickle.load(open(test_embeddings_path, 'rb'))
        print(f">> loaded {len(test_embeddings)} test embeddings from", test_embeddings_path)
        return {
            'train': train_embeddings,
            'val': val_embeddings,
            'test': test_embeddings
        }