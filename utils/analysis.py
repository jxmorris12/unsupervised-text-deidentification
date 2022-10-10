from typing import List, Tuple

import os
import pickle
import re

import datasets
import pandas as pd
import torch
import tqdm

from datamodule import WikipediaDataModule
from model import CoordinateAscentModel
from model_cfg import model_paths_dict
from utils import get_profile_embeddings


datasets.utils.logging.set_verbosity_error()

num_cpus = len(os.sched_getaffinity(0))
words_from_text_re = re.compile(r'\b\w+\b')

REID_MODEL_KEYS = ['model_3_1', 'model_3_2', 'model_3_3', 'model_3_3__idf', 'model_3_4__idf']


def load_baselines_csv(max_num_samples: int = 100) -> pd.DataFrame:
    dm = WikipediaDataModule(
        document_model_name_or_path='roberta-base',
        profile_model_name_or_path='roberta-base',
        dataset_name='wiki_bio',
        dataset_train_split='train[:256]',
        dataset_val_split='val[:256]',
        dataset_test_split=f'test[:{max_num_samples}]',
        dataset_version='1.2.0',
        num_workers=num_cpus,
        train_batch_size=256,
        eval_batch_size=256,
        max_seq_length=128,
        sample_spans=False,
        do_bert_ner_redaction=True,
    )
    dm.setup("fit")

    # Load baseline redacted data
    mini_test_dataset = dm.test_dataset[:max_num_samples]
    doc_df = pd.DataFrame(
        columns=['perturbed_text'],
        data=mini_test_dataset['document']
    )
    doc_df['model_name'] = 'document'
    doc_df['i'] = doc_df.index
    
    ner_df = pd.DataFrame(
        columns=['perturbed_text'],
        data=mini_test_dataset['document_redact_ner_bert']
    )
    ner_df['model_name'] = 'named_entity'
    ner_df['i'] = ner_df.index
        
    lex_df = pd.DataFrame(
        columns=['perturbed_text'],
        data=mini_test_dataset['document_redact_lexical']
    )
    lex_df['model_name'] = 'lexical'
    lex_df['i'] = lex_df.index
    lex_df = lex_df.iloc[:max_num_samples]

    # Combine both adversarial and baseline redacted data
    full_df = pd.concat((doc_df, lex_df, ner_df), axis=0)
    
    # Put newlines back
    full_df['perturbed_text'] = full_df['perturbed_text'].apply(lambda s: s.replace('<SPLIT>', '\n'))

    # Standardize mask tokens
    full_df['perturbed_text'] = full_df['perturbed_text'].apply(lambda s: s.replace('[MASK]', dm.mask_token))
    full_df['perturbed_text'] = full_df['perturbed_text'].apply(lambda s: s.replace('<mask>', dm.mask_token))
    
    # Fair truncation
    full_df['original_num_words'] = full_df['perturbed_text'].map(lambda s: len(s.split()))
    for i in full_df['i'].unique():
        #         df.loc[df.loc[df['a'] == 1,'b'].index[1], 'b'] = 3
        min_num_words = full_df[full_df['i'] == i]['original_num_words'].min()
        full_df.loc[full_df[full_df['i'] == i].index, 'perturbed_text'] = (
            full_df.loc[full_df[full_df['i'] == i].index, 'perturbed_text'].map(
                lambda t: ' '.join(t.split()[:min_num_words])
            )
        )
    
    full_df['num_words'] = full_df['perturbed_text'].map(lambda s: len(s.split()))
    
    
    # This makes sure sure all documents with a given index have the same number of words.
    # (this makes sure all documents are indexed starting from the same number, using the
    # same dataset split, etc.)
    assert full_df.groupby('i')['num_words'].std().max() == 0.0

    return full_df


def get_exp_cache_path(exp_folder: str, exp_name: str, percentage: float) -> str:
    return os.path.join(exp_folder, f'{exp_name}__data_{percentage}.cache')


def words_from_text(s: str) -> List[str]:
    assert isinstance(s, str)
    return words_from_text_re.findall(s)


def count_words(s: str) -> int:
    assert isinstance(s, str)
    return len(words_from_text(s))


def count_masks(s: str) -> int:
    assert isinstance(s, str)
    return s.replace('[MASK]', '<mask>').count('<mask>')


def get_predictions_from_model(model_key: str, data: List[str], max_seq_length: int = 128) -> List[int]:
    """Loads model from `model_key` and predicts a (test/val) class for each datapoint in `data`.

    Returns list of the *index* of the top prediction. So everything was correct if they're all zeros.
    """
    checkpoint_path = model_paths_dict[model_key]
    model = CoordinateAscentModel.load_from_checkpoint(
        checkpoint_path
    )
    dm = WikipediaDataModule(
        document_model_name_or_path=model.document_model_name_or_path,
        profile_model_name_or_path=model.profile_model_name_or_path,
        dataset_name='wiki_bio',
        dataset_train_split='train[:256]',
        dataset_val_split='val[:256]',
        dataset_test_split='test[:256]',
        dataset_version='1.2.0',
        num_workers=num_cpus,
        train_batch_size=256,
        eval_batch_size=256,
        max_seq_length=128,
        sample_spans=False,
    )

    all_profile_embeddings = get_profile_embeddings(model_key=model_key, use_train_profiles=False).cuda()

    model.document_model.eval()
    model.document_model.cuda()
    model.document_embed.eval()
    model.document_embed.cuda()

    topk_values = []
    topk_idxs = []
    true_profile_idxs = []
    batch_size = 32
    i = 0
    pbar = tqdm.tqdm(total=len(data), leave=False, desc='Making predictions...')
    while i < len(data):
        ex = data[i:i+batch_size]
        test_batch = dm.document_tokenizer.batch_encode_plus(
            ex,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        test_batch = {
            f'perturbed_text__{k}': v for k,v in test_batch.items()
        }
        correct_idxs = (torch.arange(batch_size) + i).cuda()[:len(ex)]
        with torch.no_grad():
            document_embeddings = model.forward_document(batch=test_batch, document_type='perturbed_text')
            document_to_profile_logits = document_embeddings @ all_profile_embeddings.T
            document_to_profile_probs = document_to_profile_logits.softmax(dim=1)
            topk_10 = document_to_profile_probs.topk(10)
            topk_values.append(topk_10.values)
            topk_idxs.append(topk_10.indices)
            
            batch_true_profile_idxs = (
                (document_to_profile_logits.argsort(dim=1).flip(1) == correct_idxs[:, None]).nonzero()[:, 1]
            )
            true_profile_idxs.append(batch_true_profile_idxs)

        i += batch_size
        pbar.update(batch_size)
    
    pred_topk_values = torch.cat(topk_values, dim=0).cpu().tolist()
    pred_topk_idxs = torch.cat(topk_idxs, dim=0).cpu().tolist()
    true_profile_idxs = torch.cat(true_profile_idxs, dim=0).cpu().tolist()
    
    
    model.document_model.cpu()
    model.document_embed.cpu()
    model.cpu()
    return true_profile_idxs


def get_reidentified_data_at_masking_percentage_uncached(exp_folder: str, exp_name: str, percentage: float) -> List[Tuple[str, bool]]:
    # 1. Load all results
    all_results = pickle.load(open(os.path.join(exp_folder, f'{exp_name}_examples.p'), 'rb'))
    # 2. Get result within masking rate
    masked_inputs = []
    for results_list in all_results:
        most_masked_result = results_list[0]
        for r in results_list[1:]:
            num_masks = count_masks(r)
            num_words = count_words(r)
            if (num_masks / num_words) <= percentage:
                most_masked_result = r
        masked_inputs.append(most_masked_result)
    assert len(masked_inputs) == len(all_results)
    # 3. Reidentify results with each model
    all_predictions = []
    for model_key in REID_MODEL_KEYS: # tmp: do use REID_MODEL_KEYS the whole thing later.
        preds = get_predictions_from_model(model_key=model_key, data=masked_inputs)
        all_predictions.append(preds)
    df = pd.DataFrame(list(zip(*all_predictions)), columns=REID_MODEL_KEYS)
    input_was_reidentified =  (df == 0).apply(lambda row: row.values.any(), axis=1)
    return list(zip(masked_inputs, input_was_reidentified))


def get_experimental_results(exp_folder: str, exp_name: str, percentage: float, use_cache: bool) -> List[Tuple[str, bool]]:
    """Loads reidentified data results from a given experiment at a certain masking rate.

    Args:
        exp_folder (str): path to experiment folder
        exp_name (str): name of experiment
        percentage (float): percentage of words to mask (max)
        use_cache (bool): whether or not to cache/load cached results
    Returns List[Tuple[bool]]:
        Each masked sample alongside whether it was reidentified.
    """
    assert 0 <= percentage <= 1
    cache_path = get_exp_cache_path(exp_folder=exp_folder, exp_name=exp_name, percentage=percentage)
    print(f'cache_path = {cache_path}')
    if not (use_cache and os.path.exists(cache_path)):
        data = get_reidentified_data_at_masking_percentage_uncached(exp_folder=exp_folder, exp_name=exp_name, percentage=percentage)
        if use_cache:
            pickle.dump(data, open(cache_path, 'wb'))
    else:
        data = pickle.load(open(cache_path, 'rb'))
    return data


