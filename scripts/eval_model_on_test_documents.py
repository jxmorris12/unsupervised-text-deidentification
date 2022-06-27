import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

from datamodule import WikipediaDataModule
from model import AbstractModel, CoordinateAscentModel
from utils import get_profile_embeddings_by_model_key

import collections
import argparse
import os

import torch
import transformers
from tqdm import tqdm


from model_cfg import model_paths_dict


num_cpus = len(os.sched_getaffinity(0))


def get_profile_embeddings(model_key: str):
    profile_embeddings = get_profile_embeddings_by_model_key(model_key=model_key)

    print("concatenating train, val, and test profile embeddings")
    all_profile_embeddings = torch.cat(
        (profile_embeddings['test'], profile_embeddings['val'], profile_embeddings['train']), dim=0
    )

    print("all_profile_embeddings:", all_profile_embeddings.shape)
    return all_profile_embeddings


def get_output_folder_by_model_key(model_key: str) -> str:
    current_folder = os.path.dirname(os.path.abspath(__file__))
    base_folder = os.path.join(current_folder, os.pardir)
    return os.path.normpath(
        os.path.join(base_folder, 'eval', model_key)
    )

def main(document_type: str, model_key: str):
    checkpoint_path = model_paths_dict[model_key]
    assert isinstance(checkpoint_path, str), f"invalid checkpoint_path {checkpoint_path} for {model_key}"
    print(f"running attack on {model_key} loaded from {checkpoint_path}")
    model = CoordinateAscentModel.load_from_checkpoint(
        checkpoint_path
    )

    print(f"loading data with {num_cpus} CPUs")
    dm = WikipediaDataModule(
        document_model_name_or_path=model.document_model_name_or_path,
        profile_model_name_or_path=model.profile_model_name_or_path,
        dataset_name='wiki_bio',
        dataset_train_split='train[:256]',
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

    all_profile_embeddings = get_profile_embeddings(model_key=model_key).cuda()

    model.document_model.eval()
    model.document_model.cuda()
    model.document_embed.eval()
    model.document_embed.cuda()

    total = 0
    total_correct_by_k = collections.defaultdict(lambda: 0)
    k_values = [1, 10, 100, 1000]

    for test_batch in tqdm(dm.test_dataloader(), desc=f'Evaluating test documents of type {args.document_type}', colour='yellow'):
        document_idxs = test_batch['text_key_id'].cuda()
        # Get probs
        with torch.no_grad():
            document_embeddings = model.forward_document(batch=test_batch, document_type=args.document_type)
            document_to_profile_logits = document_embeddings @ all_profile_embeddings.T
            # document_to_profile_probs = torch.nn.functional.softmax(
            #     document_to_profile_logits, dim=-1
            # )
        # Check statistics
        total += len(document_to_profile_logits)
        for k in k_values:
            topk_correct = (
                document_to_profile_logits.topk(k=k, dim=1)
                    .indices
                    .eq(document_idxs[:, None])
                    .any(dim=1)
                    .sum()
            )
            total_correct_by_k[k] += topk_correct
        break
        
    
    # Print statistics
    output_folder = get_output_folder_by_model_key(model_key=model_key)
    os.makedirs(output_folder, exist_ok=True)
    log_file = open(os.path.join(output_folder, f'test_{document_type}.txt'), 'w')

    print('*** Finished eval ****')
    log_file.write(f'**** Evaluated on {total} test examples of type {document_type} ****\n')
    for k in k_values:
        acc = total_correct_by_k[k] / total
        acc_str = f'Top-{k} accuracy = {acc*100.0:.2f}'
        print(acc_str)
        log_file.write(acc_str + '\n')
    log_file.close()
        
    

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluates model accuracy using all profile embeddings (train, test, and val).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--document_type', type=str, default='document',
        help='document type for evaluation',
        choices=["document", "document_redact_ner_bert", "document_redact_lexical"]
    )
    parser.add_argument('--model', '--model_key', type=str, default='model_5',
        help='model str name (see model_cfg for more info)',
        choices=model_paths_dict.keys(),
        required=True
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(document_type=args.document_type, model_key=args.model)
