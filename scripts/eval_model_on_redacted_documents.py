import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

from datamodule import WikipediaDataModule
from model import AbstractModel, CoordinateAscentModel
from utils import get_profile_embeddings_by_model_key

import argparse
import collections
import glob
import os
import re

import pandas as pd
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
    adv_csvs_folder = os.path.normpath(
        os.path.join(
            os.path.abspath(__file__), os.pardir, os.pardir, 'adv_csvs_full_2'
        )
    )
    return os.path.join(adv_csvs_folder, model_key)

def load_adv_csv(dm: WikipediaDataModule) -> pd.DataFrame:
    # Load all the stuff
    adv_df = None
    for model_name in ['model_3_1', 'model_3_2', 'model_3_3', 'model_3_4']:
        adv_csvs_folder = os.path.normpath(
            os.path.join(
                os.path.abspath(__file__), os.pardir, os.pardir, 'adv_csvs_full_2'
            )
        )
        print('adv_csvs_folder', adv_csvs_folder)
        csv_filenames = glob.glob(
            os.path.join(
                adv_csvs_folder,
                f'{model_name}*/results__b_1__k_*__n_1000.csv'
            )
        )
        print(model_name, csv_filenames)
        for filename in csv_filenames:
            df = pd.read_csv(filename)
            df['model_name'] = re.search(r'adv_csvs_full_2/(model_\d.*)/.+.csv', filename).group(1)
            df['k'] = re.search(r'adv_csvs_full_2/.+/.+__k_(\d+)__.+.csv', filename).group(1)
            df['i'] = df.index

            df = df[df['result_type'] == 'Successful']

            mini_df = df[['perturbed_text', 'model_name', 'i', 'k']]
            
            if adv_df is None:
                adv_df = mini_df
            else:
                adv_df = pd.concat((adv_df, mini_df), axis=0)
    
    # Load baseline redacted data
    mini_val_dataset = dm.test_dataset[:1000]
    ner_df = pd.DataFrame(
        columns=['perturbed_text'],
        data=mini_val_dataset['document_redact_ner_bert']
    )
    ner_df['model_name'] = 'named_entity'
    ner_df['i'] = ner_df.index
        
    lex_df = pd.DataFrame(
        columns=['perturbed_text'],
        data=mini_val_dataset['document_redact_lexical']
    )
    lex_df['model_name'] = 'lexical'
    lex_df['i'] = lex_df.index

    # Combine both adversarial and baseline redacted data
    baseline_df = pd.concat((lex_df, ner_df), axis=0)
    baseline_df['k'] = 0
    full_df = pd.concat((adv_df, baseline_df), axis=0)

    # Put newlines back
    full_df['perturbed_text'] = full_df['perturbed_text'].apply(lambda s: s.replace('<SPLIT>', '\n'))

    # Standardize mask tokens
    full_df['perturbed_text'] = full_df['perturbed_text'].apply(lambda s: s.replace('[MASK]', dm.mask_token))
    full_df['perturbed_text'] = full_df['perturbed_text'].apply(lambda s: s.replace('<mask>', dm.mask_token))

    return full_df


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
        dataset_val_split='val[:256]',
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

    adv_csv = load_adv_csv(dm=dm)

    total = 0
    total_correct_by_k = collections.defaultdict(lambda: 0)
    k_values = [1, 10, 100, 1000]

    pred_idxs = []
    batch_size = 256
    i = 0
    while i < len(adv_csv):
        ex = adv_csv.iloc[i:i+batch_size]
        test_batch = dm.document_tokenizer.batch_encode_plus(
            ex['perturbed_text'].tolist(),
            max_length=dm.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        test_batch = {
            f'perturbed_text__{k}': v for k,v in test_batch.items()
        }
        with torch.no_grad():
            document_embeddings = model.forward_document(batch=test_batch, document_type='perturbed_text')
            document_to_profile_logits = document_embeddings @ all_profile_embeddings.T
            document_to_profile_idxs = document_to_profile_logits.argmax(dim=1)
            pred_idxs.append(document_to_profile_idxs.cpu())

        i += batch_size
    
    adv_csv['model_pred_idxs'] = torch.cat(pred_idxs)
    adv_csv['is_correct'] = (adv_csv['i'] == adv_csv['model_pred_idxs'])
    print(adv_csv.groupby(['model_name', 'k'])['is_correct'].mean())

    # Count masks
    if args.count_masks:
        def count_percent_masks(s: str) -> int:
            return s.count('<mask>') / len(s.split(' '))


        def truncate_text(text: str, max_length=128) -> str:
            input_ids = dm.document_tokenizer(text, truncation=True, max_length=128)['input_ids']
            reconstructed_text = (
                dm.document_tokenizer
                    .decode(input_ids)
                    .replace('<mask>', ' <mask> ')
                    .replace('  <mask>', ' <mask>')
                    .replace('<mask>  ', '<mask> ')
                    .replace('<s>', '')
                    .replace('</s>', '')
                    .strip()
            )
            return reconstructed_text

        adv_csv['perturbed_text_truncated'] = adv_csv['perturbed_text'].apply(truncate_text)
        adv_csv['percent_masks'] = adv_csv.apply(lambda s: count_percent_masks(s['perturbed_text_truncated']), axis=1)
        print(adv_csv.groupby(['model_name', 'k']).mean()['percent_masks'])
        
    
    # Print statistics
    output_folder = get_output_folder_by_model_key(model_key=model_key)
    os.makedirs(output_folder, exist_ok=True)
    log_file_name = os.path.join(output_folder, 'adv_2_eval.txt')
    log_file = open(log_file_name, 'w')
    

    print('*** Finished adv eval ****')
    log_file.write(f'**** Evaluated on {total} test examples of type {document_type} ****\n')
    log_file.write('\n\n\n')
    log_file.write(str(adv_csv.groupby(['model_name', 'k'])['is_correct'].mean()))
    log_file.write('\n\n\n')
    log_file.write(str(adv_csv.groupby(['model_name', 'k']).sum()))
    log_file.close()

    print("wrote output to", log_file_name)
        

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
    parser.add_argument('--count_masks', action='store_true',
        help='whether to output statistics about the number of masks in the data'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(document_type=args.document_type, model_key=args.model)
