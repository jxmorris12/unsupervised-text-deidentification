"""Loads data for a single epoch and loops over it. Intended for use with
    torch.utils.bottleneck.
"""

import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

import numpy as np
import torch
import tqdm

from dataloader import WikipediaDataModule
from model import CoordinateAscentModel
from model_cfg import model_paths_dict

import os

# num_cpus = len(os.sched_getaffinity(0))
# num_workers = num_cpus
num_workers = 0

def main():
    print(f"Initializing with {num_workers} workers")
    dm = WikipediaDataModule(
        document_model_name_or_path="roberta-base",
        profile_model_name_or_path="google/tapas-base",
        max_seq_length=128,
        dataset_name='wiki_bio',
        dataset_train_split='train[:1%]',
        dataset_val_split='val[:20%]',
        dataset_version='1.2.0',
        word_dropout_ratio=0.0,
        word_dropout_perc=0.0,
        num_workers=num_workers,
        train_batch_size=512,
        eval_batch_size=512
    )
    dm.setup("fit")

    # model that was trained at the link given above, gets >99% validation accuracy,
    # and is trained with word dropout!
    checkpoint_path = model_paths_dict["model_5"]

    model = CoordinateAscentModel.load_from_checkpoint(
        checkpoint_path,
        document_model_name_or_path="roberta-base",
        profile_model_name_or_path="google/tapas-base",
        learning_rate=1e-5,
        pretrained_profile_encoder=False,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=1,
        train_batch_size=1,
        num_workers=8,
        gradient_clip_val=10.0,
    )

    model.profile_model.cuda()
    model.profile_model.eval()
    model.val_profile_embeddings = np.zeros((len(dm.val_dataset), model.profile_embedding_dim))
    for val_batch in tqdm.tqdm(dm.val_dataloader()[0], desc="Precomputing val embeddings", colour="yellow", leave=False):
        with torch.no_grad():
            profile_embeddings = model.forward_profile(batch=val_batch)
        model.val_profile_embeddings[val_batch["text_key_id"]] = profile_embeddings.cpu().numpy()
    model.val_profile_embeddings = torch.tensor(model.val_profile_embeddings, dtype=torch.float32)

    print("done!")

if __name__ == '__main__':
    main()





""" 
** Original (with on-the-fly profile tokenization) **
 Ordered by: internal time
   List reduced from 17358 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       29   31.391    1.082   31.391    1.082 {method 'cpu' of 'torch._C._TensorBase' objects}
 17280696   20.632    0.000   36.955    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:1378(_get_table_values)
    43698   18.956    0.000   18.956    0.000 {method 'encode_batch' of 'tokenizers.Tokenizer' objects}
 37063982   11.107    0.000   11.107    0.000 {method 'match' of 're.Pattern' objects}
50553617/50553609    9.462    0.000   13.017    0.000 {built-in method builtins.isinstance}
  1843084    7.910    0.000   35.434    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:2401(_parse_date)
  2901839    6.910    0.000    9.404    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:2436(get_all_spans)
 29589775    6.125    0.000   10.031    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:123(_is_inner_wordpiece)
   195352    5.796    0.000   54.634    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:2475(parse_text)
 40419324    5.716    0.000    5.716    0.000 {method 'startswith' of 'str' objects}
   382075    5.442    0.000   14.466    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:2128(_clean_text)
   904751    5.032    0.000   12.788    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:2069(_run_split_on_punc)
  2771388    4.909    0.000   10.768    0.000 {method 'sub' of 're.Pattern' objects}
   893919    4.865    0.000   11.656    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/_strptime.py:309(_strptime)
  1210459    4.672    0.000    6.911    0.000 /home/jxm3/.conda/envs/torch/lib/python3.9/site-packages/transformers/models/tapas/tokenization_tapas.py:2151(tokenize)

"""