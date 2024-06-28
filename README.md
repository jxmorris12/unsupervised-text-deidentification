<h1 align="center"> Unsupervised Text Deidentification </h1>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.8-blue">
</p>  

<img src="https://github.com/jxmorris12/unsupervised-deid/blob/master/overview.svg">


<b>Official code for 2022 paper, "Unsupervised Text Deidentification".</b> Our method, NN DeID can anonymize text without any prior notion of what constitutes personal information (i.e. without guidelines or labeled data). 

This repository all the code for training reidentification models and deidentifying data. The main tools & frameworks used are [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://github.com/Lightning-AI/lightning) (for model-training), and [TextAttack](https://github.com/QData/TextAttack) (for deidentification via greedy inference). The main dataset used is the [wikibio dataset](https://rlebret.github.io/wikipedia-biography-dataset/), [loaded through HuggingFace datasets](https://huggingface.co/datasets/wiki_bio). The models that comprise the biencoder are [TAPAS](https://github.com/google-research/tapas) and [RoBERTa](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/) (or [PMLM](https://arxiv.org/abs/2004.11579)).


## folder structure

**Model-training:**
`main.py` is the root file for model-training. All experiments can be launched by invoking this file with certain arguments, i.e. by running `python main.py --args`.

**Deidentification**:
All the text deidentification happens through `scripts/deidentify.py`. To launch all the deidentification experiments at once, run `python scripts/launch_experiments.py`. Various pieces of the deidentification logic, are situated in `deidentification/*.py`.

## models

Modeling code is in `models/*.py`. Contains code for training models using contrastive loss (biencoder), contrastive loss for a cross-encoder, and "coordinate ascent" via freezing embeddings for a bienocder.

Here are sample commands for training the models:

- 
```bash
python main.py --epochs 300 --batch_size 64 --max_seq_length 128 --word_dropout_ratio 0.8 --word_dropout_perc -1.0 --document_model_name roberta --profile_model_name tapas --dataset_name "wiki_bio" --dataset_train_split="train[:100%]" --learning_rate 1e-4 --num_validations_per_epoch 1 --loss coordinate_ascent --e 3072 --label_smoothing 0.01
```

-
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --epochs 5000 --batch_size 30 --max_seq_length 512 --word_dropout_ratio 0.0 --word_dropout_perc 0.0 --document_model_name roberta --profile_model_name tapas --local_data_path /prj0124_gpu/akr4007/data/currently_relevant_data/full_csv_most_relevant_decoded_encoded_512_token_length_notes_per_person_ehr_masked.parquet --dataset_source parquet --dataset_train_split=train[:70%] --dataset_val_split=val[:15%] --learning_rate 1e-5 --num_validations_per_epoch 1 --loss coordinate_ascent --e 768 --label_smoothing 0.00 --wandb_project_name deid_on_weill --wandb_entity deidentification --checkpoint_path /prj0124_gpu/akr4007/unsupervised-text-deidentification/deid_on_weill/07hlg09a/checkpoints/epoch=14-step=7470.ckpt
```
-- 'CUDA_VISIBLE_DEVICE' determins which cuda(gpu) device to use.
-- '--local_data_path' determines the local path of the dataset, if any.
-- '--wandb_project_name' and '--wandb_entity' : wandb specific arguments, to log on wandb(https://wandb.ai/)
-- '--checkpoint_path' arguments gives a checkpoint of previously trained model, from where the model can continue training.

## dataloading

The dataloading code is all routed through `datamodule.py`, which loads a dataset of documents & profiles and preprocesses it, including by a number of baseline deidentification techniques which are used for computing model performance on deidentified data throughout training. 

Other notable files include `masking_tokenizing_dataset.py`, which implements a custom PyTorch dataset that tokenizes data and optionally randomly masks words, and `masking_span_sampler.py`, which contains options for randomly sampling sentences from text (not used in the paper) and randomly masking text using a number of different strategies.

## scripts

There are a bunch of useful scripts in `scripts/`, including code for generating nearest-neighbors, uploading models to the HuggingFace hub, doing probing on embeddings to measure retainment of profile information such as birth dates and months, and deidentifying data using TextAttack.

## models

The models used for reidentification and identification are available through HuggingFace. All models were trained with this codebase and the hyperparameters specified in the paper.

- [`jxm/wikibio_roberta_tapas_vanilla`](https://huggingface.co/jxm/wikibio_roberta_tapas_vanilla): vanilla biencoder, trained with RoBERTa document encoder and TAPAS profile encoder on documents without any masking
- [`jxm/wikibio_roberta_tapas`](https://huggingface.co/jxm/wikibio_roberta_tapas): vanilla biencoder, trained with RoBERTa document encoder and TAPAS profile encoder on documents with uniform random masking
- [`jxm/wikibio_roberta_tapas_idf`](https://huggingface.co/jxm/wikibio_roberta_tapas_idf): vanilla biencoder, trained with RoBERTa document encoder and TAPAS profile encoder on documents with IDF-weighted random masking

- [`jxm/wikibio_pmlm_tapas`](https://huggingface.co/jxm/wikibio_pmlm_tapas): vanilla biencoder, trained with PMLM document encoder and TAPAS profile encoder on documents with uniform random masking
- [`jxm/wikibio_pmlm_tapas_idf`](https://huggingface.co/jxm/wikibio_pmlm_tapas_idf): vanilla biencoder, trained with PMLM document encoder and TAPAS profile encoder on documents with IDF-weighted random masking

- [`jxm/wikibio_roberta_roberta`](https://huggingface.co/jxm/wikibio_roberta_roberta): vanilla biencoder, trained with RoBERTa document encoder and RoBERTa profile encoder on documents with uniform random masking
- [`jxm/wikibio_roberta_roberta_idf`](https://huggingface.co/jxm/wikibio_roberta_roberta_idf): vanilla biencoder, trained with RoBERTa document encoder and RoBERTa profile encoder on documents with IDF-weighted random masking


### example model-running command

Here's an example of how you'd load a pre-trained model, in addition to the WikiBio dataloader. First you have to clone the repository to download the model:
```bash
!git clone https://huggingface.co/jxm/wikibio_roberta_roberta
```

Then run the following code.

```python

from datamodule import WikipediaDataModule
from model import CoordinateAscentModel


num_cpus = len(os.sched_getaffinity(0))

checkpoint_path = "/content/unsupervised-deid/wikibio_roberta_roberta/model.ckpt"
model = CoordinateAscentModel.load_from_checkpoint(checkpoint_path)
dm = WikipediaDataModule(
    document_model_name_or_path=model.document_model_name_or_path,
    profile_model_name_or_path=model.profile_model_name_or_path,
    dataset_name='wiki_bio',
    dataset_train_split='train[:10%]',
    dataset_val_split='val[:20%]',
    dataset_version='1.2.0',
    num_workers=1,
    train_batch_size=64,
    eval_batch_size=64,
    max_seq_length=128,
    sample_spans=False,
)
```

### example model-training command

This is an example of how to train the RoBERTa-TAPAS example from above (uniform random masking). All of the other models can be trained with a very similar command.

```bash
python main.py --epochs 60 --batch_size 128 --max_seq_length 128 --word_dropout_ratio 1.0 --word_dropout_perc -1.0 --document_model_name roberta --profile_model_name tapas --dataset_train_split="train[:100%]" --learning_rate 1e-4 --num_validations_per_epoch 1 --loss coordinate_ascent --e 3072 --label_smoothing 0.01
```

## analysis example

## Troubleshooting

#### pyarrow error : Cannot import datasets - ValueError: pyarrow.lib.IpcWriteOptions size changed, may indicate binary incompatibility

Solution : See this : https://github.com/huggingface/datasets/issues/5923

#### OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.

Solution : Install "en_core_web_sm" using the following command
```python -m spacy download en_core_web_sm```

[Similar command might work for other models as well]

#### ERROR Error while calling W&B API: project not found (<Response [404]>)

Solution : Make sure you have a project on wandb that you want to use for this experiment. Add that project name to the command line arguments using the following flag:
```--wandb_project_name <project_name>```

Also, make sure you provide an eentity which can access that project. You can do this by using the following flag:
```--wandb_entity <entity>```

Also, make sure the wandb API_key provided corresponds to the user of which the project is used.


#### Debugging issue : pdb debugger too slow

Solution : Please type this command in the terminal before rerunning the program - 'export CUDA_VISIBLE_DEVICES=0'

### Citation

If this package is useful for you, please cite the following!

```
@misc{https://doi.org/10.48550/arxiv.2210.11528,
  doi = {10.48550/ARXIV.2210.11528},
  url = {https://arxiv.org/abs/2210.11528},
  author = {Morris, John X. and Chiu, Justin T. and Zabih, Ramin and Rush, Alexander M.},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Unsupervised Text Deidentification},
  publisher = {arXiv},
  year = {2022}, 
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
