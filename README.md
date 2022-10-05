<h1 align="center"> Unsupervised Text Deidentification </h1>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.8-blue">
</p>  

<img src="https://github.com/jxmorris12/unsupervised-deid/blob/master/overview.svg">


<b>Official code for 2022 paper, "Unsupervised Text Deidentification".</b> Our method, NN DeID can anonymize text without any prior notion of what constitutes personal information (i.e. without guidelines or labeled data). 

This repository all the code for training reidentification models and deidentifying data. The main tools & frameworks used are [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://github.com/Lightning-AI/lightning) (for model-training), and [TextAttack](https://github.com/QData/TextAttack) (for deidentification via greedy inference). The main dataset used is the [wikibio dataset](https://rlebret.github.io/wikipedia-biography-dataset/), [loaded through HuggingFace datasets](https://huggingface.co/datasets/wiki_bio). The models that comprise the biencoder are [TAPAS](https://github.com/google-research/tapas) and [RoBERTa](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/).


## folder structure

**Model-training:**
`main.py` is the root file for model-training. All experiments can be launched by invoking this file with certain arguments, i.e. by running `python main.py --args`.

**Deidentification**:
All the text deidentification happens through `scripts/deidentify.py`. To launch all the deidentification experiments at once, run `python scripts/launch_experiments.py`. Various pieces of the deidentification logic, are situated in `deidentification/*.py`.

## models

Modeling code is in `models/*.py`. Contains code for training models using contrastive loss (biencoder), contrastive loss for a cross-encoder, and "coordinate ascent" via freezing embeddings for a bienocder.

Here's a sample command for training the models:

```bash
python main.py --epochs 300 --batch_size 64 --max_seq_length 128 --word_dropout_ratio 0.8 --word_dropout_perc -1.0 --document_model_name roberta --profile_model_name tapas --dataset_name "wiki_bio" --dataset_train_split="train[:100%]" --learning_rate 1e-4 --num_validations_per_epoch 1 --loss coordinate_ascent --e 3072 --label_smoothing 0.01
```

## dataloading

The dataloading code is all routed through `datamodule.py`, which loads a dataset of documents & profiles and preprocesses it, including by a number of baseline deidentification techniques which are used for computing model performance on deidentified data throughout training. 

Other notable files include `masking_tokenizing_dataset.py`, which implements a custom PyTorch dataset that tokenizes data and optionally randomly masks words, and `masking_span_sampler.py`, which contains options for randomly sampling sentences from text (not used in the paper) and randomly masking text using a number of different strategies.

## scripts

There are a bunch of useful scripts in `scripts/`, including code for generating nearest-neighbors, uploading models to the HuggingFace hub, doing probing on embeddings to measure retainment of profile information such as birth dates and months, and deidentifying data using TextAttack.


### Citation

If this package is useful for you, please cite the following!

```r
@article{morris2022deid,
  year = {2022},
}
```
