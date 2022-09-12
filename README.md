# unsupervised-deidentification

This repository contains the code for our 2022 paper, "Unsupervised Text Deidentification". It contains all the code for training reidentification models and deidentifying data. The main tools & frameworks used are PyTorch, PyTorch Lightning (for model-training), and TextAttack (for deidentification via greedy inference).

`main.py` is the root

## models

Modeling stuff is in `models/*.py`. Contains code for training models using contrastive loss (biencoder), contrastive loss for a cross-encoder, and "coordinate ascent" via freezing embeddings for a bienocder.

Here's a sample command for training the models:

```bash
python main.py --epochs 300 --batch_size 64 --max_seq_length 128 --word_dropout_ratio 0.8 --word_dropout_perc -1.0 --document_model_name roberta --profile_model_name tapas --dataset_name "wiki_bio" --dataset_train_split="train[:100%]" --learning_rate 1e-4 --num_validations_per_epoch 1 --loss coordinate_ascent --e 3072 --label_smoothing 0.01
```

## dataloading

The dataloading code is all routed through `datamodule.py`, which loads a dataset of documents & profiles and preprocesses it, including by a number of baseline deidentification techniques which are used for computing model performance on deidentified data throughout training. 

Other notable files include `masking_tokenizing_dataset.py`, which implements a custom PyTorch dataset that tokenizes data and optionally randomly masks words, and `masking_span_sampler.py`, which contains options for randomly sampling sentences from text (not used in the paper) and randomly masking text using a number of different strategies.

## scripts

There are a bunch of useful scripts in `scripts/`, including code for generating nearest-neighbors, uploading models to the HuggingFace hub, doing probing on embeddings to measure retainment of profile information such as birth dates and months, and deidentifying data using TextAttack.
