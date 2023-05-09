#!/bin/bash
# trains baseline model (no masking.)
python main.py --epochs 100 --batch_size 3 --max_seq_length 128 --document_model_name roberta --profile_model_name tapas --learning_rate 1e-5 --num_validations_per_epoch 4 --loss coordinate_ascent --e 768 --label_smoothing 0.0 --datamodule dalio --word_dropout_ratio 0.0 --word_dropout_perc -1.0