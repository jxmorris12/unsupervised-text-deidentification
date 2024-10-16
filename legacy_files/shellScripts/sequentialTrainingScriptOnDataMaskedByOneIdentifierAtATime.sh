#!/bin/bash

# Define the directory where the files are located
directory='<See above comment>'

# Loop through the files in the directory
for file in "$directory"/*masked.parquet; do
  # Check if the file exists (to handle cases with no matching files)
  if [ -f "$file" ]; then
    shellCommand="python main.py --epochs 69 --batch_size 30 --max_seq_length 512 --word_dropout_ratio 0.0 --word_dropout_perc 0.0 --document_model_name roberta --profile_model_name tapas --local_data_path $file --dataset_source parquet --dataset_train_split=train[:70%] --dataset_val_split=val[:15%] --learning_rate 1e-5 --num_validations_per_epoch 1 --loss coordinate_ascent --e 768 --label_smoothing 0.00 --wandb_project_name deid_on_weill --wandb_entity deidentification --checkpoint_path <SEE README TO KNOW WHAT TO PUT HERE>"

    echo "Running command : $shellCommand"
    
    $shellCommand	
    
  fi
done

