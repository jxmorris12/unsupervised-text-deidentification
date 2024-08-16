#!/bin/bash

# Define the file paths
file_path="./thresholds.txt"

# Reading command line arguments

## Read the cuda_device_number from command line arguments
cuda_device_number=$1

# Check if the specified cuda_device_number is valid
if (( cuda_device_number < 0 )) || (( cuda_device_number > 1 )); then
  echo "Invalid cuda_device_number. Expecting it to be either 0 or 1.(Change code if other values has to be allowed)" >> "$nohup_file_path"
  exit 1
fi

## Read the start and end lines from command line arguments
start_line=$2
end_line=$3

## Read the name of the tool used for masking the training dataset
masking_tool=$4 # Possible inputs : 'ehr', 'deidentify', 'philter'

nohup_file_path="./nohupOuts/nohup_512_${masking_tool}_masked_cuda_device_number_${cuda_device_number}.out"

# Check if the arguments are valid numbers
if ! [[ "$start_line" =~ ^[0-9]+$ ]] || ! [[ "$end_line" =~ ^[0-9]+$ ]] || ! [[ "$cuda_device_number" =~ ^[0-9]+$  ]]; then
  echo "Invalid input: Start and end lines(1-indexed), and cuda_device_number must be numbers. Provide the values in that order." >> "$nohup_file_path"
  exit 1
fi

# Check if the file exists
if [[ ! -f "$file_path" ]]; then
  echo "File not found!" >> "$nohup_file_path"
  exit 1
fi

# Log the start time
echo "Script started at $(date)" >> "$nohup_file_path"

# Read the file line by line and store non-commented lines in an array
non_commented_lines=()
while IFS= read -r line; do
  # Skip lines that start with '#'
  if [[ $line == \#* ]]; then
    continue
  fi
  non_commented_lines+=("$line")
done < "$file_path"

# Get the total number of non-commented lines
total_lines=${#non_commented_lines[@]}
echo "Total number of non-commented lines in $file_path is $total_lines"  >> "$nohup_file_path"

# Echo the non-commented lines
echo "Non-commented lines:" >> "$nohup_file_path"
for line in "${non_commented_lines[@]}"; do
  echo "$line" >> "$nohup_file_path"
done

# Check if the specified range is valid
if (( start_line < 1 )) || (( end_line > total_lines )) || (( start_line > end_line )); then
  echo "Invalid range: Please provide a valid start and end line(1-indexed) within the total number of lines." >> "$nohup_file_path"
  exit 1
fi

# Adjust start_line to be zero-indexed for array slicing
start_index=$((start_line - 1))
length=$((end_line - start_index))
echo "Running training commands on cuda device number ${cuda_device_number} for thresholds : ${non_commented_lines[@]:start_index:length}" >> "$nohup_file_path"

# Process the specified range of lines
for (( i=start_line-1; i<end_line; i++ )); do
  line=${non_commented_lines[$i]}

  if [[ $masking_tool == ehr ]]; then
    local_data_path="/prj0124_gpu/akr4007/data/currently_relevant_data/ehr_data/full_csv_most_relevant_decoded_encoded_512_token_length_notes_per_person_threshold_${line}_ehr_masked.parquet"
  fi

  if [[ $masking_tool == deidentify ]]; then
    local_data_path="/prj0124_gpu/sln4001/full_csv_most_relevant_decoded_encoded_512_token_length_notes_per_person_masked_till_20700_deidentify_threshold_${line}.parquet"
  fi

  if [[ $masking_tool == philter ]]; then
    local_data_path="/prj0124_gpu/akr4007/data/currently_relevant_data/philter_data/full_csv_most_relevant_decoded_encoded_512_token_length_notes_per_person_masked_till_20700_philter_threshold_$(printf "%.8f" "$line").parquet"
  fi

  if [[ ! -f "$local_data_path" ]]; then
    echo "Dataset file not found!" >> "$local_data_path"
    exit 1
  fi

  # Form commands to run
  command_to_run="CUDA_VISIBLE_DEVICES=${cuda_device_number} python main.py --epochs 45 --batch_size 35 --max_seq_length 512 --word_dropout_ratio 0.0 --word_dropout_perc 0.0 --document_model_name roberta --profile_model_name tapas --local_data_path ${local_data_path}  --dataset_source parquet --dataset_train_split=train[:70%] --dataset_val_split=val[:15%] --learning_rate 1e-5 --num_validations_per_epoch 1 --loss coordinate_ascent --e 768 --label_smoothing 0.00 --wandb_project_name deid_on_weill --wandb_entity deidentification --precision 16"
  
  # Run the commands and redirect their output to nohup file
  echo "Command Running: $command_to_run" >> "$nohup_file_path"
  eval $command_to_run >> "$nohup_file_path" 2>&1
  echo "Finished command: $command_to_run" >> "$nohup_file_path"
done

# Log the end time
echo "Script finished at $(date)" >> "$nohup_file_path"
