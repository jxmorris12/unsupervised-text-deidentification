import pandas as pd
import datasets
import argparse
import os
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(description='Get arguments for the program.')
    parser.add_argument('--par_directory', help="Give the path of the directory which contains the data files.", type=str, required=True)
    parser.add_argument('--output_file_name', help='Give the name of the output file. It will be stored in the par_directory itself.', type=str, default='text_masked_percentages.npy')
    return parser.parse_args()

def generate_percentages(args):
    par_dir = args.par_directory
    text_masked_percentages = np.array([])
    for file_name in os.listdir(par_dir):
        if '.parquet' in file_name:
            print(f"Processing : {file_name}\n")
            d = datasets.Dataset.from_parquet(os.path.join(par_dir, file_name))
            df = pd.DataFrame(d)
            df_notes_without_whitespaces = df['note_text\n'].apply(lambda x : x.replace(" ", ""))
            percentage = np.mean(df_notes_without_whitespaces.apply(lambda x : x.count('*') / len(x))) * 100
            text_masked_percentages = np.append(text_masked_percentages, np.array([file_name, '{:.2f}'.format(percentage)]))
            print(f"Percentage of text masked : {'{:.2f}'.format(percentage)}%\n")
    print(text_masked_percentages)
    save_path = os.path.join(par_dir, args.output_file_name)
    np.save(save_path, text_masked_percentages)
    print(f'Percentages saved at : {save_path}')
    
        

if __name__ == '__main__':
    args = get_arguments()
    generate_percentages(args)
