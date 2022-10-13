import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

from typing import Iterable

import collections
import os
import glob
import re

import numpy as np
import pandas as pd
import tqdm

from utils.analysis import get_experimental_results
from utils.analysis import load_baselines_csv


def get_experiments() -> Iterable[str]:
    """Gets path to finished experiments in the experiments/ folder."""
    exp_files = glob.glob('./experiments/*/*.p')
    for filename in exp_files:
        filename_re = re.search(r'(experiments/.+/)(.+)_examples.p', filename)
        folder_name, exp_name = filename_re.group(1), filename_re.group(2)
        yield folder_name, exp_name

def main():
    experiments = list(get_experiments())[::-1]
    print(f'Testing {len(experiments)} experiments: {experiments}')

    baselines_csv = load_baselines_csv(max_num_samples=10)
    baselines_csv.head()

    all_results = []

    base_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir
    )
    for exp_folder, exp_name in tqdm.tqdm(experiments):
        for percentage in np.arange(0.00, 1.05, 0.05)[::-1]:
            print('-->', exp_folder, exp_name, percentage)
            r = get_experimental_results(
                exp_folder=os.path.join(base_folder, exp_folder), exp_name=exp_name,
                percentage=percentage, use_cache=True
            )   
            all_results.append((exp_folder, exp_name, percentage, r))

    # unroll into df
    all_results_expanded = []
    for result in all_results:
        exp_folder, exp_name, percentage, r_list = result
        for text, was_reidentified in r_list:
            all_results_expanded.append((exp_folder, exp_name, percentage, text, was_reidentified))

    df = pd.DataFrame(all_results_expanded, columns=[
        'experiment_folder', 'experiment_name', 'masking_percentage', 'text', 'was_reidentified'
    ])

if __name__ == '__main__':
    main()
