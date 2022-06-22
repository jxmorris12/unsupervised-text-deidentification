import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

import argparse
import os
import re

import pandas as pd

from model_cfg import model_paths_dict
from generate_masked_samples import WikiDataset, main


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Generate adversarially-masked examples for a model, continued off '
            'of previous set of adversarial examples.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--k', type=int, default=1,
        help='top-K classes for adversarial goal function'
    )
    parser.add_argument('csv_path', type=str,
        help='CSV to use as starting point for adversarial redaction generation'
    )
    parser.add_argument('--model', '--model_key', type=str, default='model_3_1',
        help='model str name (see model_cfg for more info)',
        choices=model_paths_dict.keys()
    )
    parser.add_argument('--beam_width', '--b', type=int, default=1,
        help='beam width for beam search'
    )
    parser.add_argument('--use_type_swap', action='store_true',
        help=(
            'whether to swap words by type instead of token '
            '(i.e. mask all instances of the same word together'
        ),
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    assert args.csv_path.endswith('.csv')

    n = int(re.search(r'.+__n_(\d+).*.csv', args.csv_path).group(1))
    use_train_profiles = 'with_train' in args.csv_path
    out_folder_path = os.path.join(
        args.csv_path.replace('.csv', ''), 'sequential'
        # model_key will be added in main()
    )
    os.makedirs(out_folder_path, exist_ok=True)

    adv_dataset = pd.read_csv(args.csv_path)

    main(
        k=args.k,
        n=n,
        num_examples_offset=0,
        beam_width=args.beam_width,
        adv_dataset=adv_dataset,
        model_key=args.model,
        use_type_swap=args.use_type_swap,
        use_train_profiles=use_train_profiles,
        out_folder_path=out_folder_path,
    )
