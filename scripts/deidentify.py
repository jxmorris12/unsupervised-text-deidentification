import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

from typing import Dict, List, Optional, Set, Tuple

from datamodule import WikipediaDataModule
from deidentification.constraints import CertainWordsModification
from deidentification.dataset_wrappers import WikiDatasetWrapper
from deidentification.goal_functions import ChangeClassificationToBelowTopKClasses
from deidentification.loggers import CustomCSVLogger
from deidentification.model_wrappers import CrossEncoderModelWrapper, MainModelWrapper
from deidentification.transformations import WordSwapSingleWordToken, WordSwapSingleWordType
from model import ContrastiveCrossAttentionModel, CoordinateAscentModel
from model_cfg import model_paths_dict
from utils import get_profile_embeddings

import argparse
import json
import os

import textattack

from textattack import Attack, Attacker, AttackArgs
from textattack.constraints.pre_transformation import (
    MaxModificationRate, RepeatModification, MaxWordIndexModification, StopwordModification
)


num_cpus = len(os.sched_getaffinity(0))


def main(
        n: int,
        num_examples_offset: int,
        k: int,
        beam_width: int, 
        model_key: str,
        use_train_profiles: bool,
        use_type_swap: bool,
        eps: Optional[float] = None,
        min_percent_words: Optional[float] = None,
        adv_dataset: Optional = None,
        out_folder_path: str = 'adv_csvs_full_8',
        out_file_path: str = None,
        ignore_stopwords: bool = False,
        fuzzy_ratio: float = 0.95,
        max_idf_goal: Optional[float] = None,
        table_score: float = 0.0,
        min_idf_weighting: Optional[float] = None,
        do_reid: bool = False,
        no_model: bool = False,
    ):
    """Deidentifies data with these experimental parameters.

    Args:
        n (int): number of samples to deidentify
        num_examples_offset (int): starting index for identifying
        k (int): number to use if using top-k criterion for deidentification objective function
        beam_width (int): Beam width for deid beam search 
        model_key (str): identifying key of model to use for deidentification
        use_train_profiles (bool): whether to include train profiles in 'negative examples' for deid
        use_type_swap (bool): whether to swap words by token or type 
        eps (float): Epsilon if using top-eps criterion instead of top-k
        min_percent_words (float): minimum percent words to swap if using this criteria
        adv_dataset (Optional dataset): optionally a dataset to attack instead of the default wiki one
        out_folder_path (str): path to output folder
        out_file_path (str): name of output file within folder (will add .csv, .p, _args.json)
        ignore_stopwords (bool): whether to not deidentify stopwords
            [not used in paper]
        fuzzy_ratio (float): amount of fuzzy-matching to do for table scoring 
            [not used in paper]
        max_idf_goal (float): amount to use for IDF goal-scoring criterion
        table_score (float): weight of table-scoring
        min_idf_weighting (float): min threshold on IDF values weighted
            [not used in paper]
        do_reid (bool): Whether to load up a model and estimate the reidentifiability of   
            outputted samples (does not use ensemble, so gives only lower bound)
        no_model (bool): Whether we're not using an encoder and just using IDF weighting for example
    """
    saved_args = dict(locals())
    checkpoint_path = model_paths_dict[model_key]
    assert isinstance(checkpoint_path, str), f"invalid checkpoint_path {checkpoint_path} for {model_key}"
    print(f"running attack on {model_key} loaded from {checkpoint_path}")

    is_cross_encoder = ('cross_encoder' in model_key)

    if is_cross_encoder:
        model = ContrastiveCrossAttentionModel.load_from_checkpoint(
            checkpoint_path
        )
        profile_model_name_or_path = model.document_model_name_or_path
    else:
        model = CoordinateAscentModel.load_from_checkpoint(
            checkpoint_path
        )
        profile_model_name_or_path = model.profile_model_name_or_path
    
    print(f"loading data with {num_cpus} CPUs")
    dm = WikipediaDataModule(
        document_model_name_or_path=model.document_model_name_or_path,
        profile_model_name_or_path=profile_model_name_or_path,
        dataset_name='wiki_bio',
        dataset_train_split='train[:100%]',
        dataset_val_split='val[:100%]',
        dataset_test_split='test[:100%]',
        dataset_version='1.2.0',
        num_workers=num_cpus,
        train_batch_size=256,
        eval_batch_size=256,
        max_seq_length=128,
        sample_spans=False,
    )
    dm.setup("fit")

    if is_cross_encoder:
        model_wrapper = CrossEncoderModelWrapper(
            model=model,
            document_tokenizer=dm.document_tokenizer,
            max_seq_length=dm.max_seq_length,
            fake_response=no_model
        )
    else:
        all_profile_embeddings = get_profile_embeddings(model_key=model_key, use_train_profiles=use_train_profiles)
        model_wrapper = MainModelWrapper(
            model=model,
            document_tokenizer=dm.document_tokenizer,
            max_seq_length=dm.max_seq_length,
            profile_embeddings=all_profile_embeddings,
            fake_response=no_model
        )
    model_wrapper.to('cuda')

    constraints = [
        # Prevents us from trying to replace the same word multiple times.
        RepeatModification(),
        # In the sequential setting, this prevents us from re-substituting masks for masks.
        CertainWordsModification(certain_words={'mask', 'MASK'}) 
    ]
    if ignore_stopwords:
        constraints.append(StopwordModification())


    if use_type_swap:
        transformation = WordSwapSingleWordType(single_word=dm.document_tokenizer.mask_token)
    else:
        transformation = WordSwapSingleWordToken(single_word=dm.document_tokenizer.mask_token)

    assert beam_width > 0
    # search_method = textattack.search_methods.GreedyWordSwapWIR()
    search_method = textattack.search_methods.BeamSearch(beam_width=beam_width)

    print(f'***Attacking with k={k} n={n}***')
    goal_function = ChangeClassificationToBelowTopKClasses(
        model_wrapper, k=k, eps=eps,
        min_percent_words=min_percent_words,
        min_idf_weighting=min_idf_weighting,
        max_idf_goal=max_idf_goal,
        table_score=table_score,
        fuzzy_ratio=fuzzy_ratio,
    )
    dataset = WikiDatasetWrapper(
        dm=dm,
        model_wrapper=model_wrapper,
        goal_function=goal_function,
        adv_dataset=adv_dataset,
        max_samples=max(n, 1000)
    )
    attack = Attack(
        goal_function, constraints, transformation, search_method
    )
    attack_args = AttackArgs(
        num_examples_offset=num_examples_offset,
        num_examples=n,
        disable_stdout=False
    )

    if do_reid:
        if (not hasattr(attack_args, "metrics")) or (attack_args.metrics is None):
            attack_args.metrics = {}
        metric = RobertaRobertaReidMetric(
            num_examples_offset=args.num_examples_offset
        )
        attack_args.metrics["RoBERTa-RoBERTa Reid. %"] = metric
    attacker = Attacker(attack, dataset, attack_args)

    folder_path = os.path.join(out_folder_path, model_key)
    if out_file_path is None:
        out_file_path = f'results__b_{beam_width}__ts{table_score}{f"__nomodel" if no_model else ""}{f"__idf{min_idf_weighting}" if (min_idf_weighting is not None) else ""}{("__mp" + str(min_percent_words)) if min_percent_words else ""}{f"__mig{max_idf_goal}" if (max_idf_goal) else ""}__eps{eps}__k_{k}__n_{n}{"__type_swap" if use_type_swap else ""}{"__with_train" if use_train_profiles else ""}'
    out_file_path = os.path.join(folder_path, out_file_path)
    out_csv_path = out_file_path + '.csv'
    logger = CustomCSVLogger(filename=out_csv_path, color_method=None)

    for result in attacker.attack_dataset():
        logger.log_attack_result(result)
        del result

    os.makedirs(folder_path, exist_ok=True)
    logger.flush()
    print('wrote csv to', out_csv_path)

    out_json_path = out_file_path + '__args.json'
    with open(out_json_path, 'w') as json_file:
        json.dump(saved_args, json_file)
    print('wrote json to', out_json_path)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate adversarially-masked examples for a model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--min_percent_words', type=float, default=None,
        help='min_percent_words if we want this instead of k for the goal.'
    )
    parser.add_argument('--k', type=int, default=None,
        help='top-K classes for adversarial goal function'
    )
    parser.add_argument('--eps', type=float, default=None,
        help='max true prob for max-eps criteria'
    )
    parser.add_argument('--n', type=int, default=1000,
        help='number of examples to run on'
    )
    parser.add_argument('--num_examples_offset', type=int, default=0,
        help='offset for search'
    )
    parser.add_argument('--beam_width', '--b', type=int, default=1,
        help='beam width for beam search'
    )
    parser.add_argument('--table_score', type=float, default=0.0,
        help='amount of probability boost to give words that are in the table'
    )
    parser.add_argument('--fuzzy_ratio', type=float, default=0.95,
        help='min fuzzy ratio for fuzzy-matching words to be a match'
    )
    parser.add_argument('--min_idf_weighting', type=float, default=None,
        help=(
            'if weighting probabilities by word IDF, minimum prob for low-IDF words. '
            '(so')
    )
    parser.add_argument('--max_idf_goal', type=float, default=None,
        help=(
            'if we stop when the max IDF falls below a certain value'
        )
    )
    parser.add_argument('--model', '--model_key', type=str, default='model_3_1',
        help='model str name (see model_cfg for more info)',
        choices=model_paths_dict.keys()
    )
    parser.add_argument('--use_train_profiles', action='store_true',
        help='whether to include training profiles in potential people',
    )
    parser.add_argument('--use_type_swap', action='store_true',
        help=(
            'whether to swap words by type instead of token '
            '(i.e. mask all instances of the same word together'
        ),
    )
    parser.add_argument('--ignore_stopwords', default=False, action='store_true',
        help=('whether to ignore stopwords during deidentification')
    )
    parser.add_argument('--no_model', default=False, action='store_true',
        help=('whether to include model scores in deidentification search (otherwise just uses word overlap metrics)')
    )
    parser.add_argument('--do_reid', default=False, action='store_true',
        help=('whether to use a reidentification model')
    )

    args = parser.parse_args()

    assert (args.k is not None) or (args.eps is not None) or (args.min_percent_words is not None)
    return args


if __name__ == '__main__':
    args = get_args()

    if args.no_model: assert "can't use k-criterion with no model"
    main(
        k=args.k,
        eps=args.eps,
        min_percent_words=args.min_percent_words,
        n=args.n,
        num_examples_offset=args.num_examples_offset,
        beam_width=args.beam_width,
        model_key=args.model,
        use_type_swap=args.use_type_swap,
        use_train_profiles=args.use_train_profiles,
        ignore_stopwords=args.ignore_stopwords,
        no_model=args.no_model,
        fuzzy_ratio=args.fuzzy_ratio,
        min_idf_weighting=args.min_idf_weighting,
        table_score=args.table_score,
        max_idf_goal=args.max_idf_goal,
        do_reid=args.do_reid,
    )
    print(args)
