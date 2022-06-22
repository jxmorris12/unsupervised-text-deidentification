import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

from typing import Dict, List, Optional, Set, Tuple

from dataloader import WikipediaDataModule
from model import AbstractModel, CoordinateAscentModel
from model_cfg import model_paths_dict
from utils import get_profile_embeddings_by_model_key

import argparse
import os
import pickle
import re
import torch

from collections import OrderedDict

import datasets
import numpy as np
import pandas as pd
import textattack
import transformers

from textattack import Attack
from textattack import Attacker
from textattack import AttackArgs
from textattack.attack_results import SuccessfulAttackResult
from textattack.constraints import PreTransformationConstraint
from textattack.constraints.pre_transformation import RepeatModification, MaxWordIndexModification
from textattack.loggers import CSVLogger
from textattack.shared import AttackedText


num_cpus = len(os.sched_getaffinity(0))


class CertainWordsModification(PreTransformationConstraint):
    certain_words: Set[str]
    def __init__(self, certain_words: Set[str]):
        self.certain_words = certain_words

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in current_text which are able to be
        deleted."""
        matching_word_idxs = {
            i for i, word in enumerate(current_text.words) if word in self.certain_words
        }
        try:
            return (
                set(range(len(current_text.words)))
                - matching_word_idxs
            )
        except KeyError:
            raise KeyError(
                "`modified_indices` in attack_attrs required for RepeatModification constraint."
            )


class ChangeClassificationToBelowTopKClasses(textattack.goal_functions.ClassificationGoalFunction):
    k: int
    normalize_scores: bool
    def __init__(self, *args, k: int = 1, normalize_scores: bool, **kwargs):
        self.k = k
        self.normalize_scores = normalize_scores
        idf_file_path = os.path.join(
            os.path.dirname(__file__), os.pardir, 'test_val_100_idf.p') # ['test_val_100_idf_date', 'test_val_100_idf.p']
        self.idf = pickle.load(open(idf_file_path, 'rb'))
        self.mean_idf = 11.437707231811393 # mean IDF for test+val corpus
        self.max_idf = 12.176724504431347   # max IDF for test+val corpus
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, _):
        original_class_score = model_output[self.ground_truth_output]
        num_better_classes = (model_output > original_class_score).sum()
        return num_better_classes >= self.k

    def _get_score(self, model_outputs, attacked_text):
        # score = 0.0 - model_outputs.log_softmax(dim=0)[self.ground_truth_output]
        model_score = 0.0 - model_outputs.log_softmax(dim=0).max()
        newly_modified_indices = attacked_text.attack_attrs.get("newly_modified_indices", {})
        if len(newly_modified_indices):
            # Add IDF score.
            mean_idf_score = 0.0
            for word in attacked_text.newly_swapped_words:
                mean_idf_score += self.idf.get(word, self.mean_idf)
            mean_idf_score /= len(attacked_text.newly_swapped_words)
            
            return model_score#  * mean_idf_score
        else:
            return model_score
    
    """have to reimplement the following method to change the precision on the sum-to-one condition."""
    def _process_model_outputs(self, inputs, scores):
        """Processes and validates a list of model outputs.
        This is a task-dependent operation. For example, classification
        outputs need to have a softmax applied.
        """
        # Automatically cast a list or ndarray of predictions to a tensor.
        if isinstance(scores, list):
            scores = torch.tensor(scores)

        # Ensure the returned value is now a tensor.
        if not isinstance(scores, torch.Tensor):
            raise TypeError(
                "Must have list, np.ndarray, or torch.Tensor of "
                f"scores. Got type {type(scores)}"
            )

        # Validation check on model score dimensions
        if scores.ndim == 1:
            # Unsqueeze prediction, if it's been squeezed by the model.
            if len(inputs) == 1:
                scores = scores.unsqueeze(dim=0)
            else:
                raise ValueError(
                    f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
                )
        elif scores.ndim != 2:
            # If model somehow returns too may dimensions, throw an error.
            raise ValueError(
                f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
            )
        elif scores.shape[0] != len(inputs):
            # If model returns an incorrect number of scores, throw an error.
            raise ValueError(
                f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
            )
        elif self.normalize_scores and (not ((scores.sum(dim=1) - 1).abs() < 1e-4).all()):
            # Values in each row should sum up to 1. The model should return a
            # set of numbers corresponding to probabilities, which should add
            # up to 1. Since they are `torch.float` values, allow a small
            # error in the summation.
            # scores = torch.nn.functional(scores, dim=1)
            if not ((scores.sum(dim=1) - 1).abs() < 1e-4).all():
                raise ValueError("Model scores do not add up to 1.")
        return scores.cpu()

class WordSwapSingleWordToken(textattack.transformations.word_swap.WordSwap):
    """Takes a sentence and transforms it by replacing with a single fixed word.
    """
    single_word: str
    def __init__(self, single_word: str = "?", **kwargs):
        super().__init__(**kwargs)
        self.single_word = single_word

    def _get_replacement_words(self, _word: str):
        return [self.single_word]


class WordSwapSingleWordType(textattack.transformations.Transformation):
    """Replaces every instance of each unique word in the text with prechosen word `single_word`.
    """
    def __init__(self, single_word: str = "?", **kwargs):
        super().__init__(**kwargs)
        self.single_word = single_word

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        transformed_texts = []

        unique_words = set(current_text.words)
        for word in unique_words:
            if word == self.single_word:
                continue
            words_to_replace_idxs = set(
                [idx for idx, ct_word in enumerate(current_text.words) if ct_word == word]
            ).intersection(indices_to_modify)
            if not len(words_to_replace_idxs):
                continue

            transformed_texts.append(
                current_text.replace_words_at_indices(
                    list(words_to_replace_idxs), [self.single_word] * len(words_to_replace_idxs)
                )
            )

        return transformed_texts

class CustomCSVLogger(CSVLogger):
    """Logs attack results to a CSV."""

    def log_attack_result(self, result: textattack.goal_function_results.ClassificationGoalFunctionResult):
        original_text, perturbed_text = result.diff_color(self.color_method)
        original_text = original_text.replace("\n", AttackedText.SPLIT_TOKEN)
        perturbed_text = perturbed_text.replace("\n", AttackedText.SPLIT_TOKEN)
        result_type = result.__class__.__name__.replace("AttackResult", "")
        row = {
            "original_person": result.original_result._processed_output[0],
            "original_text": original_text,
            "perturbed_person": result.perturbed_result._processed_output[0],
            "perturbed_text": perturbed_text,
            "original_score": result.original_result.score,
            "perturbed_score": result.perturbed_result.score,
            "original_output": result.original_result.output,
            "perturbed_output": result.perturbed_result.output,
            "ground_truth_output": result.original_result.ground_truth_output,
            "num_queries": result.num_queries,
            "result_type": result_type,
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self._flushed = False

class WikiDataset(textattack.datasets.Dataset):
    dataset: List[Dict[str, str]]
    label_names: List[str]
    dm: WikipediaDataModule
    adv_dataset: Optional[pd.DataFrame]
    
    def __init__(self, dm: WikipediaDataModule, max_samples: int = 1000, adv_dataset: Optional[pd.DataFrame] = None):
        self.shuffled = True
        self.dm = dm
        # filter out super long examples
        self.dataset = [
            dm.test_dataset[i] for i in range(max_samples)
        ]
        self.label_names = np.array(list(dm.test_dataset['name']) + list(dm.val_dataset['name']) + list(dm.train_dataset['name']))
        self.adv_dataset = adv_dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def _truncate_text(self, text: str, max_length: int = 128) -> str:
        input_ids = self.dm.document_tokenizer(
            text,
            truncation=True,
            max_length=self.dm.max_seq_length
        )['input_ids']
        reconstructed_text = (
            # funky way to fix <mask><mask> issue where
            # spaces were removed.
            self.dm.document_tokenizer
                .decode(input_ids)
                .replace('<mask>', ' <mask> ')
                .replace('  <mask>', ' <mask>')
                .replace('<mask>  ', '<mask> ')
                .replace('<s>', '')
                .replace('</s>', '')
                .replace('[CLS]', '')
                .replace('[SEP]', '')
                .strip()
        )
        return reconstructed_text
    
    def _process_adversarial_text(self, text: str, max_length: int = 128) -> str:
        # Put newlines back
        text = text.replace('<SPLIT>', '\n')
        # Standardize mask tokens 
        text = text.replace('[MASK]', self.dm.mask_token)
        text = text.replace('<mask>', self.dm.mask_token)
        # Truncate
        return self._truncate_text(text=text)
    
    def __getitem__(self, i: int) -> Tuple[OrderedDict, int]:
        if self.adv_dataset is None:
            document = self._truncate_text(
                text=self.dataset[i]['document']
            )
        else:
            document = self._process_adversarial_text(
                text=self.adv_dataset.iloc[i]['perturbed_text']
            )

        # This removes non-ascii characters like Chinese etc. letters.
        document = re.sub(r'[^\x00-\x7f]',r'', document) 
            
        input_dict = OrderedDict([
            ('document', document)
        ])
        return input_dict, self.dataset[i]['text_key_id']

class MyModelWrapper(textattack.models.wrappers.ModelWrapper):
    model: AbstractModel
    document_tokenizer: transformers.AutoTokenizer
    profile_embeddings: torch.Tensor
    max_seq_length: int
    
    def __init__(self,
            model: AbstractModel,
            document_tokenizer: transformers.AutoTokenizer,
            profile_embeddings: torch.Tensor,
            max_seq_length: int = 128
        ):
        self.model = model
        self.model.eval()
        self.document_tokenizer = document_tokenizer
        self.profile_embeddings = profile_embeddings.clone().detach()
        self.max_seq_length = max_seq_length
                 
    def to(self, device):
        self.model.to(device)
        self.profile_embeddings = self.profile_embeddings.to(device)
        return self # so semantics `model = MyModelWrapper().to('cuda')` works properly

    def __call__(self, text_input_list):
        model_device = next(self.model.parameters()).device

        tokenized_documents = self.document_tokenizer.batch_encode_plus(
            text_input_list,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        batch = {f"document__{k}": v for k,v in tokenized_documents.items()}

        with torch.no_grad():
            document_embeddings = self.model.forward_document(batch=batch, document_type='document')
            document_to_profile_logits = document_embeddings @ (self.profile_embeddings.T)
        assert document_to_profile_logits.shape == (len(text_input_list), len(self.profile_embeddings))
        return document_to_profile_logits


def get_profile_embeddings(model_key: str, use_train_profiles: bool):
    profile_embeddings = get_profile_embeddings_by_model_key(model_key=model_key)

    print("concatenating train, val, and test profile embeddings")
    if use_train_profiles:
        all_profile_embeddings = torch.cat(
            (profile_embeddings['test'], profile_embeddings['val'], profile_embeddings['train']), dim=0
        )
    else:
        all_profile_embeddings = torch.cat(
            (profile_embeddings['test'], profile_embeddings['val']), dim=0
        )

    print("all_profile_embeddings:", all_profile_embeddings.shape)
    return all_profile_embeddings


def main(
        k: int,
        n: int,
        num_examples_offset: int,
        beam_width: int, 
        model_key: str,
        use_train_profiles: bool,
        use_type_swap: bool,
        adv_dataset: Optional[WikiDataset] = None,
        out_folder_path: str = 'adv_csvs_full_3'
    ):
    checkpoint_path = model_paths_dict[model_key]
    assert isinstance(checkpoint_path, str), f"invalid checkpoint_path {checkpoint_path} for {model_key}"
    print(f"running attack on {model_key} loaded from {checkpoint_path}")
    model = CoordinateAscentModel.load_from_checkpoint(
        checkpoint_path
    )
    
    print(f"loading data with {num_cpus} CPUs")
    dm = WikipediaDataModule(
        document_model_name_or_path=model.document_model_name_or_path,
        profile_model_name_or_path=model.profile_model_name_or_path,
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
    dataset = WikiDataset(dm=dm, adv_dataset=adv_dataset)

    all_profile_embeddings = get_profile_embeddings(model_key=model_key, use_train_profiles=use_train_profiles)
    model_wrapper = MyModelWrapper(
        model=model,
        document_tokenizer=dm.document_tokenizer,
        max_seq_length=dm.max_seq_length,
        profile_embeddings=all_profile_embeddings,
    )
    model_wrapper.to('cuda')
    # breakpoint()

    constraints = [
        RepeatModification(),
        MaxWordIndexModification(max_length=dm.max_seq_length),
        CertainWordsModification(certain_words={'mask', 'MASK'})
    ]

    if use_type_swap:
        transformation = WordSwapSingleWordType(single_word=dm.document_tokenizer.mask_token)
    else:
        transformation = WordSwapSingleWordToken(single_word=dm.document_tokenizer.mask_token)
    # search_method = textattack.search_methods.GreedyWordSwapWIR()
    search_method = textattack.search_methods.BeamSearch(beam_width=beam_width)
    # search_method = IdfWeightedBeamSearch(beam_width=beam_width)

    print(f'***Attacking with k={k} n={n}***')
    goal_function = ChangeClassificationToBelowTopKClasses(model_wrapper, k=k, normalize_scores=False)
    attack = Attack(
        goal_function, constraints, transformation, search_method
    )
    attack_args = AttackArgs(
        num_examples_offset=num_examples_offset,
        num_examples=n,
        disable_stdout=False
    )
    attacker = Attacker(attack, dataset, attack_args)

    logger = CustomCSVLogger(color_method=None)

    for result in attacker.attack_dataset():
        logger.log_attack_result(result)
        del result

    folder_path = os.path.join(out_folder_path, model_key)
    os.makedirs(folder_path, exist_ok=True)
    out_csv_path = os.path.join(folder_path, f'results__b_{beam_width}__k_{k}__n_{n}{"__type_swap" if use_type_swap else ""}{"__with_train" if use_train_profiles else ""}.csv')
    logger.df.to_csv(out_csv_path)
    print('wrote csv to', out_csv_path)
    

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate adversarially-masked examples for a model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--k', type=int, default=1,
        help='top-K classes for adversarial goal function'
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

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(
        k=args.k,
        n=args.n,
        num_examples_offset=args.num_examples_offset,
        beam_width=args.beam_width,
        model_key=args.model,
        use_type_swap=args.use_type_swap,
        use_train_profiles=args.use_train_profiles,
    )
