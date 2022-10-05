import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

from typing import Dict, List, Optional, Set, Tuple

from datamodule import WikipediaDataModule
from model import AbstractModel, CoordinateAscentModel
from model_cfg import model_paths_dict
from utils import get_profile_embeddings_by_model_key

import argparse
import csv
import functools
import json
import math
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
from textattack.constraints.pre_transformation import MaxModificationRate, RepeatModification, MaxWordIndexModification, StopwordModification
from textattack.loggers import CSVLogger
from textattack.shared import AttackedText

from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

num_cpus = len(os.sched_getaffinity(0))
eng_stopwords = set(stopwords.words('english'))


@functools.cache
def fuzz_ratio(s1: str, s2: str) -> bool:
    return fuzz.ratio(s1, s2)


class CertainWordsModification(PreTransformationConstraint):
    """Constraint to modify certain words. This prevents us from modifying any words that are 'MASK' 
    in the event we're re-masking some already-masked text."""
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

class RobertaRobertaReidMetric(textattack.metrics.Metric):
    model_key: str
    num_examples_offset: int
    print_identified_results: bool
    def __init__(self, num_examples_offset: int, print_identified_results: bool = True):
        self.model_key = "model_3_3"
        self.num_examples_offset = num_examples_offset
        # TODO: enhance this class to support shuffled indices from the attack.
        #   Right now things must be sequential.
    
    def _document_from_attack_result(self, result: textattack.attack_results.AttackResult):
        document = result.perturbed_result.attacked_text.text
        return document.replace("[MASK]", "<mask>")
    
    def calculate(self, attack_results: List[textattack.attack_results.AttackResult]) -> Dict[str, float]:
        # TODO move to logging statement
        print("Computing reidentification score...")
        # get profile embeddings
        all_profile_embeddings = get_profile_embeddings(model_key=self.model_key, use_train_profiles=True)

        # initialize model
        model = CoordinateAscentModel.load_from_checkpoint(
            model_paths_dict[self.model_key]
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
        model_wrapper = MyModelWrapper(
            model=model,
            document_tokenizer=tokenizer,
            max_seq_length=128,
            profile_embeddings=all_profile_embeddings,
        ).to('cuda')

        # get documents
        documents = list(
            map(self._document_from_attack_result, attack_results)
        )

        # check accuracy
        predictions = model_wrapper(documents)
        true_labels = (
            torch.arange(len(attack_results)) + self.num_examples_offset
        ).cuda()
        correct_preds = (predictions.argmax(dim=1) == true_labels)
        accuracy = correct_preds.float().mean()

        for i, pred in enumerate(correct_preds.tolist()):
            if pred:
                print(f'Identified example {i}:', attack_results[i])

        model_wrapper.model.cpu()
        del model_wrapper
        return f'{accuracy.item()*100.0:.2f}'


class ChangeClassificationToBelowTopKClasses(textattack.goal_functions.ClassificationGoalFunction):
    """A goal function that plugs into TextAttack to provide the top-K objective needed for deidentification.

    Also implements the IDF-based + table-scoring baseline (which don't use a DL model) as well as some other features,
    like fuzzy text matching, that we decided not to include in the paper.
    """
    k: Optional[int]
    min_percent_words: Optional[float]
    most_recent_profile_words: List[str]
    min_idf_weighting: Optional[float]
    table_score: float
    max_idf_goal: float
    fuzzy_ratio: float
    eps: float
    def __init__(self, *args, k: Optional[int] = None, eps: Optional[float] = None, max_idf_goal: Optional[float] = None, min_idf_weighting: Optional[float] = None, min_percent_words: Optional[float] = None, table_score = 0.30, 
        fuzzy_ratio: float = 0.95, **kwargs):
        self.k = k
        self.eps = eps
        # need one
        assert ((self.k is None) ^ (self.eps is None)) or (min_percent_words is not None)

        self.fuzzy_ratio = fuzzy_ratio
        self.min_percent_words = min_percent_words
        self.max_idf_goal = max_idf_goal
        self.table_score = table_score
        if self.min_percent_words is not None:
            print(f'using criteria min_percent_words = {min_percent_words} with k = {k}')

        self.most_recent_profile_words = [] # Set asynchronously by the dataset. (I know this is a bad pattern. TODO: fix this pattern.)

        idf_file_path = os.path.join(
            os.path.dirname(__file__), os.pardir, 'test_val_train_100_idf.p') # ['test_val_100_idf_dates', 'test_val_100_idf.p', 'test_val_train_100_idf_dates.p', 'test_val_train_100_idf.p']
        self.idf = pickle.load(open(idf_file_path, 'rb'))
        self.mean_idf = 11.437707231811393  # mean IDF for test+val corpus
        self.max_idf = 12.176724504431347   # max IDF for test+val corpus

        # Custom IDF values for stuff that could appear 
        self.idf[','] = 1.0
        self.idf['.'] = 1.0

        self.min_idf_weighting = min_idf_weighting

        super().__init__(*args, **kwargs)

    def _k_criterion_is_met(self, model_output, attacked_text) -> bool:
        if self.k is not None:
            assert self.eps is None
            original_class_score = model_output[self.ground_truth_output]
            # top-k criterion
            num_better_classes = (model_output > original_class_score).sum()
            return num_better_classes >= self.k
        elif self.eps is not None:
            # eps criterion
            return model_output.log_softmax(dim=0)[self.ground_truth_output] <= math.log(self.eps)
        else:
            # just min-percent-words
            return True

    def _percent_words_criterion_is_met(self, model_output, attacked_text) -> bool:
        if self.min_percent_words is None: 
            return True
        num_words_swapped = len(attacked_text.attack_attrs['modified_indices'])
        num_words_total = len(attacked_text.words)
        return (
            ((num_words_swapped + 0.5) / num_words_total) >= self.min_percent_words
        )
    
    def _max_idf_goal_is_met(self, attacked_text: AttackedText) -> bool:
        if self.max_idf_goal is None:
            return True
        try:
            max_idf = max(
                [
                    self.idf[word] 
                    for i, word in enumerate(attacked_text.words) 
                    if (i not in attacked_text.attack_attrs["modified_indices"]) and (word.isalnum())
                ]
            )
        except ValueError: # "max is an empty sequence" -> no more words to modify.
            return True

        return max_idf <= self.max_idf_goal

    def _is_goal_complete(self, model_output, attacked_text) -> bool:
        return (
            self._percent_words_criterion_is_met(model_output, attacked_text) 
            and 
            self._k_criterion_is_met(model_output, attacked_text)
            and
            self._max_idf_goal_is_met(attacked_text)
        ) 
    
    @functools.cache
    def get_word_idf_prob(self, word: str) -> float:
        # Clamp at a certain value. to prevent low-probability words from being too improbable.
        if word not in self.idf:
            if not word.isalnum(): 
                return 0.0
            else:
                print(f"warning: word with unknown IDF: `{word}`")
                return 0.0
        return max(
            self.idf.get(word, 0.0) / self.max_idf, (self.min_idf_weighting or 0.0)
        )
    
    def _word_in_table(self, word: str) -> bool:
        return (
            max([(fuzz_ratio(word, profile_word) / 100.0) for profile_word in self.most_recent_profile_words]) >= self.fuzzy_ratio
        )

    def _get_score(self, model_outputs, attacked_text) -> float:
        """Returns a score for a new AttackedText (probably a swapped word). Out of many potential scored
        texts, the one with the best score will be taken. This will probably be the word-masking that
        changes the model score the most.
        """
        newly_modified_indices = attacked_text.attack_attrs.get("newly_modified_indices", {})
        if len(newly_modified_indices) == 0:
            return 0.0 - model_outputs[self.ground_truth_output]

        assert len(self.most_recent_profile_words)
        # Add score for matching with table.
        table_score = 0.0
        idf_score = 0.0
        
        for word in attacked_text.newly_swapped_words:
            if (self.table_score > 0) and self._word_in_table(word):
                table_score += self.table_score # Intuition is we want use the table to break ties of about this much % probability.
            idf_score += self.get_word_idf_prob(word)
        idf_score /= len(attacked_text.newly_swapped_words)
        table_score /= len(attacked_text.newly_swapped_words)
        # print('\t\t',attacked_text.newly_swapped_words, table_score, '//', idf_score, '//', model_score)

        model_output_stable = model_outputs - model_outputs.max()
        softmax_denominator = model_output_stable.exp().sum()
        # This is a numerically-stable softmax that incorporates the table score in probability space.
        total_score = (
            -1.0 * model_output_stable[self.ground_truth_output].exp()
          + (softmax_denominator * (1 + table_score))
        ) / softmax_denominator

        if ((self.min_idf_weighting is not None) and self.min_idf_weighting < 1.0):
            return total_score * idf_score
        else:
            return total_score
    
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

    *Not* a wordswap since this one can swap multiple words at once.
    """
    def __init__(self, single_word: str = "?", fuzzy_ratio: int = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.single_word = single_word
        self.fuzzy_ratio = fuzzy_ratio
    
    def words_match(self, w1: str, w2: str):
        # print("\t\tw1", w1, "w2", w2, "fuzz.ratio(w1, w2)", fuzz.ratio(w1, w2))
        if min(len(w1), len(w2)) < 4:
            # Exact-match on short strings, since fuzzywuzzy doesn't seem to work quite right here.
            return w1 == w2
        else:
            return (fuzz_ratio(w1, w2) / 100.0) >= self.fuzzy_ratio

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []

        unique_words = set(current_text.words)
        for i in indices_to_modify:
            word = current_text.words[i]
            if word == self.single_word:
                continue
            words_to_replace_idxs =  set(
                    [idx for idx, ct_word in enumerate(current_text.words) if self.words_match(ct_word, word)]
                  + [i]
            ).intersection(indices_to_modify)
            if not len(words_to_replace_idxs):
                continue
            # print("word", word, "words_to_replace_idxs", words_to_replace_idxs)

            transformed_texts.append(
                current_text.replace_words_at_indices(
                    list(words_to_replace_idxs), [self.single_word] * len(words_to_replace_idxs)
                )
            )

        return transformed_texts

class CustomCSVLogger(CSVLogger):
    """Logs attack results to a CSV."""
    def __init__(self, filename="results.csv", color_method="file"):
        textattack.shared.logger.info(f"Logging to CSV at path {filename}")
        assert ".csv" in filename
        self.filename = filename
        self.pickle_filename = filename.replace(".csv", "_examples.p")
        self.color_method = color_method
        self.row_list = []
        self.example_strings_list = [] # for each example, *all the strings* 
        self._flushed = True
    
    def _get_example_strings(self, pt: AttackedText) -> List[str]:
        """The list of all texts that an AttackedText has been. Used to get the text
        at all levels of masking, from no masks to all masks.
        """
        strings = [pt.text]
        while 'prev_attacked_text' in pt.attack_attrs:
            strings.append(pt.attack_attrs['prev_attacked_text'].text)
            pt = pt.attack_attrs['prev_attacked_text']
        return strings[::-1]


    def log_attack_result(self, result: textattack.goal_function_results.ClassificationGoalFunctionResult):
        original_text, perturbed_text = result.diff_color(self.color_method)
        original_text = original_text.replace("\n", AttackedText.SPLIT_TOKEN)
        perturbed_text = perturbed_text.replace("\n", AttackedText.SPLIT_TOKEN)
        # how to go back to prev attacked text?
        # (
        #    result.perturbed_result.attacked_text
        #       .attack_attrs['prev_attacked_text']
        #       .attack_attrs['prev_attacked_text']
        #       .attack_attrs['prev_attacked_text']
        #       .attack_attrs['prev_attacked_text']
        # )
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
        self.row_list.append(row)
        self.example_strings_list.append(self._get_example_strings(result.perturbed_result.attacked_text))
        self._flushed = False

    def flush(self):
        self.df = pd.DataFrame.from_records(self.row_list)
        self.df.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
        pickle.dump(self.example_strings_list, open(self.pickle_filename, 'wb'))
        textattack.shared.logger.info(f"Wrote examples to file at {self.pickle_filename}")
        self._flushed = True

    def close(self):
        # self.fout.close()
        super().close()


class WikiDataset(textattack.datasets.Dataset):
    """TextAttack dataset to load examples from Wikipedia."""
    dataset: List[Dict[str, str]]
    label_names: List[str]
    dm: WikipediaDataModule
    adv_dataset: Optional[pd.DataFrame]
    goal_function: ChangeClassificationToBelowTopKClasses
    
    def __init__(
            self,
            dm: WikipediaDataModule,
            goal_function: ChangeClassificationToBelowTopKClasses,
            max_samples: int = 1000,
            adv_dataset: Optional[pd.DataFrame] = None
        ):
        self.shuffled = True
        self.dm = dm
        self.goal_function = goal_function
        # filter out super long examples
        dataset = []
        i = 0
        while len(dataset) < max_samples:
            if i >= len(dm.test_dataset):
                # aren't enough samples to complete; skip
                break
            # TODO add min-num-rows constraint here?
            dataset.append(dm.test_dataset[i])
            i += 1
        self.dataset = dataset
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
            self.dm.document_tokenizer.decode(input_ids).strip()
        )
        num_tokenizable_words = len(reconstructed_text.split(' '))
        # Subtract one here as a buffer in case the last word in `reconstructed_text`
        # was a half-tokenized one. Otherwise we could accidentally leak information
        # through additional subtokens in the last word!! This could happen if
        # our deid model only sees the first token of the last word, and thinks it's benign,
        # so doesn't mask it, but then it stays in the final output and is identifiable
        # by a different model with a longer max sequence length.
        return ' '.join(text.split(' ')[:num_tokenizable_words - 1])
    
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
        
        self.goal_function.most_recent_profile_words = set(
            textattack.shared.utils.words_from_text(
                self.dataset[i]['profile']
            )
        )

        input_dict = OrderedDict([
            ('document', document)
        ])
        return input_dict, self.dataset[i]['text_key_id']

class MyModelWrapper(textattack.models.wrappers.ModelWrapper):
    model: AbstractModel
    document_tokenizer: transformers.AutoTokenizer
    profile_embeddings: torch.Tensor
    max_seq_length: int
    fake_response: bool
    
    def __init__(self,
            model: AbstractModel,
            document_tokenizer: transformers.AutoTokenizer,
            profile_embeddings: torch.Tensor,
            max_seq_length: int = 128,
            fake_response: bool = False
        ):
        self.model = model
        self.model.eval()
        self.document_tokenizer = document_tokenizer
        self.profile_embeddings = profile_embeddings.clone().detach()
        self.max_seq_length = max_seq_length
        self.fake_response = fake_response
                 
    def to(self, device):
        self.model.to(device)
        self.profile_embeddings = self.profile_embeddings.to(device)
        return self # so semantics `model = MyModelWrapper().to('cuda')` works properly

    def __call__(self, text_input_list):
        if self.fake_response:
            return torch.ones((len(text_input_list), 72_831 * 2))
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
        n: int,
        num_examples_offset: int,
        k: int,
        beam_width: int, 
        model_key: str,
        use_train_profiles: bool,
        use_type_swap: bool,
        eps: Optional[float] = None,
        min_percent_words: Optional[float] = None,
        adv_dataset: Optional[WikiDataset] = None,
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
    saved_args = locals()
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

    all_profile_embeddings = get_profile_embeddings(model_key=model_key, use_train_profiles=use_train_profiles)
    model_wrapper = MyModelWrapper(
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
        # idf = pickle.load(
        #     open(
        #         os.path.join(os.path.dirname(__file__), os.pardir, 'test_val_train_100_idf_dates.p'), 
        #         'rb'
        #     )
        # )
        # stopwords = {k for k,v in idf.items() if v<2}
        # constraints.append(StopwordModification(stopwords=stopwords))
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
    dataset = WikiDataset(
        dm=dm,
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
        json.dump(vars(saved_args), json_file)
    print('wrote json to', out_json_path)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate adversarially-masked examples for a model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--min_percent_words', type=float, default=None,
        help=(
            'min_percent_words if we want this instead of k for the goal.'
        )
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
