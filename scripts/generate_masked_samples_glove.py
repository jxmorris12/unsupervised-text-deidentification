import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

from typing import Dict, List, Optional, Set, Tuple

from datamodule import WikipediaDataModule
from model import AbstractModel, CoordinateAscentModel
from model_cfg import model_paths_dict

from generate_masked_samples import get_profile_embeddings, MyModelWrapper


import argparse
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
from textattack.models.helpers import GloveEmbeddingLayer
from textattack.shared import AttackedText

from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

num_cpus = len(os.sched_getaffinity(0))
eng_stopwords = set(stopwords.words('english'))


@functools.cache
def fuzz_ratio(s1: str, s2: str) -> bool:
    return fuzz.ratio(s1, s2)


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
        # need one, can't have both
        assert (self.k is None) or (self.eps is None)
        assert (self.k is not None) or (self.eps is not None)

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
        if args.k is not None:
            assert self.eps is None
            original_class_score = model_output[self.ground_truth_output]
            # top-k criterion
            num_better_classes = (model_output > original_class_score).sum()
            return num_better_classes >= self.k
        else:
            assert self.eps is not None
            # eps criterion
            return model_output.log_softmax(dim=0)[self.ground_truth_output] <= math.log(self.eps)

    def _percent_words_criterion_is_met(self, model_output, attacked_text) -> bool:
        if self.min_percent_words is None: 
            return True
        # min-% words swapped criterion

        ########################## For counting min % words changed total. #######################
        # num_words_swapped = len(attacked_text.attack_attrs['modified_indices'])
        # num_words_total = len(attacked_text.words)
        ###########################################################################################

        ################# For counting min % overlap with profile words instead. #################
        prof_words = set(self.most_recent_profile_words)
        original_attacked_text = attacked_text
        while original_attacked_text.attack_attrs.get("previous_attacked_text") is not None:
            original_attacked_text = original_attacked_text.attack_attrs["previous_attacked_text"]

        # original number of words overlapping from profile.
        all_overlapping_words = [
            word for word in original_attacked_text.words if ((word in prof_words) and (word not in eng_stopwords))
        ]
        num_words_total = len(all_overlapping_words)

        # there is a weird edge case where there are *no* overlapping words in the profile.
        if num_words_total == 0:
            return True

        # current number of words overlapping from profile. (decreases as words from the profile have been masked.)
        remaining_overlapping_words = [
            word for word in attacked_text.words if ((word in prof_words) and (word not in eng_stopwords))
        ]
        num_words_from_profile_still_in_text = len(remaining_overlapping_words)
        num_words_swapped = num_words_total - num_words_from_profile_still_in_text
        ###########################################################################################


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

    def _is_goal_complete(self, model_output, attacked_text):
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
                print("warning: word with unknown IDF", word)
                return 0.0
        return max(
            self.idf.get(word, 0.0) / self.max_idf, (self.min_idf_weighting or 0.0)
        )
    
    def _word_in_table(self, word: str) -> bool:
        return (
            max([(fuzz_ratio(word, profile_word) / 100.0) for profile_word in self.most_recent_profile_words]) >= self.fuzzy_ratio
        )

    def _get_score(self, model_outputs, attacked_text):
        newly_modified_indices = attacked_text.attack_attrs.get("newly_modified_indices", {})
        if len(newly_modified_indices) == 0:
            return 0.0 - model_outputs[self.ground_truth_output]

        assert len(self.most_recent_profile_words)
        # Add score for matching with table.
        table_score = 0.0
        idf_score = 0.0
        
        if self.table_score > 0:
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

class GloveModelWrapper(textattack.models.wrappers.ModelWrapper):
    fake_response: bool
    
    def __init__(self, dm: WikipediaDataModule, fake_response: bool = False):
        self.model = "glove" # placeholder 
        self.fake_response = fake_response
        self.emb_layer = GloveEmbeddingLayer(emb_layer_trainable=False).cpu()
        self.max_seq_length = 128
        self.tokenizer = textattack.models.tokenizers.GloveTokenizer(
            word_id_map=self.emb_layer.word2id,
            unk_token_id=self.emb_layer.oovid,
            pad_token_id=self.emb_layer.padid,
            max_length=self.max_seq_length,
        )
        self.profile_embeddings = self._embed_test_profiles(dm=dm)
                 
    def to(self, device): # no need for cuda here
        return self
    
    def _embed_text(self, text_input_list: List[str]) -> torch.Tensor:
        with torch.no_grad():
            _input = self.tokenizer.batch_encode(text_input_list)
            _input = torch.tensor(_input).to(self.emb_layer.embedding.weight.device)
            embeddings = self.emb_layer(_input)
        assert embeddings.shape == (len(text_input_list), self.max_seq_length, 200)

        # Mean across tokens that are not padding or OOV.
        mmask = ~(_input == self.emb_layer.padid)
        # TODO: rewrite following line in a cleaner way.
        embeddings = (torch.einsum('bse,bs->be', embeddings, mmask.float()) / mmask.sum(1)[:, None])
        assert embeddings.shape == (len(text_input_list), 200)

        # Normalize embeddings
        embeddings = embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True)

        return embeddings.cpu()
    
    def _embed_test_profiles(self, dm: WikipediaDataModule) -> torch.Tensor:
        def replace_profile_special_tokens(ex: Dict) -> Dict:
            ex["profile"] = ex["profile"].replace("||", "")
            return ex
        profiles = dm.test_dataset.map(replace_profile_special_tokens)["profile"]
        return torch.tensor(
            textattack.shared.utils.batch_model_predict(
                self._embed_text, profiles, batch_size=128
            )
        )

    def __call__(self, text_input_list: List[str]):
        if self.fake_response:
            return torch.ones((len(text_input_list), 72_831 * 2))

        document_embeddings = self._embed_text(text_input_list=text_input_list)

        ##########################################################################################
        # document_to_profile_logits = torch.nn.CosineSimilarity(dim=1)(document_embeddings, self.profile_embeddings)
        document_to_profile_logits = document_embeddings @ self.profile_embeddings.T
        ##########################################################################################
        
        assert document_to_profile_logits.shape == (len(text_input_list), len(self.profile_embeddings))
        return document_to_profile_logits


def main(
        k: int,
        n: int,
        num_examples_offset: int,
        beam_width: int, 
        use_train_profiles: bool,
        use_type_swap: bool,
        eps: Optional[float] = None,
        min_percent_words: Optional[float] = None,
        adv_dataset: Optional[WikiDataset] = None,
        out_folder_path: str = 'adv_csvs_full_8',
        ignore_stopwords: bool = False,
        fuzzy_ratio: float = 0.95,
        max_idf_goal: Optional[float] = None,
        no_model: bool = False,
    ):
    print(f"loading data with {num_cpus} CPUs")
    dm = WikipediaDataModule(
        document_model_name_or_path="roberta-base",
        profile_model_name_or_path="google/tapas-base",
        dataset_name='wiki_bio',
        dataset_train_split='train[:1%]',
        dataset_val_split='val[:1%]',
        dataset_test_split='test[:1%]',
        dataset_version='1.2.0',
        num_workers=num_cpus,
        train_batch_size=256,
        eval_batch_size=256,
        max_seq_length=128,
        sample_spans=False,
    )
    dm.setup("fit")

    model_wrapper = GloveModelWrapper(
        dm=dm, fake_response=no_model
    )

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
        min_idf_weighting=args.min_idf_weighting,
        max_idf_goal=max_idf_goal,
        table_score=args.table_score,
        fuzzy_ratio=fuzzy_ratio,
    )
    dataset = WikiDataset(
        dm=dm,
        goal_function=goal_function,
        adv_dataset=adv_dataset,
        max_samples=max(n, min(1000, len(dm.test_dataset))), 
    )
    attack = Attack(
        goal_function, constraints, transformation, search_method
    )
    attack_args = AttackArgs(
        num_examples_offset=num_examples_offset,
        num_examples=n,
        disable_stdout=False
    )
    do_reid = False

    if do_reid:
        if not hasattr(attack_args, metrics) or (attack_args.metrics is None):
            attack_args.metrics = {}
        attack_args.metrics["RoBERTa-RoBERTa Reid. %"] = RobertaRobertaReidMetric(
            num_examples_offset=args.num_examples_offset
        )
    attacker = Attacker(attack, dataset, attack_args)

    logger = CustomCSVLogger(color_method=None)

    for result in attacker.attack_dataset():
        logger.log_attack_result(result)
        del result

    folder_path = os.path.join(out_folder_path, "glove")
    os.makedirs(folder_path, exist_ok=True)
    out_file_path = os.path.join(folder_path, f'results__b_{beam_width}__ts{args.table_score}{f"__nomodel" if args.no_model else ""}{f"__idf{args.min_idf_weighting}" if (args.min_idf_weighting is not None) else ""}{("__mp" + str(args.min_percent_words)) if args.min_percent_words else ""}{f"__mig{max_idf_goal}" if (max_idf_goal) else ""}__eps{eps}__k_{k}__n_{n}{"__type_swap" if use_type_swap else ""}{"__with_train" if use_train_profiles else ""}')
    out_csv_path = out_file_path + '.csv'
    logger.df.to_csv(out_csv_path)
    print('wrote csv to', out_csv_path)

    out_json_path = out_file_path + '__args.json'
    with open(out_json_path, 'w') as json_file:
        json.dump(vars(args), json_file)
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

    args = parser.parse_args()

    assert (args.k is not None) or (args.eps is not None)
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
        use_type_swap=args.use_type_swap,
        use_train_profiles=args.use_train_profiles,
        ignore_stopwords=args.ignore_stopwords,
        no_model=args.no_model,
        fuzzy_ratio=args.fuzzy_ratio,
        max_idf_goal=args.max_idf_goal,
    )
    print(args)
