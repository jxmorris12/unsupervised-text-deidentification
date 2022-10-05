from typing import List, Optional

import functools
import os
import pickle

import textattack
import torch

from fuzzywuzzy import fuzz
from textattack.shared import AttackedText


@functools.cache
def fuzz_ratio(s1: str, s2: str) -> bool:
    return fuzz.ratio(s1, s2)


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
                if word not in {'MASK', 'mask'}:
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