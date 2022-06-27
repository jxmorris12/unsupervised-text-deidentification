import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

from typing import Any, Dict, List, Optional, Tuple

import argparse
import dataclasses
import json
import os
import re
import torch

from collections import OrderedDict

import datasets
import numpy as np
import pandas as pd
import transformers

from elasticsearch import Elasticsearch
from nltk.corpus import stopwords

import textattack
from textattack import Attack
from textattack import Attacker
from textattack import AttackArgs
from textattack.attack_results import SuccessfulAttackResult
from textattack.constraints.pre_transformation import RepeatModification, MaxWordIndexModification
from textattack.loggers import CSVLogger
from textattack.shared import AttackedText

from datamodule import WikipediaDataModule

eng_stopwords = set(stopwords.words('english'))

remove_stopwords: bool = True

num_cpus = len(os.sched_getaffinity(0))

def get_elastic_search() -> Elasticsearch:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) 
    
    username = "elastic"
    password = "FjZD_LI-=AJOtsfpq9U*"

    url = f"https://{username}:{password}@rush-compute-01.tech.cornell.edu:9200"

    return Elasticsearch(
        url,
        verify_certs=False
    )

def preprocess_es_query(doc: str) -> str:
    # remove mask token to make sure it doesn't mess w bm25 results
    doc = doc.replace('<mask>', ' ')
    # remove stopwords
    if remove_stopwords:
        words = doc.strip().split(' ')
        words = [w for w in words if (len(w) > 0) and (w not in eng_stopwords)]
        doc = ' '.join(words)
    # limit 150 words
    doc = ' '.join(doc.split(' ')[:150])
    # fix braces and remove weird characters
    doc = doc.replace('-lrb-', '(').replace('-rrb-', ')')
    return re.sub(r'[^\w|\s]', ' ',doc)

def elasticsearch_msearch_by_id(
        es: Elasticsearch,
        query_strings: List[str],
        _id: int,
        max_hits: int,
        index: str,
    ):
    search_arr = []
    
    for q in query_strings:
        search_arr.append({'index': index })
        search_arr.append(
            {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "query_string": {
                                    "query": q
                                }
                            },
                        ],
                    "filter": {
                          "ids": {
                            "values": [_id]
                          }
                    }
                    },
                },
                'size': max_hits,
                'track_total_hits': True,
                '_source': False
            }
        )
    
    request = ''
    request = ' \n'.join([json.dumps(x) for x in search_arr])

    # as you can see, you just need to feed the <body> parameter,
    # and don't need to specify the <index> and <doc_type> as usual 
    resp = es.msearch(body = request)
    return resp

def msearch_total_hits_by_min_score(
        es: Elasticsearch,
        query_strings: List[str],
        min_scores: List[float],
        index: str,
    ):
    """Gets the total number of hits higher than a minimum score for a given query.
    """
    search_arr = []
    
    # from https://stackoverflow.com/a/60857312/2287177:
    #  If _search must be used instead of _count, and you're on Elasticsearch 7.0+,
    # setting size: 0 and track_total_hits: true will provide the same info as _count
    
    assert len(query_strings) == len(min_scores)

    for q, min_score in zip(query_strings, min_scores):
        search_arr.append({'index': index })
        search_arr.append(
            {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "query_string": {
                                    "query": q
                                }
                            },
                        ],
                    },
                },
                "min_score": min_score,
                "track_total_hits": True,
                'size': 1,
                '_source': False
            }
        )
    
    request = ''
    request = ' \n'.join([json.dumps(x) for x in search_arr])

    # as you can see, you just need to feed the <body> parameter,
    # and don't need to specify the <index> and <doc_type> as usual 
    resp = es.msearch(body = request)
    return resp


@dataclasses.dataclass
class Bm25SearchResult:
    # Index of the profile that really matches the query,
    # which is a potentially-redacted document which
    # does correspond to a true profile.
    profile_idx: int
    # The score BM25 assigns to the profile that really matches
    # the query document.
    correct_profile_score: float
    # The number of profiles with better scores than the true document.
    num_better_profiles: int
    # The index of the profile that best matches the query. This
    # may or may not be the same as `profile_idx`, depending if
    # BM25 got it right or not.
    best_matching_profile_idx: int
    # Score of the best-matching profile.
    best_matching_profile_score: float
    # The score of the profile which best matches the query. This
    # may or may not be the same as `profile_idx`, depending if BM25 got it right or not.
    # String of original query, a potentially-redacted document.
    query: str

    @property
    def bm25_was_correct(self) -> bool:
        return self.correct_profile_score == self.best_matching_profile_idx


class Bm25GoalFunctionResult(textattack.goal_function_results.ClassificationGoalFunctionResult):

    @property
    def _processed_output(self):
        """Takes a model output (like `1`) and returns the class labeled output
        (like `positive`), if possible.

        Also returns the associated color.
        """
        output_label = self.raw_output.best_matching_profile_idx
        if "label_names" in self.attacked_text.attack_attrs:
            output = self.attacked_text.attack_attrs["label_names"][self.output]
            output = textattack.shared.utils.process_label_name(output)
            color = textattack.shared.utils.color_from_output(output, output_label)
            del self.attacked_text.attack_attrs["label_names"]
            return output, color
        else:
            color = textattack.shared.utils.color_from_label(output_label)
            return output_label, color

    # def get_text_color_input(self):
    #     """A string representing the color this result's changed portion should
    #     be if it represents the original input."""
    #     _, color = self._processed_output
    #     return color

    # def get_text_color_perturbed(self):
    #     """A string representing the color this result's changed portion should
    #     be if it represents the perturbed input."""
    #     _, color = self._processed_output
    #     return color

    def get_colored_output(self, color_method=None):
        """Returns a string representation of this result's output, colored
        according to `color_method`."""
        output_label = self.raw_output.best_matching_profile_idx
        confidence_score = self.raw_output.best_matching_profile_score
        output, color = self._processed_output
        # concatenate with label and convert confidence score to percent, like '33%'
        output_str = f"{output} ({confidence_score:.1f})"
        return textattack.shared.utils.color_text(output_str, color=color, method=color_method)


class ChangeClassificationToBelowTopKClasses(textattack.goal_functions.ClassificationGoalFunction):
    k: int
    normalize_scores: bool
    def __init__(self, *args, k: int = 1, normalize_scores: bool, **kwargs):
        self.k = k
        self.normalize_scores = normalize_scores
        super().__init__(*args, **kwargs)

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return Bm25GoalFunctionResult

    def _is_goal_complete(self, model_output: Bm25SearchResult, _) -> bool:
        return model_output.num_better_profiles >= self.k

    def _get_score(self, model_output: Bm25SearchResult, _) -> float:
        """The search method is trying to maximize this value. We want to 
        minimize the score BM25 assigns to a (document, profile) pair. So we
        negate the score value.
        """
        return -1 * model_output.correct_profile_score
    
    def _get_displayed_output(self, raw_output: Bm25SearchResult) -> int:
        return raw_output.best_matching_profile_idx

    def _process_model_outputs(self, inputs: List[str], results: List[Bm25SearchResult]):
        """Processes and validates a list of model outputs."""
        assert isinstance(results, list)
        if len(results):
            assert isinstance(results[0], Bm25SearchResult)
        return results

class WordSwapSingleWord(textattack.transformations.word_swap.WordSwap):
    """Takes a sentence and transforms it by replacing with a single fixed word.
    """
    single_word: str
    def __init__(self, single_word: str = "?", **kwargs):
        super().__init__(**kwargs)
        self.single_word = single_word

    def _get_replacement_words(self, _word: str):
        return [self.single_word]

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
    
    def __init__(self, dm: WikipediaDataModule, model_wrapper: textattack.models.wrappers.ModelWrapper, max_samples: int = 1000):
        self.shuffled = True
        self.dm = dm
        # filter out super long examples
        self.dataset = [
            dm.test_dataset[i] for i in range(max_samples)
        ]
        self.label_names = np.array(list(dm.test_dataset['name']) + list(dm.val_dataset['name']) + list(dm.train_dataset['name']))
        self.model_wrapper = model_wrapper
    
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
    
    def __getitem__(self, i: int) -> Tuple[OrderedDict, int]:
        # Need to set the true profile ID on BM25 model wrapper so that
        # it can query for the # of better things instead of returning all top-K.
        # (It's much faster this way.)
        self.model_wrapper.most_recent_test_profile_id = i
        document = self._truncate_text(self.dataset[i]['document'])
            
        input_dict = OrderedDict([
            ('document', document)
        ])
        return input_dict, self.dataset[i]['text_key_id']


class Bm25ModelWrapper(textattack.models.wrappers.ModelWrapper):
    elastic_search: Elasticsearch
    index_names: List[str]
    use_train_profiles: bool
    most_recent_test_profile_id: Optional[int]
    def __init__(self, use_train_profiles: bool):
        self.elastic_search = get_elastic_search()
        self.use_train_profiles = use_train_profiles
        if use_train_profiles:
            index_names = ['test_100_profile_str', 'val_100_profile_str', 'train_100_profile_str']
        else:
            index_names = ['test_100_profile_str', 'val_100_profile_str']

        existing_indexes = [idx for idx in self.elastic_search.indices.get_alias().keys() if not idx.startswith('.')]
        assert set(index_names) <= set(existing_indexes)
        self.index_names = index_names

        # hack for when TextAttack goal function checks `model_wrapper.model.__class__`
        # (TODO: shouldn't need to do this; TextAttack should check with hasattr(model_wrapper, 'model') first...)
        self.model = 'n/a'
    
    @property
    def _num_documents(self) -> int:
        """Num of total documents being searched by BM25 using this elasticsearch instance.
        """
        if self.use_train_profiles:
            return 728321 # test + val + train = 72831 + 72831 + 582659
        else:
            return 145662 # test + val = 72831 + 72831
    
    @property
    def _full_num_indexes(self) -> int:
        return len(self.index_names)
    
    @property
    def _full_search_index(self) -> str:
        return ','.join(self.index_names)
    
    def _get_search_results(self, text_input_list: List[str]) -> List[Bm25SearchResult]:
        """Gets search results from Elasticsearch.

        Args:
            text_input_list List(str): queries
        Returns:
            List[Bm25SearchResult]:
                search results for each query.
        """
        query_strings = list(map(preprocess_es_query, text_input_list))

        # Get scores for the correct IDX for each document.
        correct_profile_results = elasticsearch_msearch_by_id(
            es=self.elastic_search,
            query_strings=query_strings,
            _id=self.most_recent_test_profile_id, 
            max_hits=self._full_num_indexes,
            index=self._full_search_index
        )
        correct_idx_responses = correct_profile_results['responses']
        assert len(correct_profile_results['responses']) == len(query_strings)
        
        correct_profile_score_per_query = []
        for response in correct_idx_responses:
            # Assert we got zero or one response *from each index* for each thing
            assert response['hits']['total']['value'] <= self._full_num_indexes, f"bad response {response}"
            test_set_responses = [hit for hit in response['hits']['hits'] if hit['_index'].startswith('test')]
            if len(test_set_responses) == 0:
                # Edge case where the query has zero overlap with the profile to start with.
                correct_profile_score_per_query.append(0.0)
            else:
                # Take best score from the test set
                assert len(test_set_responses) == 1
                correct_profile_score_per_query.append(test_set_responses[0]['_score'])
        assert len(correct_profile_score_per_query) == len(query_strings)
        
        # Count things that are better, and get top thing if there is one..?
        min_score_responses = msearch_total_hits_by_min_score(
            es=self.elastic_search,
            query_strings=query_strings,
            min_scores=correct_profile_score_per_query,
            index=self._full_search_index,
        )['responses']
        
        min_score_counts = [
            response['hits']['total']['value'] for response in min_score_responses
        ]
        for response, correct_score, query in zip(min_score_responses, correct_profile_score_per_query, text_input_list):
            num_profiles_with_min_score = response['hits']['total']['value']
            if num_profiles_with_min_score == 0:
                # Edge case where original thing has 0 probability and nothing matches it.
                result = Bm25SearchResult(
                    profile_idx=self.most_recent_test_profile_id,
                    correct_profile_score=correct_score,
                    num_better_profiles=self._num_documents,
                    best_matching_profile_score=0.0,
                    best_matching_profile_idx=-1,
                    query=query,
                )
            else:
                # Otherwise actually take this result.
                idx_of_best_matching_profile_with_min_score = int(response['hits']['hits'][0]['_id'])
                best_matching_profile_score = float(response['hits']['hits'][0]['_score'])
                if idx_of_best_matching_profile_with_min_score != self.most_recent_test_profile_id:
                    # If the best-matching profile isn't the true one,
                    # there should be at least one better-matching profile with a higher score...
                    # (These assertions just sanity-check what ElasticSearch is telling us.)
                    assert best_matching_profile_score > correct_score, f"bad response {response}"
                    assert num_profiles_with_min_score > 1, f"bad response {response}"
                result = Bm25SearchResult(
                    profile_idx=self.most_recent_test_profile_id,
                    correct_profile_score=correct_score,
                    num_better_profiles=num_profiles_with_min_score-1,
                    best_matching_profile_score=best_matching_profile_score,
                    best_matching_profile_idx=idx_of_best_matching_profile_with_min_score,
                    query=query,
                )
            yield result

    def __call__(self, text_input_list: List[List[Tuple[int, float]]]) -> List[Bm25SearchResult]:
        score_logits = np.zeros((len(text_input_list), self._num_documents))
        return list(self._get_search_results(text_input_list=text_input_list))


def main(k: int, n: int, num_examples_offset: int, use_train_profiles: bool):
    print(f"loading data with {num_cpus} CPUs")
    dm = WikipediaDataModule(
        document_model_name_or_path='roberta-base',
        profile_model_name_or_path='google/tapas-base',
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

    model_wrapper = Bm25ModelWrapper(
        use_train_profiles=use_train_profiles,
    )
    dataset = WikiDataset(dm=dm, model_wrapper=model_wrapper)

    constraints = [
        RepeatModification(),
    ]
    bm25_mask_token = '<mask>'
    transformation = WordSwapSingleWord(single_word=bm25_mask_token)
    # We can just use greedy-WIR here since BM25 is nonlinear. No need to do any search.
    search_method = textattack.search_methods.GreedyWordSwapWIR(unk_token=bm25_mask_token)
    # search_method = textattack.search_methods.BeamSearch(beam_width=1)

    print(f'***Attacking with k={k} n={n}***')
    goal_function = ChangeClassificationToBelowTopKClasses(model_wrapper, k=k, normalize_scores=True)
    attack = Attack(
        goal_function, constraints, transformation, search_method
    )
    attack_args = AttackArgs(
        num_examples_offset=num_examples_offset,
        num_examples=n,
        disable_stdout=False
    )
    attacker = Attacker(attack, dataset, attack_args)

    results_iterable = attacker.attack_dataset()

    logger = CustomCSVLogger(color_method=None)

    for result in results_iterable:
        logger.log_attack_result(result)

    model_name = 'bm25' if not remove_stopwords else 'bm25_remove_stopwords'
    folder_path = os.path.join('adv_csvs_full_3', model_name)
    os.makedirs(folder_path, exist_ok=True)
    if use_train_profiles:
        out_csv_path = os.path.join(folder_path, f'results__bm25__k_{k}__n_{n}_with_train.csv')
    else:
        out_csv_path = os.path.join(folder_path, f'results__bm25__k_{k}__n_{n}.csv')
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
    parser.add_argument('--use_train_profiles', type=bool,
        help='whether to include training profiles in potential people',
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(
        k=args.k,
        n=args.n,
        num_examples_offset=args.num_examples_offset,
        use_train_profiles=args.use_train_profiles,
    )
