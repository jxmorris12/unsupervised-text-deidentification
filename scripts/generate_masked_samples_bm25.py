import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')

from typing import Any, Dict, List, Tuple

import argparse
import json
import os
import re
import torch

from collections import OrderedDict

import datasets
import numpy as np
import pandas as pd
import textattack
import transformers

from elasticsearch import Elasticsearch

from textattack import Attack
from textattack import Attacker
from textattack import AttackArgs
from textattack.attack_results import SuccessfulAttackResult
from textattack.constraints.pre_transformation import RepeatModification, MaxWordIndexModification
from textattack.loggers import CSVLogger
from textattack.shared import AttackedText


from dataloader import WikipediaDataModule
from model import AbstractModel, CoordinateAscentModel
from model_cfg import model_paths_dict
from utils import get_profile_embeddings_by_model_key


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
    # limit 150 words
    doc = ' '.join(doc.split(' ')[:150])
    # fix braces and remove weird characters
    doc = doc.replace('-lrb-', '(').replace('-rrb-', ')')
    return re.sub(r'[^\w|\s]', ' ',doc)

def process_es_response_hit(response: Dict[str, Any]) -> Tuple[int, float]:
    """Gets the index in the full dataset and score from a response object.
    
    Our dataset indexing order is [...test_dataset, ...val_dataset, ...train_dataset].

    Returns: Tuple(int, float) representing (id, score) where score is un-normalized.
    """
    assert isinstance(response, dict), f"invalid response {response}"
    assert '_id' in response,  f"invalid response {response}"
    assert '_score' in response,  f"invalid response {response}"
    assert '_index' in response,  f"invalid response {response}"

    _id = int(response['_id'])
    if response['_index'] == 'test_100_profile_str':
        full_id = _id
    elif response['_index'] == 'val_100_profile_str':
        # offset by val set IDs
        full_id = _id + 72831 
    elif response['_index'] == 'train_100_profile_str':
        # offset by val set and train set IDs
        full_id = _id + 72831 + 72831
    else:
        raise ValueError(f'Unknown index from ES response hit: {response}')
    
    return full_id, float(response['_score'])


def elasticsearch_msearch(
        es: Elasticsearch,
        max_hits: int,
        query_strings: List[str],
        index: str = 'train_100_profile_str,test_100_profile_str,val_100_profile_str'
    ):
    search_arr = []
    
    for q in query_strings:
        search_arr.append({'index': index })
        search_arr.append(
            {
                # Queries `q` using Lucene syntax.
                "query": {
                    "query_string": {
                        "query": q
                    },
                },
                # Don't return the full profile string, etc. with the result.
                # We just want the ID and the score.
                '_source': False,
                # Only return `max_hits` documents.
                'size': max_hits 
            }
        )
    
    # Create request JSONs.
    request = ''
    request = ' \n'.join([json.dumps(x) for x in search_arr])

    # as you can see, you just need to feed the <body> parameter,
    # and don't need to specify the <index> and <doc_type> as usual 
    resp = es.msearch(body = request)
    return resp

class ChangeClassificationToBelowTopKClasses(textattack.goal_functions.ClassificationGoalFunction):
    k: int
    normalize_scores: bool
    def __init__(self, *args, k: int = 1, normalize_scores: bool, **kwargs):
        self.k = k
        self.normalize_scores = normalize_scores
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, _):
        original_class_score = model_output[self.ground_truth_output]
        num_better_classes = (model_output > original_class_score).sum()
        return num_better_classes >= self.k

    def _get_score(self, model_outputs, _):
        return 1 - model_outputs[self.ground_truth_output]
    
    
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
    
    def __init__(self, dm: WikipediaDataModule, max_samples: int = 1000):
        self.shuffled = True
        self.dm = dm
        # filter out super long examples
        self.dataset = [
            dm.test_dataset[i] for i in range(max_samples)
        ]
        self.label_names = np.array(list(dm.test_dataset['name']) + list(dm.val_dataset['name']) + list(dm.train_dataset['name']))
    
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
        document = self._truncate_text(self.dataset[i]['document'])
            
        input_dict = OrderedDict([
            ('document', document)
        ])
        return input_dict, self.dataset[i]['text_key_id']

class Bm25ModelWrapper(textattack.models.wrappers.ModelWrapper):
    elastic_search: Elasticsearch
    index_names: List[str]
    max_hits: int
    use_train_profiles: bool
    def __init__(self, use_train_profiles: bool, max_hits: int):
        self.elastic_search = get_elastic_search()
        self.use_train_profiles = use_train_profiles
        if use_train_profiles:
            index_names = ['test_100_profile_str', 'val_100_profile_str', 'train_100_profile_str']
        else:
            index_names = ['test_100_profile_str', 'val_100_profile_str']

        existing_indexes = [idx for idx in self.elastic_search.indices.get_alias().keys() if not idx.startswith('.')]
        assert set(index_names) <= set(existing_indexes)
        self.index_names = index_names
        assert max_hits > 0, "need to request at least 1 hit per query to Elasticsearch"
        self.max_hits = max_hits

        # hack for when TextAttack goal function checks `model_wrapper.model.__class__`
        self.model = 'n/a'
    
    @property
    def _num_documents(self) -> int:
        """Num of total documents being searched by BM25 using this elasticsearch instance.
        """
        if self.use_train_profiles:
            return 728321 # test + val + train = 72831 + 72831 + 582659
        else:
            return 145662 # test + val = 72831 + 72831
    
    def _get_search_results(self, text_input_list: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Gets search results from Elasticsearch.

        Args:
            text_input_list List(str): queries
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]:
                search results for each query. Should be of shape (len(text_input_list), 2, self.max_hits).

                Each element in the output list contains an array of top indices returned and 
                    their corresponding scores.
        """
        query_strings = [preprocess_es_query(t) for t in text_input_list]
        results = elasticsearch_msearch(
            es=self.elastic_search,
            max_hits=self.max_hits,
            query_strings=query_strings,
            index=','.join(self.index_names),
        )
        assert len(results['responses']) == len(query_strings)

        result_tuples = [
            [process_es_response_hit(hit) for hit in response['hits']['hits']]
            for response in results['responses']
        ]
        assert len(result_tuples) == len(query_strings)
        assert len(result_tuples[0]) == self.max_hits
        assert len(result_tuples[0][0]) == 2

        # Now unroll each list of tuples to a tuple of lists
        result_lists = [zip(*list_of_tuples) for list_of_tuples in result_tuples]

        # Now convert to np.ndarrays
        return [
            (np.array(idxs_list), np.array(scores_list)) for idxs_list, scores_list in result_lists
        ]

    def __call__(self, text_input_list: List[List[Tuple[int, float]]]) -> np.ndarray:
        score_logits = np.zeros((len(text_input_list), self._num_documents))
        search_results = self._get_search_results(text_input_list=text_input_list)
        for idx, (result_profile_idxs, result_profile_scores) in enumerate(search_results):
            score_logits[idx, result_profile_idxs] = result_profile_scores

        return torch.tensor(score_logits, dtype=torch.float32).softmax(dim=1)


def main(k: int, n: int, num_examples_offset: int, beam_width: int, use_train_profiles: bool):
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

    dataset = WikiDataset(dm=dm)

    model_wrapper = Bm25ModelWrapper(
        use_train_profiles=use_train_profiles,
        # TODO: What's the best way to set max_hits? The problem is that if the correct document
        # is not included in the search results, we won't have a score for it. 
        # Maybe we can specifically get the score for the proper document, so that even when it's outside of
        # the list we know how good it is? But hopefully this works
        # for now.
        max_hits=k*10
    )

    constraints = [
        RepeatModification(),
        MaxWordIndexModification(max_length=dm.max_seq_length),
    ]
    transformation = WordSwapSingleWord(single_word=dm.document_tokenizer.mask_token)
    # search_method = textattack.search_methods.GreedyWordSwapWIR()
    search_method = textattack.search_methods.BeamSearch(beam_width=beam_width)

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

    folder_path = os.path.join('adv_csvs_full_2', model_key)
    os.makedirs(folder_path, exist_ok=True)
    if use_train_profiles:
        out_csv_path = os.path.join(folder_path, f'results__b_{beam_width}__k_{k}__n_{n}_with_train.csv')
    else:
        out_csv_path = os.path.join(folder_path, f'results__b_{beam_width}__k_{k}__n_{n}.csv')
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
        beam_width=args.beam_width,
        use_train_profiles=args.use_train_profiles,
    )
