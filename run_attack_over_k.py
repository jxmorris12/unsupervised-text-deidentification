#!/usr/bin/env python
# coding: utf-8

# # Gradient-based word deletion
# 
# I trained a model to "reidentify" individuals from information about them. Specifically, this model tries to read the beginning of a Wikipedia page and predict (given the infoboxes of many people's Wikipedia page) which person the page is about. Now I'm going to try and fool this "reidentifier" model, and see how many words I have to delete in order to fool the reidentifier a certain percentage of the time.

# ## 1. Load the model and make a prediction

# In[1]:


import sys
sys.path.append('/home/jxm3/research/deidentification/unsupervised-deidentification')


# In[60]:


from model import DocumentProfileMatchingTransformer

checkpoint_path = "/home/jxm3/research/deidentification/unsupervised-deidentification/saves/deid-wikibio_deid_exp/okpvvffw_46/checkpoints/epoch=7-step=1823.ckpt"
model = DocumentProfileMatchingTransformer.load_from_checkpoint(
    checkpoint_path
    dataset_name='wiki_bio',
    model_name_or_path='distilbert-base-uncased',
    num_workers=1,
    loss_fn='exact',
    num_neighbors=2048,
    base_folder="/home/jxm3/research/deidentification/unsupervised-deidentification",
)


# In[3]:


from dataloader import WikipediaDataModule
import os

num_cpus = os.cpu_count()

dm = WikipediaDataModule(
    model_name_or_path='distilbert-base-uncased',
    dataset_name='wiki_bio',
    num_workers=min(8, num_cpus),
    train_batch_size=64,
    eval_batch_size=64,
    max_seq_length=64,
    redaction_strategy="",
    base_folder="/home/jxm3/research/deidentification/unsupervised-deidentification",
)
dm.setup("fit")


# ## 2. Define attack in TextAttack 

# In[4]:


import textattack


# ### (a) Greedy word search + replace with `[MASK]`

# In[5]:


search_method = textattack.search_methods.GreedyWordSwapWIR()

class WordSwapSingleWord(textattack.transformations.word_swap.WordSwap):
    """Takes a sentence and transforms it by replacing with a single fixed word.
    """
    single_word: str
    def __init__(self, single_word: str = "?", **kwargs):
        super().__init__(**kwargs)
        self.single_word = single_word

    def _get_replacement_words(self, _word: str):
        return [self.single_word]


# In[6]:


transformation = WordSwapSingleWord(single_word='[MASK]')
transformation(textattack.shared.AttackedText("Hello my name is Jack"))


# ### (b) "Attack success" as fullfilment of the metric

# In[89]:


from typing import List
import torch

class ChangeClassificationToBelowTopKClasses(textattack.goal_functions.ClassificationGoalFunction):
    k: int
    def __init__(self, *args, k: int = 1, **kwargs):
        self.k = k
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, _):
        original_class_score = model_output[self.ground_truth_output]
        num_better_classes = (model_output > original_class_score).sum()
        return num_better_classes >= self.k

    def _get_score(self, model_output, _):
        return 1 - model_output[self.ground_truth_output]
    
    
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
        elif not ((scores.sum(dim=1) - 1).abs() < 1e-4).all():
            # Values in each row should sum up to 1. The model should return a
            # set of numbers corresponding to probabilities, which should add
            # up to 1. Since they are `torch.float` values, allow a small
            # error in the summation.
            scores = torch.nn.functional.softmax(scores, dim=1)
            if not ((scores.sum(dim=1) - 1).abs() < 1e-4).all():
                raise ValueError("Model scores do not add up to 1.")
        return scores.cpu()


# ## (c) Model wrapper that computes similarities of input documents with validation profiles

# In[69]:


import transformers

class MyModelWrapper(textattack.models.wrappers.ModelWrapper):
    model: DocumentProfileMatchingTransformer
    tokenizer: transformers.PreTrainedTokenizer
    profile_embeddings: torch.Tensor
    max_seq_length: int
    
    def __init__(self, model: DocumentProfileMatchingTransformer, tokenizer: transformers.PreTrainedTokenizer, max_seq_length: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.profile_embeddings = torch.tensor(model.val_embeddings)
        self.max_seq_length = max_seq_length
                 
    def to(self, device):
        self.model.to(device)
        self.profile_embeddings.to(device)
        return self # so semantics `model = MyModelWrapper().to('cuda')` works properly

    def __call__(self, text_input_list, batch_size=32):
        model_device = next(self.model.parameters()).device
        tokenized_ids = self.tokenizer.batch_encode_plus(
            text_input_list,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True
        )
        try:
            tokenized_ids = {k: torch.tensor(v).to(model_device) for k,v in tokenized_ids.items()}
        except:
            breakpoint()
        
        # TODO: implement batch size if we start running out of memory here.
        with torch.no_grad():
            document_embeddings = self.model.document_model(**tokenized_ids)
            document_embeddings = document_embeddings['last_hidden_state'][:, 0, :] # (batch, document_emb_dim)
            document_embeddings = self.model.lower_dim_embed(document_embeddings) # (batch, emb_dim)

        document_to_profile_probs = torch.nn.functional.softmax(
            document_embeddings @ self.profile_embeddings.T.to(model_device), dim=-1)
        assert document_to_profile_probs.shape == (len(text_input_list), len(self.profile_embeddings))
        return document_to_profile_probs
            


# ## (d) Dataset that loads Wikipedia documents with names as labels

# In[9]:


next(iter(dm.val_dataloader()))


# In[20]:


from typing import Tuple

from collections import OrderedDict

import datasets

class WikiDataset(textattack.datasets.Dataset):
    dataset: datasets.Dataset
    
    def __init__(self, dm: WikipediaDataModule):
        self.shuffled = True
        self.dataset = dm.val_dataset
        self.label_names = list(dm.val_dataset['name'])
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, i: int) -> Tuple[OrderedDict, int]:
        input_dict = OrderedDict([
            ('document', self.dataset['document'][i])
        ])
        return input_dict, self.dataset['text_key_id'][i].item()
        


# ## 3. Run attack once

# In[107]:


from textattack.loggers import CSVLogger
from textattack.shared import AttackedText

import pandas as pd
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


# In[90]:


from textattack import Attack
from textattack.constraints.pre_transformation import RepeatModification

model_wrapper = MyModelWrapper(model, dm.tokenizer)
model_wrapper.to('cuda')

goal_function = ChangeClassificationToBelowTopKClasses(model_wrapper, k=10)
constraints = [RepeatModification()]
transformation = WordSwapSingleWord(single_word='[MASK]')
search_method = textattack.search_methods.GreedyWordSwapWIR()

attack = Attack(
    goal_function, constraints, transformation, search_method
)


# In[108]:


# 
#  Initialize attack
# 
from tqdm import tqdm # tqdm provides us a nice progress bar.
from textattack.attack_results import SuccessfulAttackResult
from textattack import Attacker
from textattack import AttackArgs

attack_args = AttackArgs(num_examples=25, disable_stdout=True)
dataset = WikiDataset(dm)

attacker = Attacker(attack, dataset, attack_args)

results_iterable = attacker.attack_dataset()

logger = CustomCSVLogger(color_method='html')


# ## 4. Run attack in loop and make plot for multiple values of $\epsilon$



# 
#  Initialize attack
# 
from tqdm import tqdm # tqdm provides us a nice progress bar.
from textattack.attack_results import SuccessfulAttackResult
from textattack import Attacker
from textattack import AttackArgs

dataset = WikiDataset(dm)

meta_results = []
max_k = 5
total_num_examples = 10
for k in range(0, max_k):
    print(f'***Attacking with k={k}***')
    dataset = WikiDataset(dm)
    goal_function = ChangeClassificationToBelowTopKClasses(model_wrapper, k=k)
    attack = Attack(
        goal_function, constraints, transformation, search_method
    )
    attack_args = AttackArgs(num_examples=total_num_examples, disable_stdout=True)
    attacker = Attacker(attack, dataset, attack_args)

    results_iterable = attacker.attack_dataset()

    logger = CustomCSVLogger(color_method='html')

    for result in results_iterable:
        logger.log_attack_result(result)
    
    meta_results.append( (k,  logger.df['result_type'].value_counts().get('Successful', 0),  logger.df['result_type'].value_counts().get('Failed', 0) ) )

import pandas as pd
meta_df = pd.DataFrame(meta_results, columns=['k', 'Successes', 'Failures']).head()
meta_df.to_csv(f'meta_df_{max_k}_{total_num_examples}.csv')


# In[ ]:




