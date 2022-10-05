from typing import Dict, List, Optional, Tuple

from collections import OrderedDict

import numpy as np
import pandas as pd
import textattack

from datamodule import WikipediaDataModule
from deidentification.goal_functions import ChangeClassificationToBelowTopKClasses

class WikiDatasetWrapper(textattack.datasets.Dataset):
    """TextAttack dataset to load examples from Wikipedia."""
    dataset: List[Dict[str, str]]
    label_names: List[str]
    dm: WikipediaDataModule
    model_wrapper: textattack.models.wrappers.ModelWrapper
    adv_dataset: Optional[pd.DataFrame]
    goal_function: ChangeClassificationToBelowTopKClasses
    
    def __init__(
            self,
            dm: WikipediaDataModule,
            model_wrapper: textattack.models.wrappers.ModelWrapper,
            goal_function: ChangeClassificationToBelowTopKClasses,
            max_samples: int = 1000,
            adv_dataset: Optional[pd.DataFrame] = None
        ):
        self.shuffled = True
        self.dm = dm
        self.model_wrapper = model_wrapper
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
        
        self.model_wrapper.most_recent_datapoint = self.dataset[i]
        self.model_wrapper.most_recent_datapoint_idx = i
        self.goal_function.most_recent_profile_words = set(
            textattack.shared.utils.words_from_text(
                self.dataset[i]['profile']
            )
        )

        input_dict = OrderedDict([
            ('document', document)
        ])
        return input_dict, self.dataset[i]['text_key_id']