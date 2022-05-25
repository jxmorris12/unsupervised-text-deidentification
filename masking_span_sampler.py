from typing import Dict, List

import functools
import re
import random

from nltk.corpus import stopwords
import torch
import transformers

from utils import words_from_text, word_start_and_end_idxs_from_text

eng_stopwords = set(stopwords.words('english'))

@functools.cache
def _cached_words_from_text(s: str) -> List[str]:
    return words_from_text(s)

class MaskingSpanSampler:
    word_dropout_ratio: float     # Percentage of the time to do word dropout
    word_dropout_perc: float      # Percentage of words to replace with mask token
    sample_spans: bool            # Whether or not to sample spans.
    dropout_stopwords: bool
    mask_token: str
    min_num_words: int

    def __init__(
            self,
            word_dropout_ratio: float,
            word_dropout_perc: float, 
            mask_token: str,
            sample_spans: bool,
            min_num_words: int = 8,
            dropout_stopwords: bool = False
        ):
        self.word_dropout_ratio = word_dropout_ratio
        self.word_dropout_perc = word_dropout_perc
        self.dropout_stopwords = dropout_stopwords
        self.sample_spans = sample_spans
        self.mask_token = mask_token
        self.min_num_words = min_num_words

    def _sample_spans(self, text: str) -> str:
        """Sample spans of some words from `text`."""
        #
        # [1/2] Sample spans of words.
        #
        start_and_end_idxs = word_start_and_end_idxs_from_text(text)
        num_words = len(start_and_end_idxs)

        min_num_words = self.min_num_words

        if num_words > min_num_words:
            span_length = random.randint(min_num_words, num_words)
            span_start = random.randint(0, num_words - span_length)
            span_idxs = start_and_end_idxs[span_start:span_start+span_length]
            start_idx = span_idxs[0][0]
            end_idx = span_idxs[-1][1]
            text = text[start_idx : end_idx]
        return text
    
    def _word_dropout(self, text: str) -> str:
        """Randomly mask some words."""
        if random.random() < self.word_dropout_ratio:
            # Don't do dropout this % of the time
            words = set(words_from_text(text))
            if not self.dropout_stopwords:
                words = words - eng_stopwords
            for w in words:
                if random.random() < self.word_dropout_perc:
                    text = re.sub(
                    (r'\b{}\b').format(re.escape(w)),
                        self.mask_token, text, count=0
                    )
        return text
    
    def random_redact_str(self, text: str) -> str:
        """Applies word dropout to a string."""
        assert isinstance(text, str)
        assert len(text) > 0
        assert isinstance(text, str)
        if self.sample_spans:
            text = self._sample_spans(text=text)
        if self.word_dropout_ratio > 0:
            text = self._word_dropout(text=text)
        return text
    
    def fixed_redact_str(
        self, text: str, words_to_mask: List[str]) -> str:
        for w in words_to_mask:
            text = re.sub(
                (r'\b{}\b').format(re.escape(w)),
                self.mask_token, text, count=0
            )
        return text
