from typing import Dict, List

import torch
import transformers

import re
import random

from utils import words_from_text, word_start_and_end_idxs_from_text

class MaskingSpanSampler:
    word_dropout_ratio: float     # Percentage of the time to do word dropout
    word_dropout_perc: float      # Percentage of words to replace with mask token
    sample_spans: True            # Whether or not to sample spans.
    mask_token: str
    min_num_words: int

    def __init__(
            self,
            word_dropout_ratio: float,
            word_dropout_perc: float, 
            mask_token: str,
            sample_spans: bool,
            min_num_words: int = 8
        ):
        self.word_dropout_ratio = word_dropout_ratio
        self.word_dropout_perc = word_dropout_perc
        self.sample_spans = sample_spans
        self.mask_token = mask_token
        self.min_num_words = min_num_words
        
        print('[***] Masking hyperparameters:', 
            'ratio:', word_dropout_ratio, '/',
            'percentage:', word_dropout_perc, '/',
            '\t sample_spans:', sample_spans,
        )

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
            for w in words_from_text(text):
                if random.random() < self.word_dropout_perc:
                    text = re.sub(
                        (r'\b{}\b').format(w),
                        self.mask_token, text, 1
                    )
        return text
    
    def redact(self, text: str) -> Dict[str, torch.Tensor]:
        assert len(text) > 0
        assert isinstance(text, str)
        if self.sample_spans:
            text = self._sample_spans(text=text)
        if self.word_dropout_ratio > 0:
            text = self._word_dropout(text=text)
        return text

    
    # def redact_and_tokenize_ids_from_grad(self, 
    #     input_ids: torch.Tensor, model: transformers.PreTrainedModel, k: int, mask_token_id: int) -> torch.Tensor:
    #     """Masks tokens in `input_ids` proportional to gradient."""
    #     assert hasattr(model, 'embeddings.word_embeddings')
    #     assert isinstance(model.embeddings.word_embeddings, torch.nn.Embedding)
    #     topk_tokens = (
    #         model.embeddings.word_embeddings.weight.grad.norm(p=2, dim=1).argsort()
    #     )
    #     special_tokens_mask = (
    #         (topk_tokens == 0) | (topk_tokens == 100) | (topk_tokens == 101) | (topk_tokens == 102) | (topk_tokens == 103)
    #     )
    #     topk_tokens = topk_tokens[~special_tokens_mask][-k:]
    #     topk_mask = (
    #         input_ids[..., None].to(topk_tokens.device) == topk_tokens[None, :]).any(dim=-1)
    #     # print('topk_tokens:', self.tokenizer.decode(topk_tokens))

    #     return (
    #         topk_tokens, torch.where(
    #             topk_mask,
    #             torch.tensor(mask_token_id)[None, None].to(topk_tokens.device),
    #             input_ids.to(topk_tokens.device)
    #         )
    #     )