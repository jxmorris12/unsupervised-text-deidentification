from typing import Dict, List

import torch
import transformers

import re
import random

from utils import words_from_text, word_start_and_end_idxs_from_text

class MaskingTokenizer:
    tokenizer: transformers.AutoTokenizer
    max_seq_length: int
    word_dropout_ratio: float     # Percentage of the time to do word dropout
    word_dropout_perc: float      # Percentage of words to replace with mask token
    sample_spans: True              # Whether or not to sample spans.
    adversarial_mask_k_tokens: int  # Number of tokens to adversarially mask, if doing this. (0 if not.)

    def __init__(
        self, tokenizer: transformers.AutoTokenizer, max_seq_length: int,
        word_dropout_ratio: float, word_dropout_perc: float, 
        sample_spans: bool, adversarial_mask_k_tokens: int
        ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.word_dropout_ratio = word_dropout_ratio
        self.word_dropout_perc = word_dropout_perc
        self.sample_spans = sample_spans
        self.adversarial_mask_k_tokens = adversarial_mask_k_tokens

        err = "Doing span-sampling and adversarially masking jointly is not currently supported"
        assert not (sample_spans and (adversarial_mask_k_tokens > 0)), err
        
        print('[***] Masking hyperparameters:', 
            'ratio:', word_dropout_ratio, '/',
            'percentage:', word_dropout_perc, '/',
            'token:', tokenizer.mask_token,
            '\t sample_spans:', sample_spans,
            '\t adversarial_mask_k_tokens:', adversarial_mask_k_tokens
        )

    def _sample_and_word_dropout_text(self, text: List[str]) -> Dict[str, torch.Tensor]:
        """Apply word dropout to list of text inputs."""
        # TODO: implement this in dataloader to take advantage of multiprocessing
        for i in range(len(text)):
            #
            # [1/2] Sample spans of words.
            #
            if self.sample_spans:
                start_and_end_idxs = word_start_and_end_idxs_from_text(text[i])
                num_words = len(start_and_end_idxs)
                span_length = random.randint(1, num_words)
                span_start = random.randint(0, num_words - span_length)
                span_idxs = start_and_end_idxs[span_start:span_start+span_length]
                start_idx = span_idxs[0][0]
                end_idx = span_idxs[-1][1]
                text[i] = text[i][start_idx : end_idx]

            #
            # [2/2] Randomly mask some words.
            #
            if random.random() > self.word_dropout_ratio:
                # Don't do dropout this % of the time
                continue
            for w in words_from_text(text[i]):
                if random.random() < self.word_dropout_perc:
                    text[i] = re.sub(
                        (r'\b{}\b').format(w),
                        self.tokenizer.mask_token, text[i], 1
                    )
        return text
    
    def redact_and_tokenize_str(self, text: List[str], training: bool) -> Dict[str, torch.Tensor]:
        assert isinstance(text, list)
        assert len(text) > 0
        assert isinstance(text[0], str)
        if training:
            # Do word dropout.
            breakpoint()
            text = self._sample_and_word_dropout_text(text=text)
            
        inputs = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        if training and self.sample_spans:
            # Sample spans.
            inputs = self._sample_spans_from_inputs(inputs)
        return inputs

    
    def redact_and_tokenize_ids_from_grad(self, 
        input_ids: torch.Tensor, model: transformers.PreTrainedModel, k: int, mask_token_id: int) -> torch.Tensor:
        """Masks tokens in `input_ids` proportional to gradient."""
        assert hasattr(model, 'embeddings.word_embeddings')
        assert isinstance(model.embeddings.word_embeddings, torch.nn.Embedding)
        topk_tokens = (
            model.embeddings.word_embeddings.weight.grad.norm(p=2, dim=1).argsort()
        )
        special_tokens_mask = (
            (topk_tokens == 0) | (topk_tokens == 100) | (topk_tokens == 101) | (topk_tokens == 102) | (topk_tokens == 103)
        )
        topk_tokens = topk_tokens[~special_tokens_mask][-k:]
        topk_mask = (
            input_ids[..., None].to(topk_tokens.device) == topk_tokens[None, :]).any(dim=-1)
        # print('topk_tokens:', self.tokenizer.decode(topk_tokens))

        return (
            topk_tokens, torch.where(
                topk_mask,
                torch.tensor(mask_token_id)[None, None].to(topk_tokens.device),
                input_ids.to(topk_tokens.device)
            )
        )