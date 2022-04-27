import pytest
import transformers

from masking_tokenizer import MaskingTokenizer

class TestMaskingTokenizer:
    def test_word_dropout(self):
        tokenizer = (
            transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        )
        mt = MaskingTokenizer(
            tokenizer=tokenizer,
            max_seq_length=32,
            word_dropout_ratio=1.0,
            word_dropout_perc=0.5,
            sample_spans=False,
            adversarial_mask_k_tokens=0
        )
        outputs = mt.redact_and_tokenize_str(
            ["Silly seller Sally sells scaly seashells by the southern seashore"], True
        )
        tokens = outputs['input_ids'][0].tolist()
        non_special_tokens = [t for t in tokens if t not in {101, 102}]
        # Make sure there are some masks
        assert non_special_tokens.count(103) > 0
        # Make sure they are not all masks
        assert non_special_tokens.count(103) < len(non_special_tokens)

    def test_no_word_dropout(self):
        tokenizer = (
            transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        )
        mt = MaskingTokenizer(
            tokenizer=tokenizer,
            max_seq_length=32,
            word_dropout_ratio=0.0,
            word_dropout_perc=0.5,
            sample_spans=False,
            adversarial_mask_k_tokens=0
        )
        outputs = mt.redact_and_tokenize_str(
            ["Silly seller Sally sells scaly seashells by the southern seashore"], True
        )
        tokens = outputs['input_ids'][0].tolist()
        non_special_tokens = [t for t in tokens if t not in {101, 102}]
        # Make sure there are no masks
        assert non_special_tokens.count(103) == 0
    
    def test_sample_spans_str(self):
        tokenizer = (
            transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        )
        mt = MaskingTokenizer(
            tokenizer=tokenizer,
            max_seq_length=32,
            word_dropout_ratio=0.0,
            word_dropout_perc=0.0,
            sample_spans=True,
            adversarial_mask_k_tokens=0
        )
        # Every time we call redact_and_tokenize_str(),
        # we should randomly get back things of length [4, 8, ..., 32].
        # And they should be sublists of the proper input.
        spans_seen = set()
        num_words = 10
        s = "Silly seller Sally sells scaly seashells by the southern seashore"
        for _ in range(5555):
            spans_seen.add(mt._sample_and_word_dropout_text([s])[0])
        # The number of possible spans is 1+2+...+num_words.
        assert len(spans_seen) == int(num_words * (num_words + 1) / 2)
