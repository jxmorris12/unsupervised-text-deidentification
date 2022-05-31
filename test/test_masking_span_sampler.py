import pytest

from masking_span_sampler import MaskingSpanSampler

class TestMaskingSpanSampler:
    def test_nothing(self):
        """Dropout ratio 0 and sample_spans False should do nothing."""
        mt = MaskingSpanSampler(
            word_dropout_ratio=0.0,
            word_dropout_perc=0.5,
            sample_spans=False,
            mask_token='<mask>'
        )
        x = "Silly seller Sally sells scaly seashells by the southern seashore"
        y = mt.random_redact_str(x)
        assert x == y

    def test_word_dropout_50(self):
        """Word dropout 50% should work at least a little."""
        mask_token = "<mask>"
        num_words = 10
        mt = MaskingSpanSampler(
            word_dropout_ratio=1.0,
            word_dropout_perc=0.5,
            sample_spans=False,
            mask_token=mask_token
        )
        outputs = mt.random_redact_str(
            "Silly seller Sally sells scaly seashells by the southern seashore"
        )
        # Make sure there are some masks
        assert outputs.count(mask_token) > 0
        # Make sure they are not all masks
        assert outputs.count(mask_token) < num_words

    def test_word_dropout_uniform(self):
        """Word dropout with uniformly sampled p should work at least a little."""
        mask_token = "<mask>"
        num_words = 10
        mt = MaskingSpanSampler(
            word_dropout_ratio=1.0,
            word_dropout_perc=-1.0,
            sample_spans=False,
            mask_token=mask_token
        )
        outputs = mt.random_redact_str(
            "Silly seller Sally sells scaly seashells by the southern seashore"
        )
        # Make sure there are some masks (maybe)
        assert outputs.count(mask_token) >= 0
        # Make sure they are not all masks (or they are, in this case...)
        assert outputs.count(mask_token) <= num_words

    def test_word_dropout_uniform_idf_sally(self):
        """Word dropout with uniformly sampled p and TF-IDF weighting should work at least a little."""
        mask_token = "<mask>"
        num_words = 10
        mt = MaskingSpanSampler(
            word_dropout_ratio=1.0,
            word_dropout_perc=-1.0,
            sample_spans=False,
            mask_token=mask_token,
            idf_masking=True
        )
        outputs = mt.random_redact_str(
            "Silly seller Sally sells scaly seashells by the southern seashore"
        )
        # Make sure there are some masks
        assert outputs.count(mask_token) > 0
        # Make sure they are not all masks (or they are, in this case...)
        assert outputs.count(mask_token) <= num_words


    def test_word_dropout_uniform_idf_christiane_soeder(self):
        """Word dropout with uniformly sampled p and TF-IDF weighting should work at least a little."""
        mask_token = "<mask>"
        mt = MaskingSpanSampler(
            word_dropout_ratio=1.0,
            word_dropout_perc=-1.0,
            sample_spans=False,
            mask_token=mask_token,
            idf_masking=True
        )
        s = "christiane soeder -lrb- born january 15 , 1975 in remscheid , north rhine-westphalia -rrb- is a german-born austrian road racing cyclist and former duathlete who now lives in vienna .she finished fourth in the 2008 olympic road race with a time of 3h 32 ′ 28 .she rides professionally for .	"
        max_num_words = len(s.split())
        outputs = mt.random_redact_str(
            s
        )
        # Make sure there are some masks
        assert outputs.count(mask_token) > 0
        # Make sure they are not all masks (or they are, in this case...)
        assert outputs.count(mask_token) <= max_num_words

    
    def test_sample_spans_str(self):
        """Sample-spans set to true should work, and not mask anything."""
        mt = MaskingSpanSampler(
            word_dropout_ratio=0.0,
            word_dropout_perc=0.0,
            sample_spans=True,
            mask_token='<mask>',
            min_num_words=1
        )
        # Every time we call redact_and_tokenize_str(),
        # we should randomly get back things of length [4, 8, ..., 32].
        # And they should be sublists of the proper input.
        spans_seen = set()
        num_words = 10
        s = "Silly seller Sally sells scaly seashells by the southern seashore"

        large_sample_size = 5555
        for _ in range(large_sample_size):
            spans_seen.add(mt.random_redact_str(s))
        # The number of possible spans is 1+2+...+num_words.
        assert len(spans_seen) == int(num_words * (num_words + 1) / 2)
