import pandas as pd

from utils import get_profile_df, words_from_text


class TestUtils:

    def test_get_profile_df(self):
        keys = ['name', 'date', 'hometown']
        values = ['jack', '2021-05-25', 'arlington, virginia']
        df = get_profile_df(keys=keys, values=values)
        sample_df = pd.DataFrame(
            data=[['jack', '2021-05-25', 'arlington, virginia']],
            columns=['name', 'date', 'hometown']
        )
        assert (df == sample_df).to_numpy().all()
    
    def test_words_from_text_empty(self):
        assert words_from_text('') == []
    
    def test_words_from_text_empty_but_punc(self):
        assert words_from_text('. /; -') == []

    def test_words_from_text_simple(self):
        s = 'sally sells seashells to make money for grad school'
        assert words_from_text(s) == s.split()
