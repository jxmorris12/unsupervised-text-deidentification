from typing import Iterable, List

import textattack
from textattack.shared import AttackedText

from utils import fuzz_ratio


class WordSwapSingleWordType(textattack.transformations.Transformation):
    """Replaces every instance of each unique word in the text with prechosen word `single_word`.

    *Not* a wordswap since this one can swap multiple words at once.
    """
    def __init__(self, single_word: str = "?", fuzzy_ratio: int = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.single_word = single_word
        self.fuzzy_ratio = fuzzy_ratio
    
    def words_match(self, w1: str, w2: str):
        # print("\t\tw1", w1, "w2", w2, "fuzz.ratio(w1, w2)", fuzz.ratio(w1, w2))
        if min(len(w1), len(w2)) < 4:
            # Exact-match on short strings, since fuzzywuzzy doesn't seem to work quite right here.
            return w1 == w2
        else:
            return (fuzz_ratio(w1, w2) / 100.0) >= self.fuzzy_ratio

    def _get_transformations(
        self, current_text: AttackedText, indices_to_modify: Iterable[int]) -> List[AttackedText]:
        transformed_texts = []

        unique_words = set(current_text.words)
        for i in indices_to_modify:
            word = current_text.words[i]
            if word == self.single_word:
                continue
            words_to_replace_idxs =  set(
                    [idx for idx, ct_word in enumerate(current_text.words) if self.words_match(ct_word, word)]
                  + [i]
            ).intersection(indices_to_modify)
            if not len(words_to_replace_idxs):
                continue
            # print("word", word, "words_to_replace_idxs", words_to_replace_idxs)

            transformed_texts.append(
                current_text.replace_words_at_indices(
                    list(words_to_replace_idxs), [self.single_word] * len(words_to_replace_idxs)
                )
            )

        return transformed_texts


class WordSwapSingleWordToken(textattack.transformations.word_swap.WordSwap):
    """Takes a sentence and transforms it by replacing with a single fixed word.
    """
    single_word: str
    def __init__(self, single_word: str = "?", **kwargs):
        super().__init__(**kwargs)
        self.single_word = single_word

    def _get_replacement_words(self, _word: str) -> List[str]:
        return [self.single_word]