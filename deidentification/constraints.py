from typing import Set

from textattack.constraints import PreTransformationConstraint


class CertainWordsModification(PreTransformationConstraint):
    """Constraint to modify certain words. This prevents us from modifying any words that are 'MASK' 
    in the event we're re-masking some already-masked text."""
    certain_words: Set[str]
    def __init__(self, certain_words: Set[str]):
        self.certain_words = certain_words

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in current_text which are able to be
        deleted."""
        matching_word_idxs = {
            i for i, word in enumerate(current_text.words) if word in self.certain_words
        }
        try:
            return (
                set(range(len(current_text.words)))
                - matching_word_idxs
            )
        except KeyError:
            raise KeyError(
                "`modified_indices` in attack_attrs required for RepeatModification constraint."
            )