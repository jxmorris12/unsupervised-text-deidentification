from typing import List, Tuple

import re

import torch
import transformers


def find_row_from_key(table_rows: List[str], key: str) -> Tuple[str, str]:
    """Finds row in wikipedia infobox by a key

    Raises: AssertionError if key not found
    """
    matching_rows = [r for r in table_rows if r[0] == key]
    assert len(matching_rows) > 0, f"name not found in table_rows: {table_rows}"
    return matching_rows[0]


def name_from_table_rows(table_rows: List[str]) -> str:
    """gets person's name from rows of a Wikipedia infobox"""
    try:
        row = find_row_from_key(table_rows, key='name')
    except AssertionError:
        # a few articles have this key instead of a 'name' key
        row = find_row_from_key(table_rows, key='article_title')
    name = row[1]
    # capitalize name and return
    return ' '.join((word.capitalize() for word in name.split()))

words_from_text_re = re.compile(r'\b\w+\b')
def words_from_text(s: str):
    return words_from_text_re.findall(s)

def redact_text_from_grad(
    input_ids: torch.Tensor, model: transformers.PreTrainedModel, k: int, mask_token_id: int) -> torch.Tensor:
    """Masks tokens in `input_ids` proportional to gradient."""
    topk_tokens = (
        model.embeddings.word_embeddings.weight.grad.norm(p=2, dim=1).argsort()
    )
    special_tokens_mask = (
        (topk_tokens == 0) | (topk_tokens == 100) | (topk_tokens == 101) | (topk_tokens == 102) | (topk_tokens == 103)
    )
    topk_tokens = topk_tokens[~special_tokens_mask][-k:]
    topk_mask = (
        input_ids[..., None].to(topk_tokens.device) == topk_tokens[None, :]).any(dim=-1)

    return (
        topk_tokens, torch.where(
            topk_mask,
            torch.tensor(mask_token_id)[None, None].to(topk_tokens.device),
            input_ids.to(topk_tokens.device)
        )
    )