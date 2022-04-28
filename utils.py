from typing import Dict, List, Tuple

import collections
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


def get_table_minus_name(table_rows: List[str]) -> List[str]:
    """gets person's name from rows of a Wikipedia infobox"""
    out_rows = []
    for row in table_rows:
        if row[0] in ['name', 'article_title']: continue
        out_rows.append(row)
    return out_rows


words_from_text_re = re.compile(r'\b\w+\b')
def word_start_and_end_idxs_from_text(s: str) -> List[Tuple[int, int]]:
    assert isinstance(s, str)
    return [(m.start(0), m.end(0)) for m in words_from_text_re.finditer(s)]


def words_from_text(s: str) -> List[str]:
    assert isinstance(s, str)
    return words_from_text_re.findall(s)


def create_document_and_profile_from_wikibio(ex: Dict) -> Dict:
    """
    transforms wiki_bio example into (document, profile) pair

    >>> ex['target_text']
    'walter extra is a german award-winning aerobatic pilot , chief aircraft designer and founder of extra....
    >>> ex['input_text']
    {'table': {'column_header': ['nationality', 'name', 'article_title', 'occupation', 'birth_date'], 'row_number': [1, 1, 1, 1, 1], 'content': ['german', 'walter extra', 'walter extra\n', 'aircraft designer and manufacturer', '1954']}, 'context': 'walter extra\n'}
    """
    # replace weird textual artifacts: -lrb- with ( and -rrb- with )
    fixed_target_text = ex['target_text'].replace('-lrb- ', '(').replace(' -rrb-', ')')
    # transform table to str
    table_info = ex['input_text']['table']
    table_rows = list(zip(
        map(lambda s: s.strip(), table_info['column_header']),
        map(lambda s: s.strip(), table_info['content']))
    )
    table_text = (
        '\n'.join([' | '.join(row) for row in table_rows])
    )
    # table_text_without_name = (
    #     '\n'.join([' | '.join(row) for row in get_table_minus_name(table_rows)])
    # )
    # also add profile col and row
    profile_keys = table_info['column_header']
    profile_values = table_info['content']
    # return example: transformed table + first paragraph
    return {
        'name': name_from_table_rows(table_rows),
        'document': fixed_target_text,                      # First paragraph of biography
        'profile': table_text,                              # Table re-printed as a string
        # 'profile_without_name': table_text_without_name,    # Table with name removed
        'profile_keys': '||'.join(profile_keys),             # Keys in profile box
        'profile_values': '||'.join(profile_values),         # Values in profile box
        'text_key': ex['target_text'] + ' ' + table_text,   # store (document, profile) str key
    }