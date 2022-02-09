import re
import spacy

from nltk.corpus import stopwords


nlp = spacy.load("en_core_web_sm")

eng_stopwords = stopwords.words('english')

def remove_named_entities_spacy(x: str, mask_token: str = "[MASK]") -> str:
    """
    Replaces named entities in `x` with `mask_token`.
    
    From spacy.io/usage/rule-based-matching/#regex-text:
        nsubj: Nominal subject.
        prep: Preposition.
        pobj: Object of preposition.
        NNP: Proper noun, singular.
        VBD: Verb, past tense.
        IN: Conjunction, subordinating or preposition.
    """
    doc = nlp(x)
    new_tokens = [t.text_with_ws if not t.ent_type_ else ("[MASK]" + t.whitespace_) for t in doc]
    return "".join(new_tokens)

def remove_overlapping_words(t1: str, t2: str, mask_token: str = "[MASK]", case_sensitive=False) -> str:
    """Replaces words in `t1` that occur in `t2` with `mask_token`.

    Ignores english stopwords. If case_sensitive=False, will replace without checking case.
    
    """
    for word in t2.split():
        if word.lower() in eng_stopwords:
            continue

        if case_sensitive:
            t1 = t1.replace(word, mask_token)
        else:
            # stackoverflow.com/questions/919056/case-insensitive-replace
            re_replace = re.compile(re.escape(word.lower()), re.IGNORECASE)
            t1 = re_replace.sub(mask_token, t1)
    return t1

if __name__ == '__main__':
    # print(remove_named_entities_spacy("Apple is looking. And looking. And looking at buying U.K. startup for $1 billion!"))

    data = {'input_text': {'table': {'column_header': ['nationality', 'name', 'article_title', 'occupation', 'birth_date'], 'row_number': [1, 1, 1, 1, 1], 'content': ['german', 'walter extra', 'walter extra\n', 'aircraft designer and manufacturer', '1954']}, 'context': 'walter extra\n'}, 'target_text': 'walter extra is a german award-winning aerobatic pilot , chief aircraft designer and founder of extra flugzeugbau -lrb- extra aircraft construction -rrb- , a manufacturer of aerobatic aircraft .\nextra was trained as a mechanical engineer .\nhe began his flight training in gliders , transitioning to powered aircraft to perform aerobatics .\nhe built and flew a pitts special aircraft and later built his own extra ea-230 .\nextra began designing aircraft after competing in the 1982 world aerobatic championships .\nhis aircraft constructions revolutionized the aerobatics flying scene and still dominate world competitions .\nthe german pilot klaus schrodt won his world championship title flying an aircraft made by the extra firm .\nwalter extra has designed a series of performance aircraft which include unlimited aerobatic aircraft and turboprop transports .\n'}
    def map_ex(ex):
            """
            transforms wiki_bio example into (text1, text2) pair

            >>> ex['target_text']
            'walter extra is a german award-winning aerobatic pilot , chief aircraft designer and founder of extra....
            >>> ex['input_text']
            {'table': {'column_header': ['nationality', 'name', 'article_title', 'occupation', 'birth_date'], 'row_number': [1, 1, 1, 1, 1], 'content': ['german', 'walter extra', 'walter extra\n', 'aircraft designer and manufacturer', '1954']}, 'context': 'walter extra\n'}
            """
            # transform table to str
            table_info = ex['input_text']['table']
            table_rows = list(zip(
                map(lambda s: s.strip(), table_info['column_header']),
                map(lambda s: s.strip(), table_info['content']))
            )
            table_text = '\n'.join([' | '.join(row) for row in table_rows])
            # return example: transformed table + first paragraph
            return {
                'text1': ex['target_text'],     # First paragraph of biography
                'text2': table_text,            # Table re-printed as a string
            }
    ex = map_ex(data)
    print(ex['text1'])
    print()
    print(ex['text2'])
    print()
    print(remove_overlapping_words(ex['text1'], ex['text2']))