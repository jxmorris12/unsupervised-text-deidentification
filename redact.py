import spacy

nlp = spacy.load("en_core_web_sm")
def remove_named_entities_spacy(x: str, mask_token: str = "[MASK]") -> str:
    """
    
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

if __name__ == '__main__':
    print(remove_named_entities_spacy("Apple is looking. And looking. And looking at buying U.K. startup for $1 billion!"))