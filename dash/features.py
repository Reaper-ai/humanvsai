import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

def extract_features(text):
    doc = nlp(text)

    tokens = [t for t in doc if not t.is_punct]
    word_len = len(tokens)
    char_len = len(text)
    uniq_words = len(set([t.text.lower() for t in tokens]))

    stop_count = sum(t.is_stop for t in tokens)
    non_stop = word_len - stop_count

    pron = sum(t.pos_ == "PRON" for t in tokens)
    noun = sum(t.pos_ in ("NOUN", "PROPN") for t in tokens)
    verb = sum(t.pos_ == "VERB" for t in tokens)
    adj = sum(t.pos_ == "ADJ" for t in tokens)

    denom = word_len + 1e-6

    features = {
        "word_len": word_len,
        "char_len": char_len,
        "uniq_words": uniq_words,
        "lexical_diversity": uniq_words / denom,
        "word_len_ns": non_stop,
        "stopword_ratio": stop_count / denom,
        "pronoun_ratio": pron / denom,
        "noun_ratio": noun / denom,
        "verb_ratio": verb / denom,
        "adj_ratio": adj / denom,
    }

    return features