from src import definitions
from src.utils.nlp_tools import NLPTools
import html

def word_level_tokenization_to_preserve_labels(sentence_conll: [], labels_conll: [], tools: NLPTools):
    '''
    Gets a sentence with tags in CONLL-like, re-builds it, tokenize to apply POS tagger
    and returns everything, so that we avoid tokenization mismatch with the original
    tokenization (conll). This is helpful in case we do not know the exact tokenizer used
    to build the conll file. 2 Approaches

    (1) original CoNLL file has len(tokens) == len(tokens) used tokenizer
        - nothing to do, perfect scenario. Unlike to happen for the whole dataset though, unless you know a priori
        which tokenizer has been used to create the CoNLL file.

    (2) original CoNLL file has len(tokens) < len(tokens) used tokenizer
        - duplicate the labels from original CoNLL to the merged token from the new tokenizer, if that is the case.

    (3) original CoNLL file has len(tokens) > len(tokens) used tokenizer
        - are the subset of tokens from the original CoNLL file the same?
            3.1) YES: no problem, merge into one single token for the new tokenizer.
            3.2) NO: we have a problem...

    To avoid 3.2 some people use the word-level tokenization, so that you always have list >= original CoNLL. Thus,
    having a perfect label alignment.
    The problem with this approach is that POS tag labels are likely to not be the same since the whole sentence
    is not considered, but rather a unique token per process (i.e., word-level tokenization).


    :param sentence_conll:
    :param labels_conll:
    :param tokenizer:
    :return:
    '''
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence_conll, labels_conll):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tools.tokenize_and_pos_twitter(word)[0]
        #tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def fully_unescape_token(text):
    out1 = html.unescape(text)
    out2 = html.unescape(out1)
    if out1 == out2:
        return out2
    else:
        return fully_unescape_token(out2)
