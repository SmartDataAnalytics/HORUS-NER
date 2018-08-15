from nltk import word_tokenize, pos_tag, ne_chunk

sentence = "Mark and John are working at Google."

print ne_chunk(pos_tag(word_tokenize(sentence)))

from nltk.chunk import conlltags2tree, tree2conlltags

sentence = "Mark and John are working at Google."
ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))

iob_tagged = tree2conlltags(ne_tree)
print iob_tagged


ne_tree = conlltags2tree(iob_tagged)
print ne_tree
import os
import collections

ner_tags = collections.Counter()

corpus_root = "/media/hady/My Passport/gmb-2.2.0"  # Make sure you set the proper path to the unzipped corpus

for root, dirs, files in os.walk(corpus_root):
    for filename in files:
        if filename.endswith(".tags"):
            with open(os.path.join(root, filename), 'rb') as file_handle:
                file_content = file_handle.read().decode('utf-8').strip()
                annotated_sentences = file_content.split('\n\n')  # Split sentences
                for annotated_sentence in annotated_sentences:
                    annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]  # Split words

                    standard_form_tokens = []

                    for idx, annotated_token in enumerate(annotated_tokens):
                        annotations = annotated_token.split('\t')  # Split annotations
                        word, tag, ner = annotations[0], annotations[1], annotations[3]

                        ner_tags[ner] += 1

print ner_tags

ner_tags = collections.Counter()

for root, dirs, files in os.walk(corpus_root):
    for filename in files:
        if filename.endswith(".tags"):
            with open(os.path.join(root, filename), 'rb') as file_handle:
                file_content = file_handle.read().decode('utf-8').strip()
                annotated_sentences = file_content.split('\n\n')  # Split sentences
                for annotated_sentence in annotated_sentences:
                    annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]  # Split words

                    standard_form_tokens = []

                    for idx, annotated_token in enumerate(annotated_tokens):
                        annotations = annotated_token.split('\t')  # Split annotation
                        word, tag, ner = annotations[0], annotations[1], annotations[3]

                        # Get only the primary category
                        if ner != 'O':
                            ner = ner.split('-')[0]

                        ner_tags[ner] += 1

print ner_tags
# Counter({u'O': 1146068, u'geo': 58388, u'org': 48094, u'per': 44254, u'tim': 34789, u'gpe': 20680, u'art': 867, u'eve': 709, u'nat': 300})

print "Words=", sum(ner_tags.values())
# Words= 1354149

import string
from nltk.stem.snowball import SnowballStemmer

from sklearn.externals import joblib
tf_idf_clone = joblib.load('tf-idf+svm.pkl')
# print model_clone

def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """

    # init the stemmer
    stemmer = SnowballStemmer('english')

    # Pad the sequence with placeholders
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'),
                                                                                    ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)

    # shift the index with 2, to accommodate the padding
    index += 2

    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])

    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase

    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase

    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase

    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,
        'klass': tf_idf_clone.predict([word])[0],

        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
        'next-klass': tf_idf_clone.predict([nextword])[0],

        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
        'next-next-klass': tf_idf_clone.predict([nextnextword])[0],

        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
        'prev-klass': tf_idf_clone.predict([prevword])[0],

        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
        'prev-prev-klass': tf_idf_clone.predict([prevprevword])[0],

        'prev-iob': previob,

        'contains-dash': contains_dash,
        'contains-dot': contains_dot,

        'all-caps': allcaps,
        'capitalized': capitalized,

        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,

        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }


def to_conll_iob(annotated_sentence):
    """
    `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        tag, word, ner = annotated_token

        if ner != 'O':
            if idx == 0:
                ner = "B-" + ner
            elif annotated_sentence[idx - 1][2] == ner:
                ner = "I-" + ner
            else:
                ner = "B-" + ner
        proper_iob_tokens.append((tag, word, ner))
    return proper_iob_tokens


def read_gmb(corpus_root):
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]

                        standard_form_tokens = []

                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            word, tag, ner = annotations[0], annotations[1], annotations[3]

                            if ner != 'O':
                                ner = ner.split('-')[0]

                            if tag in ('LQU', 'RQU'):  # Make it NLTK compatible
                                tag = "``"

                            standard_form_tokens.append((word, tag, ner))

                        conll_tokens = to_conll_iob(standard_form_tokens)

                        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        # Because the classfier expects a tuple as input, first item input, second the class
                        yield [((w, t), iob) for w, t, iob in conll_tokens]


reader = read_gmb(corpus_root)

import pickle
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI


class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)

        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)

    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)

        # Transform the result from [((w1, t1), iob1), ...]
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]

        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)


reader = read_gmb(corpus_root)
data = list(reader)
training_samples = data[:int(len(data) * 0.9)]
test_samples = data[int(len(data) * 0.9):]

print "#training samples = %s" % len(training_samples)  # training samples = 55809
print "#test samples = %s" % len(test_samples)  # test samples = 6201

for iob in test_samples:
    for (w, t) in iob:
        print (w, t, iob)

print pos_tag(word_tokenize("I'm going to Germany this Monday."))

# chunker = NamedEntityChunker(training_samples[:2000])
#
# from nltk import pos_tag, word_tokenize
# print chunker.parse(pos_tag(word_tokenize("I'm going to Germany this Monday.")))
#
# score = chunker.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]])
# print score.accuracy()  , "new_accu"      # 0.931132334092 - Awesome :D
# ner_old = joblib.load('ner_old.pkl')
# # joblib.dump(chunker, 'ner_old.pkl', compress=9)
# score_old  = ner_old.evaluate([conlltags2tree([(w, t, iob) for (w, t), iob in iobs]) for iobs in test_samples[:500]])
#
#
# print score_old.accuracy(), "old_ accu"
#
#
#
# #
# CONST_WIKI_ALL = "/home/hady/PycharmProjects/text-classification-benchmarks/data/ritter_ner.tsv"
#
# # dataset = np.genfromtxt(CONST_WIKI_ALL, delimiter='\t', skip_header=1)
#
# import csv
# X_test_final = []
# y_test_final = []
# with open(CONST_WIKI_ALL,'rb') as tsvin, open('new.csv', 'wb') as csvout:
#     tsvin = csv.reader(tsvin, delimiter='\t')
#     csvout = csv.writer(csvout)
#
#     for row in tsvin:
#         X_test_final.append(pos_tag(word_tokenize(row[0])))
#         y_test_final.append(row[1])
#
# print chunker.parse(X_test_final[0])