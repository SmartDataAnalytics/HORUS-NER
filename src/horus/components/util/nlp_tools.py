import nltk
from nltk import word_tokenize
from nltk.tag.stanford import StanfordNERTagger
from nltk.tag import StanfordPOSTagger

from horus import definitions
from horus.postagger import CMUTweetTagger
#text = 'Diego Esteves works at Microsoft'
#tag1 = st.tag(text.split())


class NLPTools(object):

    def tokenize_and_pos_nltk(self, text):
        tokens = text
        if type(text) is not list:
            tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        return tokens, tagged, nltk.pos_tag(tokens, tagset="universal")

    def annotate_ner_nltk(self, tagged):
        t = nltk.ne_chunk(tagged, binary=False)
        x = nltk.tree2conllstr(t)
        x = x.split('\n')
        ret = []
        for xi in x:
            ret.append(xi.split(' ')[2])
        return ret


    def annotate_ner_stanford(self, text):
        return self.stanford_ner.tag(text.split())

    def tokenize_and_pos_stanford(self, text):
        tokens = text.split()
        return tokens, self.stanford_pos.tag(tokens), []

    def tokenize_and_pos_twitter(self, text):
        tokens = []
        tagged = []
        pos_universal = []
        pos_token_tag_sentence = CMUTweetTagger.runtagger_parse(text)

        for sequence_tag in pos_token_tag_sentence:
            for token_tag in sequence_tag:
                tokens.append(token_tag[0])
                tagged.append(token_tag[1])
                pos_universal.append(self.convert_penn_to_universal_tags(token_tag[1]))
        return tokens, zip(tokens, tagged), zip(tokens, pos_universal)

    def tokenize_and_pos_twitter_list(self, text):
        token_list = []
        tagged_list = []
        pos_universal_list = []
        pos_token_tag_sentence = CMUTweetTagger.runtagger_parse(text)

        for sequence_tag in pos_token_tag_sentence:
            tokens = []
            tagged = []
            pos_universal = []
            for token_tag in sequence_tag:
                tokens.append(token_tag[0])
                tagged.append(token_tag[1])
                pos_universal.append(self.convert_penn_to_universal_tags(token_tag[1]))

            token_list.append(tokens)
            tagged_list.append(zip(tokens, tagged))
            pos_universal_list.append(zip(tokens, pos_universal))
        return token_list, tagged_list, pos_universal_list

    @staticmethod
    def convert_cmu_to_universal_tags(cmu_tag):
        for item in definitions.CMU_UNI_TAGS:
            if item[0] == cmu_tag:
                return item[1]
        return "X"

    @staticmethod
    def convert_penn_to_universal_tags(penn_tag):
        for item in definitions.PENN_UNI_TAG:
            if item[0] == penn_tag:
                return item[1]
        return penn_tag

    def __init__(self):
        self.stanford_ner = StanfordNERTagger(
            '/Users/esteves/Github/horus-models/src/horus/resource/models/stanford/english.all.3class.distsim.crf.ser.gz',
            '/Users/esteves/Github/horus-models/src/horus/resource/models/stanford/stanford-ner.jar')
        self.stanford_pos = StanfordPOSTagger(
            '/Users/esteves/Github/horus-models/src/horus/resource/models/stanford/2015-04-20/english-bidirectional-distsim.tagger',
            '/Users/esteves/Github/horus-models/src/horus/resource/models/stanford/2015-04-20/stanford-postagger.jar')
        self.stanford_pos.java_options='-mx4096m'
