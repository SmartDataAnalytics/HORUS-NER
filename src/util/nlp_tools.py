# -*- coding: utf-8 -*-

import gensim
import nltk
#from nltk.tag.stanford import CoreNLPNERTagger, CoreNLPPOSTagger
import unicodedata

from src.util import definitions, CMUTweetTagger
from src.config import HorusConfig

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class NLPTools(object):
    __metaclass__ = Singleton

    def tokenize_and_pos_nltk(self, text):
        if isinstance(text, str):
            text=text.decode('utf8')
        # TODO: NLTK (ne_chunk) can't deal with unicode -> https://www.nltk.org/api/nltk.chunk.html (pre does not include Unicode support, so this module will not work with unicode strings)
        text=unicodedata.normalize('NFKC', text).encode('ascii', 'replace')
        if type(text) is not list:
            text=text.decode('utf8')
            tokens = nltk.word_tokenize(text)

        #sd = [s.decode('utf8') for s in tokens]
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

    #def annotate_ner_stanford(self, text):
    #    return self.stanford_ner.tag(text.split())

    #def tokenize_and_pos_stanford(self, text):
    #    t = text.decode('utf-8')
    #    #sd = [s.decode('utf8') for s in text.split()]
    #    return t.split(), self.stanford_pos.tag(t.split()), []

    def tokenize_and_pos_twitter(self, text):
        tokens = []
        tagged = []
        pos_universal = []
        sd = [s.decode('utf8') for s in text.split()]
        #[w.decode('utf8').encode(encoding='ascii', errors='replace') for w in text.split()]
        pos_token_tag_sentence = CMUTweetTagger.runtagger_parse(sd)

        for sequence_tag in pos_token_tag_sentence:
            for token_tag in sequence_tag:
                tokens.append(token_tag[0])
                tagged.append(token_tag[1])
                pos_universal.append(self.convert_penn_to_universal_tags(token_tag[1]))
        return tokens, zip(tokens, tagged), zip(tokens, pos_universal)

    #TODO: is it really necessary? I just changed the other adding "split" right before...no need to [] before
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

    def __init__(self, config):
        self.config = config
        #self.stanford_ner = CoreNLPNERTagger(self.config.model_stanford_filename_ner, self.config.model_stanford_path_jar_ner)
        #self.stanford_pos = CoreNLPPOSTagger(self.config.model_stanford_filename_pos, self.config.model_stanford_path_jar_pos)
        #self.stanford_pos.java_options='-mx8g'