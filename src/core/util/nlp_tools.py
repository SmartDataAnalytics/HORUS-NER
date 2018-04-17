import gensim
import nltk
from nltk.tag import StanfordPOSTagger
from nltk.tag.stanford import StanfordNERTagger
from src.core.util import definitions, CMUTweetTagger
from src.config import HorusConfig


class NLPTools(object):

    def tokenize_and_pos_nltk(self, text):
        #TODO: esta trocando '' por ``
        tokens = text
        if type(text) is not list:
            tokens = nltk.word_tokenize(text.decode('utf-8'))

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

    def annotate_ner_stanford(self, text):
        return self.stanford_ner.tag(text.split())

    def tokenize_and_pos_stanford(self, text):
        t = text.decode('utf-8')
        #sd = [s.decode('utf8') for s in text.split()]
        return t.split(), self.stanford_pos.tag(t.split()), []

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

    def __init__(self):
        config = HorusConfig()
        self.stanford_ner = StanfordNERTagger(self.config.model_stanford_filename_ner, self.config.model_stanford_path_jar_ner)
        self.stanford_pos = StanfordPOSTagger(self.config.model_stanford_filename_pos, self.config.model_stanford_path_jar_pos)
        self.stanford_pos.java_options='-mx8g'
        self.word2vec_google = gensim.models.KeyedVectors.load_word2vec_format(config.embeddings_path, binary=True)
