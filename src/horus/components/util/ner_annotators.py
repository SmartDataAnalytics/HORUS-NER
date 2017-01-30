from nltk import word_tokenize
from nltk.tag.stanford import StanfordNERTagger

#text = 'Diego Esteves works at Microsoft'
#tag1 = st.tag(text.split())


class NERAnnotators(object):

    def annotate(self, text, annotator):
        if annotator == 'stanford':
            return self.stanford_ner.st.tag(word_tokenize(text))
        else:
            return "not implemented"

    def __init__(self):
        self.stanford_ner = StanfordNERTagger(
            '/Users/esteves/Github/horus-models/src/horus/resource/models/stanford/english.all.3class.distsim.crf.ser.gz',
            '/Users/esteves/Github/horus-models/src/horus/resource/models/stanford/stanford-ner.jar')


