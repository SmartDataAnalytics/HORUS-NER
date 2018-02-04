import shorttext
import en_core_web_sm
import os

from horus.core.config import HorusConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
nlp = en_core_web_sm.load()

class TopicModeling():
    def __init__(self):
        self.config = HorusConfig()
        self.wvmodel = shorttext.utils.load_word2vec_model(self.config.embeddings_path)
        self.classifier_tm = shorttext.classifiers.load_varnnlibvec_classifier(self.wvmodel,
                                                                               self.config.models_1_text_cnn)
