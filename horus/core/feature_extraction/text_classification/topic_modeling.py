import shorttext
import en_core_web_sm
import os

from horus.core.util.systemlog import SysLogger


class TopicModeling():
    def __init__(self, config):
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            #en_core_web_sm.load()
            self.wvmodel = shorttext.utils.load_word2vec_model(config.embeddings_path)
            self.classifier_tm = shorttext.classifiers.load_varnnlibvec_classifier(self.wvmodel, config.models_1_text_cnn)
        except Exception as e:
            raise e

    def detect_text_klass(self, text):
        predictions = []
        try:
            dict = self.classifier_tm.score(text)
            predictions.append(dict.get('loc'))
            predictions.append(dict.get('org'))
            predictions.append(dict.get('per'))
            predictions.append(0)
            predictions.append(0)
            return predictions
        except Exception as e:
            raise e