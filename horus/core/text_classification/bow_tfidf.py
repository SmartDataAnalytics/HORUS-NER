
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer

from horus.core import translation
from horus.core.config import HorusConfig
from horus.core.translation.util import translate


class BowTfidf():
    def __init__(self):
        self.config = HorusConfig()
        self.text_checking_model_1 = joblib.load(self.config.models_1_text)
        self.text_checking_model_2 = joblib.load(self.config.models_2_text)
        self.text_checking_model_3 = joblib.load(self.config.models_3_text)
        self.text_checking_model_4 = joblib.load(self.config.models_4_text)
        self.text_checking_model_5 = joblib.load(self.config.models_5_text)
        self.tfidf_transformer = TfidfTransformer()

    def detect_text_klass(self, t1, t2, id, t1en, t2en):

        try:

            t1final, t2final = translate(t1, t2, id, t1en, t2en)
            if t1final is False:
                return t2final  # error vector
            else:
                docs = ["{} {}".format(t1final.encode("utf-8"), t2final.encode("utf-8"))]
                if self.config.text_classification_type == 0:  # TFIDF
                    predictions = [self.text_checking_model_1.predict(docs)[0],
                                   self.text_checking_model_2.predict(docs)[0],
                                   self.text_checking_model_3.predict(docs)[0],
                                   self.text_checking_model_4.predict(docs)[0],
                                   self.text_checking_model_5.predict(docs)[0]]
                elif self.config.text_classification_type == 1:  # TopicModeling
                    dict = self.classifier_tm.score(docs)
                    predictions = []
                    predictions.append(dict.get('loc'))
                    predictions.append(dict.get('per'))
                    predictions.append(dict.get('org'))
                    predictions.append(0)
                    predictions.append(0)
                else:
                    raise Exception('parameter value not implemented: ' + str(self.config.object_detection_type))

        except Exception as e:
            self.sys.log.error(':: Error: ' + str(e))
            predictions = [-1, -1, -1, -1, -1]

        return predictions

