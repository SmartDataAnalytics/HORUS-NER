
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer

class BowTfidf():
    def __init__(self, config):
        try:
            self.config = config
            self.config.logger.debug('loading TF-IDF')
            self.text_checking_model_1 = joblib.load(config.models_1_text)
            self.text_checking_model_2 = joblib.load(config.models_2_text)
            self.text_checking_model_3 = joblib.load(config.models_3_text)
            self.text_checking_model_4 = joblib.load(config.models_4_text)
            self.text_checking_model_5 = joblib.load(config.models_5_text)
            #self.tfidf_transformer = TfidfTransformer()
        except Exception as e:
            raise e

    def detect_text_klass(self, text):
        try:
            predictions = [self.text_checking_model_1.predict(text)[0],
                        self.text_checking_model_2.predict(text)[0],
                        self.text_checking_model_3.predict(text)[0],
                        self.text_checking_model_4.predict(text)[0],
                        self.text_checking_model_5.predict(text)[0]]

            return predictions
        except Exception as e:
            raise e
