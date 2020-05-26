from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer


class BowTfidf(object):
    def __init__(self, config):
        try:
            self.config = config
            self.config.logger.debug('loading TF-IDF')
            self.tfidf_transformer = joblib.load(config.models_0_text)
            self.text_checking_model_1 = joblib.load(config.models_1_text)
            self.text_checking_model_2 = joblib.load(config.models_2_text)

        except Exception as e:
            raise e

    def detect_text_klass(self, text):
        try:
            features = self.tfidf_transformer.transform([text])
            predictions = []
            predictions.extend(self.text_checking_model_1.predict(features)[0])
            predictions.extend(self.text_checking_model_2.predict(features)[0])
            return predictions
        except Exception as e:
            raise e
