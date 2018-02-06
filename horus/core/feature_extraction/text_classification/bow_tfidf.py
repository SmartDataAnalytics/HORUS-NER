
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from horus.core.config import HorusConfig



class BowTfidf():
    def __init__(self):
        self.config = HorusConfig()
        self.text_checking_model_1 = joblib.load(self.config.models_1_text)
        self.text_checking_model_2 = joblib.load(self.config.models_2_text)
        self.text_checking_model_3 = joblib.load(self.config.models_3_text)
        self.text_checking_model_4 = joblib.load(self.config.models_4_text)
        self.text_checking_model_5 = joblib.load(self.config.models_5_text)
        self.tfidf_transformer = TfidfTransformer()

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
