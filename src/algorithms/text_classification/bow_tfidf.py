from sklearn.externals import joblib


class BowTfidf(object):
    def __init__(self, config):
        try:
            self.config = config
            self.config.logger.debug('Loading TF-IDF')
            self.tfidf_transformer = joblib.load(config.models_tfidf)
            self.config.logger.debug('Loading OvR models')
            # OvR model
            self.text_checking_model_0 = joblib.load(config.models_0_text)
            # Probabilistic models
            self.config.logger.debug('Loading multiclass models')
            self.text_checking_model_1 = joblib.load(config.models_1_text)
            self.text_checking_model_2 = joblib.load(config.models_2_text)
            self.config.logger.debug('Loading class encoder')
            self.category2idx, self.idx2category = joblib.load(config.categories_encoder)

        except Exception as e:
            raise e

    def detect_text_klass(self, text):
        try:
            features = self.tfidf_transformer.transform([text])
            pred_model_ovr = self.text_checking_model_0.predict(features)[0]

            pred_model_1 = list(self.text_checking_model_1.predict_proba(features)[0])
            idx_k_model1 = pred_model_1.index(max(pred_model_1))
            #k_model1 = self.idx2category[idx_k_model1]

            pred_model_2 = list(self.text_checking_model_2.predict_proba(features)[0])
            idx_k_model2 = pred_model_2.index(max(pred_model_2))
            #k_model2 = self.idx2category[idx_k_model2]

            return pred_model_ovr, idx_k_model1, pred_model_1, idx_k_model2, pred_model_2

        except Exception as e:
            raise e

