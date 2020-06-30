import heapq
from abc import ABC, abstractmethod, ABCMeta
import gensim as gensim
from src import definitions
from src.algorithms.computer_vision.cls_dlib import DLib_Classifier
from src.algorithms.computer_vision.cnnlogo import CNNLogo
from src.algorithms.computer_vision.places365 import Places365CV
from src.algorithms.computer_vision.sift import SIFT
from src.algorithms.text_classification.bow_tfidf import BowTfidf
from src.algorithms.text_classification.topic_modeling_short_cnn import TopicModelingShortCNN
from src.horus_meta import Horus, WordFeaturesInterface, HorusToken
from nltk import LancasterStemmer, re, WordNetLemmatizer
from nltk import WordNetLemmatizer, SnowballStemmer
import numpy as np
from functools import lru_cache
from config import HorusConfig
from src import definitions
from src.definitions import encoder_le1_name, PRE_PROCESSING_STATUS
from src.horus_meta import Horus, HorusDataLoader, WordFeaturesInterface
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import sqlite3
from sklearn import preprocessing
from sklearn.preprocessing import normalize

from src.utils.definitions_sql import SQL_TEXT_CLASS_SEL, SQL_TEXT_CLASS_UPD
from src.utils.nlp_tools import NLPTools
from src.utils.translation.azure import bing_detect_language, bing_translate_text
from src.utils.translation.bingtranslation import BingTranslator
from src.utils.util import Util
# from tensorflow.python.keras._impl.keras.applications import InceptionV3
# from tensorflow.python.keras._impl.keras.applications import InceptionV3
from tensorflow.python.keras.preprocessing import image as ppimg
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from src.algorithms.computer_vision.cls_dlib import DLib_Classifier
from src.algorithms.computer_vision.cnnlogo import CNNLogo
from src.algorithms.computer_vision.inception import InceptionCV


class HorusFeatureExtractor(metaclass=ABCMeta):

    def __init__(self, config):
        self.config = config
        self.util = Util(config)
        self.tools = NLPTools(config)
        self.config.logger.info('Database ping')
        self.conn = sqlite3.connect(self.config.database_db)

    @abstractmethod
    def extract_features(self, horus: Horus) -> bool:
        pass


class HorusExtractorImage(HorusFeatureExtractor):

    def __init__(self, config: HorusConfig):
        super().__init__(config)
        self.image_sift = SIFT(self.config)
        self.image_cnn_placesCNN = Places365CV(self.config)
        # self.image_cnn_incep_model = InceptionCV(self.config, version='V3')
        self.image_cnn_incep_model = InceptionV3(weights='imagenet')
        self.image_cnn_logo = CNNLogo(self.config)
        self.dlib_cnn = DLib_Classifier(self.config)

    def extract_features(self, horus: Horus) -> bool:
        try:
            cv_dict, cv_dict_reversed = WordFeaturesInterface.get_visual()

            i_sent = 0
            for sentence in horus.sentences:
                i_sent += 1
                i_token = 0
                for token in sentence.tokens:
                    i_token += 1
                    if token.label_pos in definitions.POS_NOUN_TAGS or token.is_compound == 1:
                        self.config.logger.debug(f'token: {token.text} ({i_token}/{len(sentence.tokens)}) | '
                                                 f'sentence ({i_sent}/{len(horus.sentences)})')

                        id_term_txt = token.features.text.values[cv_dict_reversed.get('id_db')]
                        id_ner_type = 0

        except:
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.conn.close()
            del self.image_cnn_incep_model
            del self.image_cnn_placesCNN
        except:
            pass


class HorusExtractorText(HorusFeatureExtractor):
    def __init__(self, config: HorusConfig):
        super().__init__(config)
        self.translator = BingTranslator(self.config)
        self.text_bow = BowTfidf(self.config)
        self.config.logger.info('Loading embeddings')
        self.word2vec_google = gensim.models.KeyedVectors.load_word2vec_format(self.config.embeddings_path, binary=True)
        self.config.logger.info('Loading topic modeling')
        self.text_tm = TopicModelingShortCNN(self.config, w2v=self.word2vec_google, mode='test')
        self.extended_seeds_PER = []
        self.extended_seeds_ORG = []
        self.extended_seeds_LOC = []
        self.extended_seeds_NONE = []
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.config.logger.info('Setting seeds ')
        self.__set_str_extended_seeds()

    def __get_translated_text(self, id):
        try:
            c = self.conn.cursor()

        except Exception as e:
            raise e

    def __detect_lang_and_translate(self, t1, t2, id, t1en, t2en):
        try:
            error = 0
            c = self.conn.cursor()
            if t1en is None or t1en == '':
                try:
                    lt1 = bing_detect_language(t1, self.config.translation_secret)
                    if lt1 != 'en':
                        temp = bing_translate_text(t1, 'en', self.config.translation_secret)
                    else:
                        temp = t1
                    sql = """UPDATE HORUS_SEARCH_RESULT_TEXT SET result_title_en = ? WHERE id = ?"""
                    c.execute(sql, (temp, id))
                    t1en = temp
                except Exception as e:
                    sql = """UPDATE HORUS_SEARCH_RESULT_TEXT SET error = 1, error_desc = ? WHERE id = ?"""
                    c.execute(sql, (str(e.message), id))
                    error += 1

            if t2en is None or t2en == '':
                try:
                    lt2 = bing_detect_language(t2, self.config.translation_secret)
                    if lt2 != 'en':
                        temp = bing_translate_text(t2, 'en', self.config.translation_secret)
                    else:
                        temp = t2
                    sql = """UPDATE HORUS_SEARCH_RESULT_TEXT SET result_description_en = ? WHERE id = ?"""
                    c.execute(sql, (temp, id))  # .encode("utf-8")
                    t2en = temp
                except Exception as e:
                    sql = """UPDATE HORUS_SEARCH_RESULT_TEXT SET error = 1, error_desc = ? WHERE id = ?"""
                    c.execute(sql, (str(e.message), id))
                    error += 1

            c.close()

            merged = ''
            if t1en is not None:
                merged = t1en.encode('ascii', 'ignore')
            if t2en is not None:
                merged = merged + ' ' + t2en.encode('ascii', 'ignore')

            return merged, error

        except Exception as e:
            raise e

    def __get_number_classes_in_embeedings(self, w):
        '''
        :param w: an input word
        :return:
        '''
        if self.word2vec_google is None:
            return np.array([0.0, 0.0, 0.0, 0.0]), None

        try:
            out = []

            try:
                most_similar_to_w = self.word2vec_google.most_similar(positive=w, topn=5)
            except KeyError:
                return np.array([0.0, 0.0, 0.0, 0.0]), None

            for w, prob in most_similar_to_w:
                aL, aO, aP, aN = [], [], [], []
                for z in definitions.seeds_dict_topics['loc'][0:9]:
                    try:
                        aL.append(self.word2vec_google.similarity(w, z) * prob)
                    except KeyError:
                        continue

                for z in definitions.seeds_dict_topics['org'][0:9]:
                    try:
                        aO.append(self.word2vec_google.similarity(w, z) * prob)
                    except KeyError:
                        continue

                for z in definitions.seeds_dict_topics['per'][0:9]:
                    try:
                        aP.append(self.word2vec_google.similarity(w, z) * prob)
                    except KeyError:
                        continue

                for z in definitions.seeds_dict_topics['none'][0:9]:
                    try:
                        aN.append(self.word2vec_google.similarity(w, z) * prob)
                    except KeyError:
                        continue

                out.append([np.average(np.array(aL)), np.average(np.array(aO)),
                            np.average(np.array(aP)), np.average(np.array(aN))])

            return np.average(np.array(out), axis=0), most_similar_to_w

        except:
            raise

    def __set_str_extended_seeds(self):
        try:
            if not definitions.seeds_dict_img_classes:
                raise ('image seeds is empty!')

            self.config.logger.debug('Extending seeds')
            for k, V in definitions.seeds_dict_img_classes.items():
                for v in V:
                    for i, syns in enumerate(wn.synsets(v)):
                        # print(' hypernyms = ', ' '.join(list(chain(*[l.lemma_names() for l in syns.hypernyms()]))))
                        # print(' hyponyms = ', ' '.join(list(chain(*[l.lemma_names() for l in syns.hyponyms()]))))
                        _s = wn.synset_from_pos_and_offset(syns.pos(), syns.offset())
                        if k == 'per':
                            self.extended_seeds_PER.append(str(_s._name).split('.')[0])
                            # self.extended_seeds_PER.extend([str(_s._name).split('.')[0] for _s in syns.hypernyms()])
                            # ids_PER.extend(syns.hyponyms())
                        elif k == 'org':
                            self.extended_seeds_ORG.append(str(_s._name).split('.')[0])
                            # self.extended_seeds_ORG.extend([str(_s._name).split('.')[0] for _s in syns.hypernyms()])
                            # ids_ORG.extend(syns.hyponyms())
                        elif k == 'loc':
                            self.extended_seeds_LOC.append(str(_s._name).split('.')[0])
                            # self.extended_seeds_LOC.extend([str(_s._name).split('.')[0] for _s in syns.hypernyms()])
                            # ids_LOC.extend(syns.hyponyms())
                        elif k == 'none':
                            self.extended_seeds_NONE.append(str(_s._name).split('.')[0])
                            # self.extended_seeds_NONE.extend([str(_s._name).split('.')[0] for _s in syns.hypernyms()])
                        else:
                            raise ('key error')

            self.extended_seeds_LOC = set(self.extended_seeds_LOC)
            self.extended_seeds_ORG = set(self.extended_seeds_ORG)
            self.extended_seeds_PER = set(self.extended_seeds_PER)
            self.extended_seeds_NONE = set(self.extended_seeds_NONE)

            self.config.logger.debug('done!')
            self.config.logger.debug('LOC seeds: ' + ','.join(self.extended_seeds_LOC))
            self.config.logger.debug('ORG seeds: ' + ','.join(self.extended_seeds_ORG))
            self.config.logger.debug('PER seeds: ' + ','.join(self.extended_seeds_PER))
            self.config.logger.debug('NON seeds: ' + ','.join(self.extended_seeds_NONE))

        except:
            raise

    def __get_basic_stats(self, vec: np.array()) -> []:

        _sum = [np.sum(vec[:, 0]), np.sum(vec[:, 1]), np.sum(vec[:, 2]), np.sum(vec[:, 3])]
        _avg = [np.average(vec[:, 0]), np.average(vec[:, 1]), np.average(vec[:, 2]), np.average(vec[:, 3])]
        _max = [np.max(vec[:, 0]), np.max(vec[:, 1]), np.max(vec[:, 2]), np.max(vec[:, 3])]
        _min = [np.min(vec[:, 0]), np.min(vec[:, 1]), np.min(vec[:, 2]), np.min(vec[:, 3])]

        return _sum, _avg, _max, _min

    def __set_token_statistics(self, token: HorusToken, y_bow: np.array, y_tm: np.array, limit_txt: int,
                               nr_results_txt: int, tx_dict: dict, tx_dict_rev: dict):
        try:

            self.config.logger.info("token statistics")

            # -- TOKEN STATISTICS --
            tot_error_translation = 0
            klass_top = []
            tm_cnn_w = []

            embs, top5_sim = self.__get_number_classes_in_embeedings(token.text)
            if self.text_tm.wvmodel is not None:
                # TM+CNN - term
                tm_cnn_w = self.text_tm.detect_text_klass(token.text)
                tm_cnn_w_exp = [np.math.pow(i, 2) for i in tm_cnn_w]
                klass_top.append(tm_cnn_w_exp)
            if self.text_tm.wvmodel is not None and top5_sim is not None:
                for top in top5_sim:
                    klass_top.append(self.text_tm.detect_text_klass(top[0]))
            else:
                klass_top = [[np.math.pow(10, -5)] * 5] * 5
            #klass_top.append(tm_cnn_w_exp)


            # get OvR predictions
            yyb = np.array(y_bow[:, 0])
            # get probs model 1
            yym1 = np.array(y_bow[:, 1])
            # get probs model 2
            yym2 = np.array(y_bow[:, 2])
            # top 5 most similar predictions
            klass_top = np.array(klass_top)

            gpb = [np.count_nonzero(yyb == self.text_bow.category2idx['PER']),
                   np.count_nonzero(yyb == self.text_bow.category2idx['ORG']),
                   np.count_nonzero(yyb == self.text_bow.category2idx['LOC']),
                   np.count_nonzero(yyb == self.text_bow.category2idx['MISC'])]

            tm_k_top_sum, tm_k_top_avg, tm_k_top_max, tm_k_top_min = self.__get_basic_stats(klass_top)

            topic_sums, topic_avg, topic_max, topic_min = self.__get_basic_stats(y_tm)

            # note that encoders for text and cv might have a different klass id from the NER klass id
            # we should always interchange per label instead of klass id in this particular case
            horus_tx_ner_label = self.text_bow.idx2category(gpb.index(max(gpb)))

            avg_probs_model1 = np.average(yym1)
            avg_probs_model2 = np.average(yym2)

            token.features.text.values[tx_dict_rev.get('total.retrieved.results.search_engine')] = limit_txt
            token.features.text.values[tx_dict_rev.get('total.error.translation')] = tot_error_translation

            token.features.text.values[tx_dict_rev.get('total.ovr.k.loc')] = gpb[self.text_bow.category2idx['LOC']]
            token.features.text.values[tx_dict_rev.get('total.ovr.k.org')] = gpb[self.text_bow.category2idx['ORG']]
            token.features.text.values[tx_dict_rev.get('total.ovr.k.per')] = gpb[self.text_bow.category2idx['PER']]
            token.features.text.values[tx_dict_rev.get('total.ovr.k.misc')] = gpb[self.text_bow.category2idx['MISC']]

            token.features.text.values[tx_dict_rev.get('avg.probs1.k.loc')] = avg_probs_model1[
                self.text_bow.category2idx['LOC']]
            token.features.text.values[tx_dict_rev.get('avg.probs1.k.org')] = avg_probs_model1[
                self.text_bow.category2idx['ORG']]
            token.features.text.values[tx_dict_rev.get('avg.probs1.k.per')] = avg_probs_model1[
                self.text_bow.category2idx['PER']]
            token.features.text.values[tx_dict_rev.get('avg.probs1.k.misc')] = avg_probs_model1[
                self.text_bow.category2idx['MISC']]

            token.features.text.values[tx_dict_rev.get('avg.probs2.k.per')] = avg_probs_model2[
                self.text_bow.category2idx['PER']]
            token.features.text.values[tx_dict_rev.get('avg.probs2.k.org')] = avg_probs_model2[
                self.text_bow.category2idx['ORG']]
            token.features.text.values[tx_dict_rev.get('avg.probs2.k.loc')] = avg_probs_model2[
                self.text_bow.category2idx['LOC']]
            token.features.text.values[tx_dict_rev.get('avg.probs2.k.misc')] = avg_probs_model2[
                self.text_bow.category2idx['MISC']]

            token.features.text.values[tx_dict_rev.get('total.topic.k.loc')] = 0 if len(tm_cnn_w) == 0 else tm_cnn_w[0]
            token.features.text.values[tx_dict_rev.get('total.topic.k.org')] = 0 if len(tm_cnn_w) == 0 else tm_cnn_w[1]
            token.features.text.values[tx_dict_rev.get('total.topic.k.per')] = 0 if len(tm_cnn_w) == 0 else tm_cnn_w[2]
            token.features.text.values[tx_dict_rev.get('total.topic.k.misc')] = 0 if len(tm_cnn_w) == 0 else tm_cnn_w[3]

            if len(tm_cnn_w) != 0:
                horus_tx_ner_cnn = tm_cnn_w.index(max(tm_cnn_w)) + 1
            else:
                horus_tx_ner_cnn = self.text_bow.category2idx['MISC'] # forcing NONE

            maxs_tx = heapq.nlargest(2, gpb)
            maxs_tm = 0 if len(tm_cnn_w) == 0 else heapq.nlargest(2, tm_cnn_w)
            dist_tx_indicator = max(maxs_tx) - min(maxs_tx)
            dist_tx_indicator_tm = 0 if np.sum(y_tm[:,]) == 0 else (max(maxs_tm) - min(maxs_tm))

            token.features.text.values[tx_dict_rev.get('dist.k')] = dist_tx_indicator
            token.features.text.values[tx_dict_rev.get('dist.k.topic_model')] = dist_tx_indicator_tm
            token.features.text.values[tx_dict_rev.get('total.results.search_engine')] = nr_results_txt

            token.features.text.values[tx_dict_rev.get('total.emb.similar.loc')] = embs[0]
            token.features.text.values[tx_dict_rev.get('total.emb.similar.org')] = embs[1]
            token.features.text.values[tx_dict_rev.get('total.emb.similar.per')] = embs[2]
            token.features.text.values[tx_dict_rev.get('total.emb.similar.misc')] = embs[3]

            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.sum.loc')] = tm_k_top_sum[0]
            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.sum.org')] = tm_k_top_sum[1]
            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.sum.per')] = tm_k_top_sum[2]
            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.sum.misc')] = tm_k_top_sum[3]

            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.avg.loc')] = tm_k_top_avg[0]
            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.avg.org')] = tm_k_top_avg[1]
            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.avg.per')] = tm_k_top_avg[2]
            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.avg.misc')] = tm_k_top_avg[3]

            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.max.loc')] = tm_k_top_max[0]
            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.max.org')] = tm_k_top_max[1]
            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.max.per')] = tm_k_top_max[2]
            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.max.misc')] = tm_k_top_max[3]

            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.min.loc')] = tm_k_top_min[0]
            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.min.org')] = tm_k_top_min[1]
            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.min.per')] = tm_k_top_min[2]
            token.features.text.values[tx_dict_rev.get('stats.topic.top.k.min.misc')] = tm_k_top_min[3]

            token.features.text.values[tx_dict_rev.get('stats.topic.sum.loc')] = topic_sums[0]
            token.features.text.values[tx_dict_rev.get('stats.topic.sum.org')] = topic_sums[1]
            token.features.text.values[tx_dict_rev.get('stats.topic.sum.per')] = topic_sums[2]
            token.features.text.values[tx_dict_rev.get('stats.topic.sum.misc')] = topic_sums[3]

            token.features.text.values[tx_dict_rev.get('stats.topic.avg.loc')] = topic_avg[0]
            token.features.text.values[tx_dict_rev.get('stats.topic.avg.org')] = topic_avg[1]
            token.features.text.values[tx_dict_rev.get('stats.topic.avg.per')] = topic_avg[2]
            token.features.text.values[tx_dict_rev.get('stats.topic.avg.misc')] = topic_avg[3]

            token.features.text.values[tx_dict_rev.get('stats.topic.max.loc')] = topic_max[0]
            token.features.text.values[tx_dict_rev.get('stats.topic.max.org')] = topic_max[1]
            token.features.text.values[tx_dict_rev.get('stats.topic.max.per')] = topic_max[2]
            token.features.text.values[tx_dict_rev.get('stats.topic.max.misc')] = topic_max[3]

            token.features.text.values[tx_dict_rev.get('stats.topic.min.loc')] = topic_min[0]
            token.features.text.values[tx_dict_rev.get('stats.topic.min.org')] = topic_min[1]
            token.features.text.values[tx_dict_rev.get('stats.topic.min.per')] = topic_min[2]
            token.features.text.values[tx_dict_rev.get('stats.topic.min.misc')] = topic_min[3]

            if limit_txt != 0:
                token.features.text.values[tx_dict_rev.get('top.binary.k')] = definitions.encoder_4MUC_NER_idx2category[horus_tx_ner_label]
                token.features.text.values[tx_dict_rev.get('top.topic.k')] = definitions.encoder_4MUC_NER_idx2category[horus_tx_ner_cnn]
            else:
                token.features.text.values[tx_dict_rev.get('top.binary.k')] = definitions.encoder_4MUC_NER_idx2category[4]
                token.features.text.values[tx_dict_rev.get('top.topic.k')] = definitions.encoder_4MUC_NER_idx2category[4]

            return token

        except Exception as e:
            raise e

    def extract_features(self, horus: Horus) -> bool:
        '''
        Please see src/training/ for more information
        :param horus:
        :return:
        '''
        try:
            i_sent = 0
            tx_dict, tx_dict_reversed = WordFeaturesInterface.get_textual()
            for sentence in horus.sentences:
                i_sent += 1
                i_token = 0
                # -- PERFORMS THE PREDICTIONS PER TOKEN --
                for token in sentence.tokens:
                    i_token += 1
                    if token.label_pos in definitions.POS_NOUN_TAGS or token.is_compound == 1:
                        self.config.logger.debug(f'token: {token.text} ({i_token}/{len(sentence.tokens)}) | '
                                                 f'POS: {token.label_pos} ({token.label_pos_prob}) | '
                                                 f'sentence ({i_sent}/{len(horus.sentences)})')

                        with self.conn:
                            cursor = self.conn.cursor()
                            cursor.execute(SQL_TEXT_CLASS_SEL % (token.features.text.db_id, 0))
                            rows = cursor.fetchall()
                            nr_results_txt = len(rows)
                            limit_txt = min(nr_results_txt, int(self.config.search_engine_tot_resources))
                            if nr_results_txt == 0:
                                self.config.logger.debug("token/term has not returned web sites!")
                                # (OvR, 4-MUC model1, 4-MUC model2) * max documents
                                predictions_bow = [[np.zeros(1), np.zeros(4), np.zeros(4)]] * int(self.config.search_engine_tot_resources)
                                predictions_topic = [[0] * 5] * int(self.config.search_engine_tot_resources)
                            else:
                                predictions_bow = []
                                predictions_topic = []

                            # -- PERFORMS THE PREDICTIONS FOR EACH ASSOCIATED DOCUMENT, PER TOKEN --
                            ret_bow = []
                            ret_tm = []
                            for itxt in range(limit_txt):
                                try:
                                    text_title = rows[itxt][2]
                                    text_description = rows[itxt][3]
                                    text_title_en = rows[itxt][4]
                                    text_description_en = rows[itxt][5]
                                    if text_title_en is None:
                                        self.config.logger.warn("english translation [title] not available")
                                        text_title_en = text_title
                                    if text_description_en is None:
                                        self.config.logger.warn("english translation [description] not available")
                                        text_description_en = text_description
                                    merged_en = text_title_en + '. ' + text_description_en
                                    if merged_en.strip() == '':
                                        self.config.logger.warn('[merged_en] field should not be empty!')
                                    self.config.logger.debug(" -- text --> " + merged_en)
                                    # TODO: for now we do not store each document's prediction (e.g., 10 documents
                                    # per token, we do not have 10 predictions, but instead the average.
                                    # maybe in the future, we can create a VO object to save that info.
                                    if self.text_bow is not None:
                                        ret_bow = []
                                        pred_model_ovr, idx_k_model1, pred_model_1, idx_k_model2, pred_model_2 = \
                                            self.text_bow.detect_text_klass(merged_en)
                                        ret_bow.append(np.array([pred_model_ovr])) #1
                                        ret_bow.append(np.array(pred_model_1)) #4
                                        ret_bow.append(np.array(pred_model_2)) #4
                                    else:
                                        ret_bow = np.zeros(9)  # ovr, model1 and model2 probs
                                    if self.text_tm.wvmodel is not None:
                                        ret_tm = self.text_tm.detect_text_klass(merged_en)
                                    else:
                                        ret_tm = [0.0] * 5  # LOC, ORG, PER, NONE, free-slot for OTHER

                                    ret_bow = np.array(ret_bow)
                                    predictions_bow.append(ret_bow)
                                    predictions_topic.append(ret_tm)

                                    # TODO: fix and create a new cache mechanism later, as models have changed.
                                    '''
                                    index based is not the best approach, without a dictionary. 
                                    I will just comment this and always reprocess, also, removing the language detection
                                    and translation, I am supposed to have all text already translated in the DB
                                    
                                    if rows[itxt][6] is None or rows[itxt][6] == 0:  # not processed yet
                                        merged_en, error_translation = \
                                            self.__detect_lang_and_translate(text_title,
                                                                             text_description,
                                                                             rows[itxt][0],
                                                                             text_title_en,
                                                                             text_description_en)
                                        tot_error_translation += error_translation
                                        ret_bow = [0.0] * 3
                                        ret_tm = [0.0] * 5
                                        if merged_en.strip() != '':
                                            if self.text_bow is not None:
                                                pred_model_ovr, idx_k_model1, pred_model_1, idx_k_model2, pred_model_2 = \
                                                    self.text_bow.detect_text_klass(merged_en)
                                                ret_bow = [pred_model_ovr, pred_model_1, pred_model_2]
                                            if self.text_tm is not None:
                                                ret_tm = self.text_tm.detect_text_klass(merged_en)

                                        y_bow.append(ret_bow)
                                        y_tm.append(ret_tm)

                                        cursor.execute(SQL_TEXT_CLASS_UPD % (
                                            pred_model_ovr, str(ret_bow[0]), str(ret_bow[1]),
                                            ret_tm[0], ret_tm[1], ret_tm[2], ret_tm[3], ret_tm[4],
                                            embs[0], embs[1], embs[2], embs[3], rows[itxt][0]))
                                    else:
                                        # these are old models predictions, I will keep it to not mess up the index
                                        # based solution, which should be revised at some point
                                        # y_bow.append(rows[itxt][7:11])

                                        ret_bow = [rows[itxt][21], list(rows[itxt][22]), list(rows[itxt][23])]
                                        y_bow.append(ret_bow)

                                        y_tm.append(rows[itxt][12:16])
                                        embs = []
                                        embs.append(rows[itxt][17])
                                        embs.append(rows[itxt][18])
                                        embs.append(rows[itxt][19])
                                        embs.append(rows[itxt][20])
                                    '''
                                except Exception as e:
                                    self.config.logger.error(str(e.message))
                                    pass

                            #self.conn.commit()

                            # -- SAVE PREDICTIONS AND GENERATE OTHER STATS --
                            predictions_topic = np.array(predictions_topic)
                            predictions_bow = np.array(predictions_bow)
                            self.__set_token_statistics(token=token,
                                                        y_bow=predictions_bow,
                                                        y_tm=predictions_topic,
                                                        limit_txt=limit_txt,
                                                        nr_results_txt=nr_results_txt,
                                                        tx_dict=tx_dict,
                                                        tx_dict_rev=tx_dict_reversed)

        except Exception as e:
            raise e


class HorusExtractorLexical(HorusFeatureExtractor):
    def __init__(self, config: HorusConfig):
        super().__init__(config)
        self.config.logger.info('loading encoders')
        self.enc_le1 = joblib.load(self.config.dir_encoders + definitions.encoder_le1_name)
        self.enc_le2 = joblib.load(self.config.dir_encoders + definitions.encoder_le2_name)
        self.enc_word = joblib.load(self.config.dir_encoders + definitions.encoder_int_words_name)
        self.enc_lemma = joblib.load(self.config.dir_encoders + definitions.encoder_int_lemma_name)
        self.enc_stem = joblib.load(self.config.dir_encoders + definitions.encoder_int_stem_name)
        self.le = joblib.load(self.config.dir_encoders + encoder_le1_name)
        self.config.logger.info('loading brown corpus')
        self.dict_brown_c1000 = joblib.load(self.config.dir_clusters + 'gha.500M-c1000-p1.paths_dict.pkl')
        self.dict_brown_c640 = joblib.load(self.config.dir_clusters + 'gha.64M-c640-p1.paths_dict.pkl')
        self.dict_brown_c320 = joblib.load(self.config.dir_clusters + 'gha.64M-c320-p1.paths_dict.pkl')
        self.config.logger.info('loading lemmatizers')
        stemmer = SnowballStemmer('english')
        self.stop = set(stopwords.words('english'))
        wnl = WordNetLemmatizer()
        self.lemmatize = lru_cache(maxsize=50000)(wnl.lemmatize)
        self.stemo = lru_cache(maxsize=50000)(stemmer.stem)

    def extract_features(self, horus: Horus) -> bool:
        try:
            lx_dict, lx_dict_reversed = WordFeaturesInterface.get_lexical()
            tot_slide_brown_cluster = 5
            for sentence in horus.sentences:
                for token in sentence.tokens:
                    brown_1000_path = '{:<016}'.format(self.dict_brown_c1000.get(token.text, '0000000000000000'))
                    brown_640_path = '{:<016}'.format(self.dict_brown_c640.get(token.text, '0000000000000000'))
                    brown_320_path = '{:<016}'.format(self.dict_brown_c320.get(token.text, '0000000000000000'))

                    for i in range(0, tot_slide_brown_cluster - 1):
                        token.features.lexical.values[
                            lx_dict_reversed.get('brown_1000.' + str(i + 1))] = brown_1000_path[
                                                                                :i + 1]
                        token.features.lexical.values[lx_dict_reversed.get('brown_640.' + str(i + 1))] = brown_640_path[
                                                                                                         :i + 1]
                        token.features.lexical.values[lx_dict_reversed.get('brown_320.' + str(i + 1))] = brown_320_path[
                                                                                                         :i + 1]

                    token.features.lexical.values[lx_dict_reversed.get('word.lower')] = token.text.lower()

                    lemma = ''
                    try:
                        lemma = self.lemmatize(token.text.lower())
                    except:
                        pass

                    stem = ''
                    try:
                        stem = self.stemo(token.text.lower())
                    except:
                        pass

                    token.features.lexical.values[lx_dict_reversed.get('word.lemma')] = lemma
                    token.features.lexical.values[lx_dict_reversed.get('word.stem')] = stem
                    token.features.lexical.values[lx_dict_reversed.get('word.len.1')] = int(len(token.text) == 1)
                    token.features.lexical.values[lx_dict_reversed.get('word.has.special')] = int(
                        len(re.findall('(http://\S+|\S*[^\w\s]\S*)', token.text)) > 0)
                    token.features.lexical.values[lx_dict_reversed.get('word[0].isupper')] = int(
                        token.text[0].isupper())
                    token.features.lexical.values[lx_dict_reversed.get('word.isupper')] = int(token.text.isupper())
                    token.features.lexical.values[lx_dict_reversed.get('word.istitle')] = int(token.text.istitle())
                    token.features.lexical.values[lx_dict_reversed.get('word.isdigit')] = int(token.text.isdigit())
                    token.features.lexical.values[lx_dict_reversed.get('word.len.issmall')] = int(len(token.text) <= 2)
                    token.features.lexical.values[lx_dict_reversed.get('word.has.minus')] = int('-' in token.text)
                    token.features.lexical.values[lx_dict_reversed.get('word.stop')] = int(token.text in self.stop)
                    token.features.lexical.values[lx_dict_reversed.get('word.shape')] = self._shape(token.text)

            return True

        except Exception as e:
            raise e

    def _append_word_lemma_stem(self, w, l, s):
        t = []
        try:
            t.append(self.enc_word.transform(str(w)))
        except:
            self.config.logger.warn('enc_word.transform error')
            t.append(0)

        try:
            t.append(self.enc_lemma.transform(l.decode('utf-8')))
        except:
            self.config.logger.warn('enc_lemma.transform error')
            t.append(0)

        try:
            t.append(self.enc_stem.transform(s.decode('utf-8')))
        except:
            self.config.logger.warn('enc_stem.transform error')
            t.append(0)

        return t

    def _shape(self, word):
        word_shape = 0  # 'other'
        if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
            word_shape = 1  # 'number'
        elif re.match('\W+$', word):
            word_shape = 2  # 'punct'
        elif re.match('[A-Z][a-z]+$', word):
            word_shape = 3  # 'capitalized'
        elif re.match('[A-Z]+$', word):
            word_shape = 4  # 'uppercase'
        elif re.match('[a-z]+$', word):
            word_shape = 5  # 'lowercase'
        elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
            word_shape = 6  # 'camelcase'
        elif re.match('[A-Za-z]+$', word):
            word_shape = 7  # 'mixedcase'
        elif re.match('__.+__$', word):
            word_shape = 8  # 'wildcard'
        elif re.match('[A-Za-z0-9]+\.$', word):
            word_shape = 9  # 'ending-dot'
        elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
            word_shape = 10  # 'abbreviation'
        elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
            word_shape = 11  # 'contains-hyphen'

        return word_shape
