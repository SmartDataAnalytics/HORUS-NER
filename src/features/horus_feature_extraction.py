from abc import ABC, abstractmethod, ABCMeta
import gensim as gensim
from src import definitions
from src.algorithms.computer_vision.cls_dlib import DLib_Classifier
from src.algorithms.computer_vision.cnnlogo import CNNLogo
from src.algorithms.computer_vision.places365 import Places365CV
from src.algorithms.computer_vision.sift import SIFT
from src.algorithms.text_classification.bow_tfidf import BowTfidf
from src.algorithms.text_classification.topic_modeling_short_cnn import TopicModelingShortCNN
from src.horus_meta import Horus, WordFeaturesInterface
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

from src.utils.definitions_sql import SQL_TEXT_CLASS_SEL
from src.utils.nlp_tools import NLPTools
from src.utils.translation.bingtranslation import BingTranslator
from src.utils.util import Util
#from tensorflow.python.keras._impl.keras.applications import InceptionV3
#from tensorflow.python.keras._impl.keras.applications import InceptionV3
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
        self.config.logger.info('database connecting ...')
        self.conn = sqlite3.connect(self.config.database_db)

    @abstractmethod
    def extract_features(self, horus: Horus) -> bool:
        pass


class HorusExtractorImage(HorusFeatureExtractor):

    def __init__(self, p3):
        super().__init__()
        self.image_sift = SIFT(self.config)
        self.image_cnn_placesCNN = Places365CV(self.config)
        # self.image_cnn_incep_model = InceptionCV(self.config, version='V3')
        self.image_cnn_incep_model = InceptionV3(weights='imagenet')
        self.image_cnn_logo = CNNLogo(self.config)
        self.dlib_cnn = DLib_Classifier(self.config)

    def extract_features(self, horus: Horus) -> bool:
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.conn.close()
            del self.image_cnn_incep_model
            del self.image_cnn_placesCNN
        except:
            pass

class HorusExtractorText(HorusFeatureExtractor):
    def __init__(self):
        self.translator = BingTranslator(self.config)
        self.config.logger.info('loading word2vec embeedings...')
        self.text_bow = BowTfidf(self.config)
        self.word2vec_google = gensim.models.KeyedVectors.load_word2vec_format(self.config.embeddings_path, binary=True)
        self.text_tm = TopicModelingShortCNN(self.config, w2v=self.word2vec_google)
        self.extended_seeds_PER = []
        self.extended_seeds_ORG = []
        self.extended_seeds_LOC = []
        self.extended_seeds_NONE = []
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.config.logger.info('setting the seeds ')
        self.__set_str_extended_seeds()
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

        super().__init__()

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

    def __detect_and_translate(self, t1, t2, id, t1en, t2en):
        try:
            # if isinstance(t1, str):
            #    t1 = unicode(t1, "utf-8")
            # if isinstance(t2, str):
            #    t2 = unicode(t2, "utf-8")
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
                    if not isinstance(temp, unicode):
                        temp = temp.decode('utf-8')
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
                    if not isinstance(temp, unicode):
                        temp = temp.decode('utf-8')
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

            self.config.logger.debug('extending seeds')
            for k, V in definitions.seeds_dict_img_classes.iteritems():
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

    def extract_features(self, horus: Horus) -> bool:
        '''
        Please see src/training/ for more information
        :param horus:
        :return:
        '''
        try:
            tx_dict, tx_dict_reversed = WordFeaturesInterface.get_textual()

            i_sent = 0
            for sentence in horus.sentences:
                i_sent += 1
                i_token = 0
                for token in sentence.tokens:
                    i_token += 1
                    if token.label_pos in definitions.POS_NOUN_TAGS or token.is_compound == 1:
                        self.config.logger.debug(f'token: {token.text} ({i_token}/{len(sentence.tokens)}) | '
                                            f'sentence ({i_sent}/{len(horus.sentences)})')

                        id_term_txt = token.features.text.values[tx_dict_reversed.get('id_db')]
                        id_ner_type = 0
                        tot_geral_faces = 0
                        tot_geral_logos = 0
                        tot_geral_locations = 0
                        tot_geral_pos_locations = 0
                        tot_geral_neg_locations = 0
                        tot_geral_faces_cnn = 0
                        tot_geral_logos_cnn = 0
                        out_geral_cnn_features_loc = []

                        with self.conn:
                            cursor = self.conn.cursor()
                            y_bow = [[0] * 5] * int(self.config.search_engine_tot_resources)
                            y_tm = [[0] * 5] * int(self.config.search_engine_tot_resources)
                            cursor.execute(SQL_TEXT_CLASS_SEL % (id_term_txt, id_ner_type))
                            rows = cursor.fetchall()

                            nr_results_txt = len(rows)
                            if nr_results_txt == 0:
                                self.config.logger.debug("token/term has not returned web sites!")

                            limit_txt = min(nr_results_txt, int(self.config.search_engine_tot_resources))
                            tot_error_translation = 0
                            embs, top5_sim = self.__get_number_classes_in_embeedings(term)

                            klass_top = []
                            tm_cnn_w = []
                            tm_cnn_w_exp = []
                            if self.text_tm is not None:
                                # TM+CNN - term
                                tm_cnn_w = self.text_tm.detect_text_klass(term)
                                tm_cnn_w_exp = [np.math.pow(i, 2) for i in tm_cnn_w]

                            if self.text_tm is not None and top5_sim is not None:
                                for top in top5_sim:
                                    klass_top.append(self.text_tm.detect_text_klass(top[0]))
                            else:
                                klass_top = [[np.math.pow(10, -5)] * 5] * 5

                            klass_top.append(tm_cnn_w_exp)

                            if limit_txt > 0:
                                y_bow = []
                                y_tm = []
                            for itxt in range(limit_txt):
                                try:
                                    if rows[itxt][6] == 0 or rows[itxt][6] is None:  # not processed yet
                                        merged_en, error_translation = self.__detect_and_translate(rows[itxt][2],
                                                                                                   rows[itxt][3],
                                                                                                   rows[itxt][0],
                                                                                                   rows[itxt][4],
                                                                                                   rows[itxt][5])
                                        tot_error_translation += error_translation
                                        ret_bow = [0.0] * 5
                                        ret_tm = [0.0] * 5
                                        if merged_en != '':
                                            if self.text_bow is not None:
                                                ret_bow = self.text_bow.detect_text_klass(merged_en)
                                            if self.text_tm is not None:
                                                ret_tm = self.text_tm.detect_text_klass(merged_en)

                                        y_bow.append(ret_bow)

                                        # ret_tm = self.min_max_scaler.fit_transform(np.array(ret_tm).reshape(1,-1))[0]

                                        y_tm.append(ret_tm)

                                        cursor.execute(SQL_TEXT_CLASS_UPD % (
                                        ret_bow[0], ret_bow[1], ret_bow[2], ret_bow[3], ret_bow[4],
                                        ret_tm[0], ret_tm[1], ret_tm[2], ret_tm[3], ret_tm[4],
                                        embs[0], embs[1], embs[2], embs[3], rows[itxt][0]))
                                    else:
                                        y_bow.append(rows[itxt][7:11])
                                        y_tm.append(rows[itxt][12:16])
                                        embs = []
                                        embs.append(rows[itxt][17])
                                        embs.append(rows[itxt][18])
                                        embs.append(rows[itxt][19])
                                        embs.append(rows[itxt][20])



                                except Exception as e:
                                    self.config.logger.error(str(e.message))
                                    pass

                            self.conn.commit()

                            yyb = np.array(y_bow)
                            yytm = np.array(y_tm)
                            klass_top = np.array(klass_top)
                            gpb = [np.count_nonzero(yyb == 1), np.count_nonzero(yyb == 2), np.count_nonzero(yyb == 3)]

                            topic_klass_top_sums = [np.sum(klass_top[:, 0]), np.sum(klass_top[:, 1]),
                                                    np.sum(klass_top[:, 2]), np.sum(klass_top[:, 3])]
                            topic_klass_top_avg = [np.average(klass_top[:, 0]), np.average(klass_top[:, 1]),
                                                   np.average(klass_top[:, 2]), np.average(klass_top[:, 3])]
                            topic_klass_top_max = [np.max(klass_top[:, 0]), np.max(klass_top[:, 1]),
                                                   np.max(klass_top[:, 2]), np.max(klass_top[:, 3])]
                            topic_klass_top_min = [np.min(klass_top[:, 0]), np.min(klass_top[:, 1]),
                                                   np.min(klass_top[:, 2]), np.min(klass_top[:, 3])]

                            topic_sums = [np.sum(yytm[:, 0]), np.sum(yytm[:, 1]), np.sum(yytm[:, 2]),
                                          np.sum(yytm[:, 3])]
                            topic_avg = [np.average(yytm[:, 0]), np.average(yytm[:, 1]), np.average(yytm[:, 2]),
                                         np.average(yytm[:, 3])]
                            topic_max = [np.max(yytm[:, 0]), np.max(yytm[:, 1]), np.max(yytm[:, 2]), np.max(yytm[:, 3])]
                            topic_min = [np.min(yytm[:, 0]), np.min(yytm[:, 1]), np.min(yytm[:, 2]), np.min(yytm[:, 3])]

                            horus_tx_ner = gpb.index(max(gpb)) + 1

                            self.horus_matrix[index][definitions.INDEX_TOT_RESULTS_TX] = limit_txt
                            self.horus_matrix[index][definitions.INDEX_TOT_TX_LOC] = gpb[0]
                            self.horus_matrix[index][definitions.INDEX_TOT_TX_ORG] = gpb[1]
                            self.horus_matrix[index][definitions.INDEX_TOT_TX_PER] = gpb[2]
                            self.horus_matrix[index][definitions.INDEX_TOT_ERR_TRANS] = tot_error_translation

                            self.horus_matrix[index][definitions.INDEX_TOT_TX_LOC_TM_CNN] = 0 if len(tm_cnn_w) == 0 else \
                            tm_cnn_w[0]
                            self.horus_matrix[index][definitions.INDEX_TOT_TX_ORG_TM_CNN] = 0 if len(tm_cnn_w) == 0 else \
                            tm_cnn_w[1]
                            self.horus_matrix[index][definitions.INDEX_TOT_TX_PER_TM_CNN] = 0 if len(tm_cnn_w) == 0 else \
                            tm_cnn_w[2]
                            self.horus_matrix[index][definitions.INDEX_TOT_TX_NONE_TM_CNN] = 0 if len(
                                tm_cnn_w) == 0 else tm_cnn_w[3]

                            maxs_tx = heapq.nlargest(2, gpb)
                            maxs_tm = 0 if len(tm_cnn_w) == 0 else heapq.nlargest(2, tm_cnn_w)
                            dist_tx_indicator = max(maxs_tx) - min(maxs_tx)
                            dist_tx_indicator_tm = 0 if len(yytm) == 0 else (max(maxs_tm) - min(maxs_tm))

                            self.horus_matrix[index][definitions.INDEX_DIST_TX_I] = dist_tx_indicator
                            self.horus_matrix[index][definitions.INDEX_NR_RESULTS_SE_TX] = nr_results_txt
                            self.horus_matrix[index][definitions.INDEX_DIST_TX_I_TM_CNN] = dist_tx_indicator_tm

                            self.horus_matrix[index][definitions.INDEX_TOT_EMB_SIMILAR_LOC] = embs[0]
                            self.horus_matrix[index][definitions.INDEX_TOT_EMB_SIMILAR_ORG] = embs[1]
                            self.horus_matrix[index][definitions.INDEX_TOT_EMB_SIMILAR_PER] = embs[2]
                            self.horus_matrix[index][definitions.INDEX_TOT_EMB_SIMILAR_NONE] = embs[3]

                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_LOC] = \
                            topic_klass_top_sums[0]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_ORG] = \
                            topic_klass_top_sums[1]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_PER] = \
                            topic_klass_top_sums[2]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_NONE] = \
                            topic_klass_top_sums[3]

                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_LOC] = \
                            topic_klass_top_avg[0]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_ORG] = \
                            topic_klass_top_avg[1]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_PER] = \
                            topic_klass_top_avg[2]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_NONE] = \
                            topic_klass_top_avg[3]

                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_LOC] = \
                            topic_klass_top_max[0]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_ORG] = \
                            topic_klass_top_max[1]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_PER] = \
                            topic_klass_top_max[2]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_NONE] = \
                            topic_klass_top_max[3]

                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_LOC] = \
                            topic_klass_top_min[0]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_ORG] = \
                            topic_klass_top_min[1]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_PER] = \
                            topic_klass_top_min[2]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_NONE] = \
                            topic_klass_top_min[3]

                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_SUM_LOC] = topic_sums[0]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_SUM_ORG] = topic_sums[1]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_SUM_PER] = topic_sums[2]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_SUM_NONE] = topic_sums[3]

                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_AVG_LOC] = topic_avg[0]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_AVG_ORG] = topic_avg[1]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_AVG_PER] = topic_avg[2]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_AVG_NONE] = topic_avg[3]

                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_MAX_LOC] = topic_max[0]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_MAX_ORG] = topic_max[1]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_MAX_PER] = topic_max[2]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_MAX_NONE] = topic_max[3]

                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_MIN_LOC] = topic_min[0]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_MIN_ORG] = topic_min[1]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_MIN_PER] = topic_min[2]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_MIN_NONE] = topic_min[3]

                            if limit_txt != 0:
                                self.horus_matrix[index][definitions.INDEX_MAX_KLASS_PREDICT_TX] = \
                                definitions.PLOMNone_index2label[horus_tx_ner]
                            else:
                                self.horus_matrix[index][definitions.INDEX_MAX_KLASS_PREDICT_TX] = \
                                definitions.PLOMNone_index2label[4]

        except Exception as e:
            raise e


class HorusExtractorLexical(HorusFeatureExtractor):
    def __init__(self, p3):
        self.p3 = p3
        super().__init__()

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










