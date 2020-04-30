from abc import ABC, abstractmethod, ABCMeta
from src import definitions
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
from src.utils.nlp_tools import NLPTools


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
        self.image_sift = SIFT(config)
        self.image_cnn_placesCNN = Places365CV(config)
        # self.image_cnn_incep_model = InceptionCV(self.config, version='V3')
        self.image_cnn_incep_model = InceptionV3(weights='imagenet')
        self.image_cnn_logo = CNNLogo(config)
        self.dlib_cnn = DLib_Classifier(config)
        super().__init__()

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










