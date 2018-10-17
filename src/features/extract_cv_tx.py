# -*- coding: utf-8 -*-

"""
==========================================================
HORUS: Named Entity Recognition Algorithm
==========================================================

HORUS is a Named Entity Recognition Algorithm specifically
designed for short-text, i.e., microblogs and other noisy
datasets existing on the web, e.g.: social media, some web-
sites, blogs and etc..

It is a simplistic approach based on multi-level machine
learning combined with computer vision techniques.

more info at: https://github.com/dnes85/components-models

"""

# Author: Esteves <diegoesteves@gmail.com>
# Version: 1.0
# Version Label: HORUS_NER_2016_1.0
# License: BSD 3 clause
from src.classifiers.computer_vision.sift import SIFT
from src.classifiers.computer_vision.places365 import Places365CV




import gensim
import unicodecsv as csv
import heapq
import json
import sys
import sqlite3
import nltk
import numpy as np
import pandas as pd
from PIL import Image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.inception_v3 import preprocess_input
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from tensorflow.python.keras._impl.keras.applications import InceptionV3
from tensorflow.python.keras.preprocessing import image as ppimg
from src.classifiers.text_classification.topic_modeling_short_cnn import TopicModelingShortCNN
# print cv2.__version__
from src.classifiers.text_classification.bow_tfidf import BowTfidf
from nltk.corpus import wordnet as wn
import os
from nltk import re
from src.config import HorusConfig
from src.translation.azure import bing_detect_language, bing_translate_text
from src.translation.bingtranslation import BingTranslator
from src.util import definitions
from src.util.definitions_sql import SQL_TEXT_CLASS_SEL, SQL_OBJECT_DETECTION_SEL, SQL_OBJECT_DETECTION_UPD, \
    SQL_TEXT_CLASS_UPD
from src.util.nlp_tools import NLPTools
from src.util.util import Util
from src.classifiers.computer_vision.cls_dlib import DLib_Classifier
from src.classifiers.computer_vision.cnnlogo import CNNLogo
from src.classifiers.computer_vision.inception import InceptionCV

class FeatureExtraction(object):
    def __init__(self, config, load_sift=1, load_tfidf=1, load_cnn=1, load_topic_modeling=1):
        '''
        The HORUS feature_extraction class
        :param config: the configuration file
        :param load_sift: 1/0 (yes/no)
        :param load_tfidf: 1/0 (yes/no)
        :param load_cnn: 1/0 (yes/no)
        :param load_topic_modeling: 1/0 (yes/no)
        '''
        self.horus_matrix = [] # TODO: convert to pandas dataframe => pd.DataFrame(columns=definitions.HORUS_MATRIX_HEADER)
        self.config = config
        self.config.logger.info('loading components ...')
        self.util = Util(self.config)
        self.tools = NLPTools(self.config)
        self.translator = BingTranslator(self.config)
        self.config.logger.info('loading extractors! might take awhile...')
        self.image_cnn_logo = None
        self.image_sift = None
        self.text_bow = None
        self.text_tm = None
        self.dlib_cnn = None
        self.config.logger.info('loading word2vec embeedings...')
        self.word2vec_google = gensim.models.KeyedVectors.load_word2vec_format(self.config.embeddings_path, binary=True)
        if load_cnn==1:
            self.image_cnn_placesCNN = Places365CV(self.config)
            #self.image_cnn_incep_model = InceptionCV(self.config, version='V3')
            self.image_cnn_incep_model = InceptionV3(weights='imagenet')
            self.image_cnn_logo = CNNLogo(self.config)
            self.dlib_cnn = DLib_Classifier(self.config)
        if load_sift==1: self.image_sift = SIFT(self.config)
        if load_tfidf==1: self.text_bow = BowTfidf(self.config)
        if load_topic_modeling==1: self.text_tm = TopicModelingShortCNN(self.config, w2v=self.word2vec_google)
        self.config.logger.info('database connecting ...')
        self.conn = sqlite3.connect(self.config.database_db)
        self.extended_seeds_PER = []
        self.extended_seeds_ORG = []
        self.extended_seeds_LOC = []
        self.extended_seeds_NONE = []
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.config.logger.info('setting the seeds ')
        self.__set_str_extended_seeds()
        self.config.logger.info('hooray!')

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.conn.close()
            del self.image_cnn_incep_model
            del self.image_cnn_placesCNN
        except:
            pass

    def jsonDefault(object):
        return object.__dict__

    def __export_data(self, path, format='tsv'):

        try:
            self.config.logger.info('exporting metadata to: ' + path)
            data=self.horus_matrix
            if format == 'json':
                with open(path + '.json', 'wb') as outfile:
                    json.dump(data, outfile) #indent=4, sort_keys=True
            elif format == 'csv':
                writer = csv.writer(open(path + '.csv', 'wb'), quoting=csv.QUOTE_ALL)
                writer.writerow(definitions.HORUS_MATRIX_HEADER)
                # writer.writerow([s.encode('utf8') if type(s) is unicode else s for s in self.horus_matrix])
                writer.writerows(data)
            elif format == 'tsv':
                writer = csv.writer(open(path, 'wb'), dialect="excel", delimiter='\t', skipinitialspace=True, encoding='utf-8')
                writer.writerow(definitions.HORUS_MATRIX_HEADER)
                writer.writerows(data)
            else:
                raise Exception('format not implemented')
            self.config.logger.info("data exported successfully")
        except Exception as e:
            raise(e)

    def __detect_and_translate(self, t1, t2, id, t1en, t2en):
        try:
            #if isinstance(t1, str):
            #    t1 = unicode(t1, "utf-8")
            #if isinstance(t2, str):
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
                    error+=1


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
                    c.execute(sql, (temp, id)) #.encode("utf-8")
                    t2en = temp
                except Exception as e:
                    sql = """UPDATE HORUS_SEARCH_RESULT_TEXT SET error = 1, error_desc = ? WHERE id = ?"""
                    c.execute(sql, (str(e.message), id))
                    error+=1

            c.close()

            merged = ''
            if t1en is not None:
                merged = t1en.encode('ascii','ignore')
            if t2en is not None:
                merged = merged + ' ' + t2en.encode('ascii','ignore')

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
                            #self.extended_seeds_PER.extend([str(_s._name).split('.')[0] for _s in syns.hypernyms()])
                            # ids_PER.extend(syns.hyponyms())
                        elif k == 'org':
                            self.extended_seeds_ORG.append(str(_s._name).split('.')[0])
                            #self.extended_seeds_ORG.extend([str(_s._name).split('.')[0] for _s in syns.hypernyms()])
                            # ids_ORG.extend(syns.hyponyms())
                        elif k == 'loc':
                            self.extended_seeds_LOC.append(str(_s._name).split('.')[0])
                            #self.extended_seeds_LOC.extend([str(_s._name).split('.')[0] for _s in syns.hypernyms()])
                            # ids_LOC.extend(syns.hyponyms())
                        elif k == 'none':
                            self.extended_seeds_NONE.append(str(_s._name).split('.')[0])
                            #self.extended_seeds_NONE.extend([str(_s._name).split('.')[0] for _s in syns.hypernyms()])
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

    def __get_cnn_features_vector(self, imgpath):
        try:

            p1=[]
            p2=[]

            from PIL import Image
            from os.path import splitext
            file_name, extension = splitext(imgpath)
            if extension not in ('.jpg', '.jpeg', '.png'):
                try:
                    im = Image.open(imgpath)
                    rgb_im = im.convert('RGB')
                    newfilename=str(imgpath).replace(extension, '.jpg')
                    rgb_im.save(newfilename)
                    imgpath=newfilename
                except:
                    pass

            try:
                self.config.logger.debug('predicting CNN places365...')
                p1 = self.image_cnn_placesCNN.predict(imgpath)
            except Exception as e:
                self.config.logger.debug('error: ' + str(e))
                pass

            try:
                self.config.logger.debug('predicting CNN inception...')
                img = ppimg.load_img(imgpath, target_size=(299, 299))
                x = ppimg.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                y = self.image_cnn_incep_model.predict(x)
                for index, res in enumerate(decode_predictions(y)[0]):
                    p2.append((res[1], res[2]))
                    #print('{}. {}: {:.3f}%'.format(index + 1, res[1], 100 * res[2]))
                    if index>5:
                        break

            except Exception as e:
                self.config.logger.debug('error: ' + str(e))
                pass

            p1.extend(p2)

            tot_loc = 0
            tot_org = 0
            tot_per = 0
            tot_none = 0

            sim_loc = []
            sim_org = []
            sim_per = []
            sim_none = []

            assert len(p1) > 0

            # text classification Topic Modeling+CNN
            for i in range(0, len(p1)):
                out = []
                labels = p1[i][0]
                prob = p1[i][1]
                cleaned_tokens = ' '.join((re.sub('[^0-9a-zA-Z]+', ' ', l) for l in labels.split()))
                T = [l.replace('/outdoor', '').replace('/indoor', '') for l in cleaned_tokens.split()]
                T = set(T)
                for t in T:
                    i += 1
                    d = self.text_tm.predict(t.lower())
                    tot_loc += (d.get('loc') * prob)
                    tot_per += (d.get('per') * prob)
                    tot_org += (d.get('org') * prob)
                    tot_none += (d.get('none') * prob)


                    for s in self.extended_seeds_LOC:
                        try:
                            sim_loc.append(self.word2vec_google.similarity(s, t))
                        except:pass

                    for s in self.extended_seeds_ORG:
                        try:
                            sim_org.append(self.word2vec_google.similarity(s, t))
                        except:pass

                    for s in self.extended_seeds_PER:
                        try:
                            sim_per.append(self.word2vec_google.similarity(s, t))
                        except:pass

                    for s in self.extended_seeds_NONE:
                        try:
                            sim_none.append(self.word2vec_google.similarity(s, t))
                        except:pass


            assert i>0

            sim_loc = self.min_max_scaler.fit_transform(np.array(sim_loc).reshape(-1,1))
            sim_org = self.min_max_scaler.fit_transform(np.array(sim_org).reshape(-1,1))
            sim_per = self.min_max_scaler.fit_transform(np.array(sim_per).reshape(-1,1))
            sim_none = self.min_max_scaler.fit_transform(np.array(sim_none).reshape(-1,1))

            return [tot_loc/i, tot_org/i, tot_per/i, tot_none/i,
                    np.mean(sim_loc) if len(sim_loc) > 0 else 0,
                    np.mean(sim_org) if len(sim_org) > 0 else 0,
                    np.mean(sim_per) if len(sim_per) > 0 else 0,
                    np.mean(sim_none) if len(sim_none) > 0 else 0]

        except Exception as e:
            self.config.logger.error(str(e))
            return [0] * 8

    def __extract_features(self, horus_matrix, features_text=True, features_vision=True):
        """
        receives a horus_matrix and iterate over the tokens, detecting objects for each image/document
        in a set of images/documents related to a given token
        :param horus_matrix: the horus_matrix
        :return: an updated horus matrix
        """
        try:

            self.horus_matrix = horus_matrix

            assert (features_text is False or features_text is True and (self.text_tm is not None or self.text_bow is not None))

            assert (features_vision is False or features_vision is True and ((self.image_cnn_placesCNN is not None and self.dlib_cnn is not None
                                                and self.image_cnn_incep_model is not None and self.image_cnn_logo is not None) or
                                                (self.image_sift is not None)))

            toti = len(self.horus_matrix)
            self.config.logger.info('extracting features for %s terms/tokens' % str(toti))
            auxi = 0
            for index in range(len(self.horus_matrix)):
                auxi += 1
                if (self.horus_matrix[index][definitions.INDEX_POS] in definitions.POS_NOUN_TAGS) or self.horus_matrix[index][definitions.INDEX_IS_COMPOUND] == 1:

                    term = self.horus_matrix[index][definitions.INDEX_TOKEN]
                    self.config.logger.info('token %d of %d [%s]' % (auxi, toti, term))

                    id_term_img = self.horus_matrix[index][definitions.INDEX_ID_TERM_IMG]
                    id_term_txt = self.horus_matrix[index][definitions.INDEX_ID_TERM_TXT]
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
                        if features_vision is True:
                            cursor.execute(SQL_OBJECT_DETECTION_SEL % (id_term_img, id_ner_type))
                            rows = cursor.fetchall()
                            nr_results_img = len(rows)
                            if nr_results_img == 0:
                                self.config.logger.debug("token/term has not returned images!")
                            limit_img = min(nr_results_img, int(self.config.search_engine_tot_resources))

                            # 0 = file path | 1 = id | 2 = processed | 3=nr_faces | 4=nr_logos | 5 to 14=nr_places_1-to-10
                            # 14 = nr_faces_cnn, 15 = nr_logos_cnn, 16-25=nr_places_1-to-10_cnn
                            tot_processed_img = 0
                            for i in range(limit_img):
                                _id = rows[i][1]
                                img_full_path = self.config.cache_img_folder + rows[i][0]
                                try:
                                    Image.open(img_full_path)
                                    tot_processed_img+=1
                                except IOError:
                                    self.config.logger.error('image error: ' + img_full_path)
                                    continue

                                if rows[i][2] == 1: # processed
                                    tot_geral_faces += rows[i][3] if rows[i][3] != None else 0
                                    tot_geral_logos += rows[i][4] if rows[i][4] != None else 0
                                    if (rows[i][5:14]).count(1) >= int(self.config.models_location_theta):
                                        tot_geral_locations += 1
                                    tot_geral_pos_locations += rows[i][5:14].count(1)
                                    tot_geral_neg_locations += (rows[i][5:14].count(0) * -1)

                                    tot_geral_faces_cnn += rows[i][15] if rows[i][15] != None else 0
                                    tot_geral_logos_cnn += rows[i][16] if rows[i][16] != None else 0

                                    out_geral_cnn_features_loc.append(rows[i][17:26])

                                    #if (rows[i][17:26]).count(1) >= int(self.config.models_location_theta):
                                    #    tot_geral_locations_cnn += 1
                                    #tot_geral_pos_locations_cnn += rows[i][17:26].count(1)
                                    #tot_geral_neg_locations_cnn += (rows[i][17:26].count(0) * -1)

                                else:
                                    tot_faces = 0
                                    tot_logos = 0
                                    res = [0] * 10
                                    tot_faces_cnn = 0
                                    tot_logos_cnn = 0
                                    out_cnn_features_loc = [0] * 8

                                    #CV - SIFT detection
                                    if self.image_sift is not None:
                                        # ----- face recognition -----
                                        tot_faces = self.image_sift.detect_faces(img_full_path)
                                        if tot_faces > 0:
                                            tot_geral_faces += 1
                                            self.config.logger.debug("found {0} faces!".format(tot_faces))
                                        # ----- logo recognition -----
                                        tot_logos = self.image_sift.detect_logo(img_full_path)
                                        if tot_logos > 0:
                                            tot_geral_logos += 1
                                            self.config.logger.debug("found {0} logo(s)!".format(1))
                                        # ----- place recognition -----
                                        res = self.image_sift.detect_place(img_full_path)
                                        tot_geral_pos_locations += res.count(1)
                                        tot_geral_neg_locations += (res.count(0) * -1)

                                        if res.count(1) >= int(self.config.models_location_theta):
                                            tot_geral_locations += 1
                                            self.config.logger.debug("found {0} place(s)!".format(1))

                                    # CV - CNN detection (logo feature)
                                    if self.image_cnn_logo is not None:
                                        tot_logos_cnn = self.image_cnn_logo.predict(img_full_path)
                                        if tot_logos_cnn > 0:
                                            tot_geral_logos_cnn += 1
                                            self.config.logger.debug("found {0} logo(s)!".format(1))

                                    # CV - CNN detection (place features)
                                    if ((self.image_cnn_placesCNN is not None) or (self.image_cnn_incep_model is not None)):
                                        out_cnn_features_loc=self.__get_cnn_features_vector(img_full_path)
                                        out_geral_cnn_features_loc.append(out_cnn_features_loc)

                                    # CV - CNN detection (face feature)
                                    if self.dlib_cnn is not None:
                                        tot_faces_cnn = self.dlib_cnn.detect_faces_cnn(img_full_path)
                                        tot_geral_faces_cnn += tot_faces_cnn
                                        self.config.logger.debug("found {0} faces!".format(tot_faces_cnn))

                                    # CV - blob detection for logos
                                    # TODO: to create this function

                                    param = []
                                    param.append(tot_faces)
                                    param.append(tot_logos)
                                    param.extend(res)

                                    param.append(tot_faces_cnn)
                                    param.append(tot_logos_cnn)
                                    param.extend(out_cnn_features_loc)

                                    param.append(_id)
                                    cursor.execute(SQL_OBJECT_DETECTION_UPD, param)

                            self.conn.commit()

                            outs = [tot_geral_locations, tot_geral_logos, tot_geral_faces]
                            maxs_cv = heapq.nlargest(2, outs)
                            dist_cv_indicator = max(maxs_cv) - min(maxs_cv)
                            place_cv_indicator = tot_geral_pos_locations + tot_geral_neg_locations

                            #outs_cnn = [tot_geral_locations_cnn, tot_geral_logos_cnn, tot_geral_faces_cnn]
                            #maxs_cv_cnn = heapq.nlargest(2, outs_cnn)
                            #dist_cv_indicator_cnn = max(maxs_cv_cnn) - min(maxs_cv_cnn)
                            #place_cv_indicator_cnn = tot_geral_pos_locations_cnn + tot_geral_neg_locations_cnn

                            self.horus_matrix[index][definitions.INDEX_TOT_IMG] = tot_processed_img
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_LOC] = tot_geral_locations  # 1
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_ORG] = tot_geral_logos  # 2
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_PER] = tot_geral_faces  # 3
                            self.horus_matrix[index][definitions.INDEX_DIST_CV_I] = dist_cv_indicator  # 4
                            self.horus_matrix[index][definitions.INDEX_PL_CV_I] = place_cv_indicator  # 5
                            self.horus_matrix[index][definitions.INDEX_NR_RESULTS_SE_IMG] = nr_results_img  # 5
                            x=np.sum(out_geral_cnn_features_loc, axis=0)
                            if isinstance(x,list) == False:
                                x=[0] * 8
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_LOC_1_CNN] = x[0]  # 1
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_LOC_2_CNN] = x[1]
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_LOC_3_CNN] = x[2]
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_LOC_4_CNN] = x[3]
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_LOC_5_CNN] = x[4]
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_LOC_6_CNN] = x[5]
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_LOC_7_CNN] = x[6]
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_LOC_8_CNN] = x[7]
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_ORG_CNN] = tot_geral_logos_cnn  # 2
                            self.horus_matrix[index][definitions.INDEX_TOT_CV_PER_CNN] = tot_geral_faces_cnn  # 3
                            #self.horus_matrix[index][definitions.INDEX_DIST_CV_I_CNN] = dist_cv_indicator_cnn  # 5
                            #self.horus_matrix[index][definitions.INDEX_PL_CV_I_CNN] = place_cv_indicator_cnn  # 6

                            '''
                            self.config.logger.debug('CV statistics:[BOW: LOC=%s, ORG=%s, PER=%s, DIST=%s, PLC=%s | '
                                              'CNN: LOC1=%s,LOC2=%s,LOC3=%s,LOC4=%s,LOC5=%s,LOC6=%s,LOC7=%s,LOC8=%s, '
                                                     'ORG=%s, PER=%s]' %
                                              (str(tot_geral_locations).zfill(2), str(tot_geral_logos).zfill(2),
                                               str(tot_geral_faces).zfill(2), str(dist_cv_indicator).zfill(2), place_cv_indicator,
                                               str(x[0]).zfill(4),
                                               str(x[1]).zfill(4),
                                               str(x[2]).zfill(4),
                                               str(x[3]).zfill(4),
                                               str(x[4]).zfill(4),
                                               str(x[5]).zfill(4),
                                               str(x[6]).zfill(4),
                                               str(x[7]).zfill(4),
                                               str(tot_geral_logos_cnn).zfill(2),
                                               str(tot_geral_faces_cnn).zfill(2)))
                            '''

                            if tot_processed_img != 0:
                                self.horus_matrix[index][definitions.INDEX_MAX_KLASS_PREDICT_CV] = definitions.PLONone_index2label[outs.index(max(outs)) + 1]
                                #self.horus_matrix[index][definitions.INDEX_MAX_KLASS_PREDICT_CV_CNN] = definitions.KLASSES[outs_cnn.index(max(outs_cnn)) + 1]
                                #TODO: this does not make sense now, since CNN classifiers are a bit more complex...thinkg about that later...it does not impact the algorithm
                                self.horus_matrix[index][definitions.INDEX_MAX_KLASS_PREDICT_CV_CNN] = definitions.PLONone_index2label[4]
                            else:
                                self.horus_matrix[index][definitions.INDEX_MAX_KLASS_PREDICT_CV] = definitions.PLONone_index2label[4]
                                self.horus_matrix[index][definitions.INDEX_MAX_KLASS_PREDICT_CV_CNN] = definitions.PLONone_index2label[4]

                        if features_text is True:
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
                            tm_cnn_w_exp=[]
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
                                y_bow =[]
                                y_tm=[]
                            for itxt in range(limit_txt):
                                try:
                                    if rows[itxt][6] == 0 or rows[itxt][6] is None:  # not processed yet
                                        merged_en, error_translation = self.__detect_and_translate(rows[itxt][2], rows[itxt][3], rows[itxt][0], rows[itxt][4], rows[itxt][5])
                                        tot_error_translation += error_translation
                                        ret_bow = [0.0] * 5
                                        ret_tm = [0.0] * 5
                                        if merged_en != '':
                                            if self.text_bow is not None:
                                                ret_bow = self.text_bow.detect_text_klass(merged_en)
                                            if self.text_tm is not None:
                                                ret_tm = self.text_tm.detect_text_klass(merged_en)

                                        y_bow.append(ret_bow)

                                        #ret_tm = self.min_max_scaler.fit_transform(np.array(ret_tm).reshape(1,-1))[0]

                                        y_tm.append(ret_tm)

                                        cursor.execute(SQL_TEXT_CLASS_UPD % (ret_bow[0], ret_bow[1], ret_bow[2], ret_bow[3], ret_bow[4],
                                                                       ret_tm[0], ret_tm[1], ret_tm[2], ret_tm[3], ret_tm[4],
                                                                       embs[0], embs[1], embs[2], embs[3], rows[itxt][0]))
                                    else:
                                        y_bow.append(rows[itxt][7:11])
                                        y_tm.append(rows[itxt][12:16])
                                        embs=[]
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

                            topic_klass_top_sums = [np.sum(klass_top[:, 0]), np.sum(klass_top[:, 1]), np.sum(klass_top[:, 2]), np.sum(klass_top[:, 3])]
                            topic_klass_top_avg = [np.average(klass_top[:, 0]), np.average(klass_top[:, 1]), np.average(klass_top[:, 2]), np.average(klass_top[:, 3])]
                            topic_klass_top_max = [np.max(klass_top[:, 0]), np.max(klass_top[:, 1]), np.max(klass_top[:, 2]), np.max(klass_top[:, 3])]
                            topic_klass_top_min = [np.min(klass_top[:, 0]), np.min(klass_top[:, 1]), np.min(klass_top[:, 2]), np.min(klass_top[:, 3])]


                            topic_sums = [np.sum(yytm[:,0]), np.sum(yytm[:,1]), np.sum(yytm[:,2]), np.sum(yytm[:,3])]
                            topic_avg = [np.average(yytm[:,0]), np.average(yytm[:,1]), np.average(yytm[:,2]), np.average(yytm[:,3])]
                            topic_max = [np.max(yytm[:,0]), np.max(yytm[:,1]), np.max(yytm[:,2]), np.max(yytm[:,3])]
                            topic_min = [np.min(yytm[:,0]), np.min(yytm[:,1]), np.min(yytm[:,2]), np.min(yytm[:,3])]

                            horus_tx_ner = gpb.index(max(gpb)) + 1

                            self.horus_matrix[index][definitions.INDEX_TOT_RESULTS_TX] = limit_txt
                            self.horus_matrix[index][definitions.INDEX_TOT_TX_LOC] = gpb[0]
                            self.horus_matrix[index][definitions.INDEX_TOT_TX_ORG] = gpb[1]
                            self.horus_matrix[index][definitions.INDEX_TOT_TX_PER] = gpb[2]
                            self.horus_matrix[index][definitions.INDEX_TOT_ERR_TRANS] = tot_error_translation

                            self.horus_matrix[index][definitions.INDEX_TOT_TX_LOC_TM_CNN] = 0 if len(tm_cnn_w) == 0 else tm_cnn_w[0]
                            self.horus_matrix[index][definitions.INDEX_TOT_TX_ORG_TM_CNN] = 0 if len(tm_cnn_w) == 0 else tm_cnn_w[1]
                            self.horus_matrix[index][definitions.INDEX_TOT_TX_PER_TM_CNN] = 0 if len(tm_cnn_w) == 0 else tm_cnn_w[2]
                            self.horus_matrix[index][definitions.INDEX_TOT_TX_NONE_TM_CNN] = 0 if len(tm_cnn_w) == 0 else tm_cnn_w[3]

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

                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_LOC] = topic_klass_top_sums[0]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_ORG] = topic_klass_top_sums[1]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_PER] = topic_klass_top_sums[2]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_SUM_NONE] = topic_klass_top_sums[3]

                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_LOC] = topic_klass_top_avg[0]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_ORG] = topic_klass_top_avg[1]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_PER] = topic_klass_top_avg[2]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_AVG_NONE] = topic_klass_top_avg[3]

                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_LOC] = topic_klass_top_max[0]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_ORG] = topic_klass_top_max[1]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_PER] = topic_klass_top_max[2]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MAX_NONE] = topic_klass_top_max[3]

                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_LOC] = topic_klass_top_min[0]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_ORG] = topic_klass_top_min[1]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_PER] = topic_klass_top_min[2]
                            self.horus_matrix[index][definitions.INDEX_TX_CNN_STAT_T_PLUS_TOP5_K_MIN_NONE] = topic_klass_top_min[3]

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
                                self.horus_matrix[index][definitions.INDEX_MAX_KLASS_PREDICT_TX] = definitions.PLONone_index2label[horus_tx_ner]
                            else:
                                self.horus_matrix[index][definitions.INDEX_MAX_KLASS_PREDICT_TX] = definitions.PLONone_index2label[4]


                            '''
                            self.config.logger.debug('TX statistics:'
                                               '[BoW: LOC=%s, ORG=%s, PER=%s, DIST=%s | ' 'TM: LOC=%s, ORG=%s, PER=%s, DIST=%s, '
                                                     'TOT_EMB_LOC=%s, TOT_EMB_ORG=%s, TOT_EMB_PER=%s, TOT_EMB_NONE=%s]' %
                                               (str(self.horus_matrix[index][definitions.INDEX_TOT_TX_LOC]).zfill(2),
                                                str(self.horus_matrix[index][definitions.INDEX_TOT_TX_ORG]).zfill(2),
                                                str(self.horus_matrix[index][definitions.INDEX_TOT_TX_PER]).zfill(2),
                                                str(self.horus_matrix[index][definitions.INDEX_DIST_TX_I]).zfill(2),
                                                "{:.4f}".format(self.horus_matrix[index][definitions.INDEX_TOT_TX_LOC_TM_CNN]).zfill(2),
                                                "{:.4f}".format(self.horus_matrix[index][definitions.INDEX_TOT_TX_ORG_TM_CNN]).zfill(2),
                                                "{:.4f}".format(self.horus_matrix[index][definitions.INDEX_TOT_TX_PER_TM_CNN]).zfill(2),
                                                "{:.4f}".format(self.horus_matrix[index][definitions.INDEX_DIST_TX_I_TM_CNN]).zfill(2),
                                                str(self.horus_matrix[index][definitions.INDEX_TOT_EMB_SIMILAR_LOC]).zfill(2),
                                                str(self.horus_matrix[index][definitions.INDEX_TOT_EMB_SIMILAR_ORG]).zfill(2),
                                                str(self.horus_matrix[index][definitions.INDEX_TOT_EMB_SIMILAR_PER]).zfill(2),
                                                str(self.horus_matrix[index][definitions.INDEX_TOT_EMB_SIMILAR_NONE]).zfill(2))
                                                )
                            self.config.logger.debug('-------------------------------------------------------------')
                            '''

            return True

        except Exception as e:
            self.config.logger.error(repr(e))
            self.conn.rollback()
            self.config.logger.info('saving partially processed file...')
            try:
                #self.horus_matrix[index] = [0] * int(definitions.HORUS_TOT_FEATURES)
                self.__export_data(self.config.dir_output + 'outfeatures.horus', 'json')
            except Exception as e2:
                self.config.logger.error('error: not possible to save the file! ' + str(e2))
            return False

    def extract_features_from_text(self, text):
        """
        extracts the HORUS features for a given input text
        :param text: the input text
        :return: the features
        """
        try:
            if text is None:
                raise Exception("Provide an input text")
            self.config.logger.info('text: ' + text)
            self.config.logger.info('tokenizing sentences ...')
            sent_tokenize_list = nltk.sent_tokenize(text, language='english')
            self.config.logger.info('processing ' + str(len(sent_tokenize_list)) + ' sentence(s).')
            sentences = []
            for sentence in sent_tokenize_list:
                sentences.append(self.util.process_and_save_sentence(-1, sentence))

            self.horus_matrix = self.util.sentence_to_horus_matrix(sentences)
            self.util.download_and_cache_results(self.horus_matrix)
            if self.__extract_features() is True:
                self.__export_data(self.config.dir_output + 'features_horus', 'json')

            return self.horus_matrix

        except Exception as error:
            self.config.logger.error('extract_features_from_text() error: ' + repr(error))

    def extract_features_from_conll(self, horus_m2, label, sep='\t'):
        """
        generates the feature_extraction data for HORUS
        do not use the config file to choose the models, exports all features (self.detect_objects())
        :param file: a dataset src/core/feature_extraction/features.py(CoNLL format)
        :param label: a dataset label
        :param token_index: column index of the token (word)
        :param ner_index: column index if the target class (NER)
        :return: the feature file
        """
        try:

            if horus_m2 is None:
                raise Exception("Provide an input file format to be annotated")
            if not os.path.isfile(horus_m2):
                config.logger.error('file %s not found!' % (horus_m2))
                raise Exception

            horus_m3_path = os.path.dirname(horus_m2) + '/' + os.path.basename(horus_m2).replace('.horus2', '.horus3')

            self.config.logger.info('extracting features from -> %s' % label)

            if os.path.isfile(horus_m3_path):
                config.logger.warn('file already exists: %s' % (horus_m3_path))
                return
            else:

                with open(horus_m2) as f:
                    reader = csv.reader(f, delimiter=sep)
                    next(reader)
                    horus_matrix = [r for r in reader]

                if self.__extract_features(horus_matrix, features_vision=True, features_text=True) is True:
                    #filename = label + "." + self.util.path_leaf(file) + ".horus3"
                    #path = self.config.dir_output + output_folder + filename
                    self.__export_data(horus_m3_path)
                else:
                    self.config.logger.warn('well, nothing to do today...')
        except Exception as error:
            self.config.logger.error('extract_features_from_conll() error: ' + repr(error))



if __name__ == "__main__":

    config = HorusConfig()

    try:

        extractor = FeatureExtraction(config, load_sift=1, load_tfidf=1, load_cnn=1, load_topic_modeling=1)

        for ds in definitions.NER_DATASETS:
            extractor.extract_features_from_conll(horus_m2=ds[1] + ds[2].replace('.horusx', '.horus2'), label=ds[0])

        '''
        attention: change POS tag lib in the HORUS.ini to NLTK before run this
        extractor.extract_features_from_conll('coNLL2003/nodocstart_coNLL2003.eng.testA', exp_folder, 'conll03', 0, 3)
        '''

    except Exception as e:
        config.logger.error(repr(e))
