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
import csv
import heapq
import json
import sys
import sqlite3
import nltk
import numpy
import pandas as pd

from src.classifiers.computer_vision.cls_dlib import DLib_Classifier
from src.classifiers.computer_vision.inception import InceptionCV
from src.classifiers.computer_vision.sift import SIFT
from src.classifiers.text_classification.topic_modeling_short_cnn import TopicModelingShortCNN
from src.core.translation.bingtranslation import BingTranslator
from src.core.util.util import Util
from src.core.translation.azure import *
from src.core.util.nlp_tools import NLPTools
from src.core.util.definitions_sql import *
# print cv2.__version__
from src.classifiers.text_classification.bow_tfidf import BowTfidf
from src.core.util import definitions


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
        self.horus_matrix = []
        self.config = config
        self.config.logger.info('==================================================================')
        self.config.logger.info(':: HORUS Framework - Feature Extractor')
        self.config.logger.info(':: ' + self.config.version)
        self.config.logger.info(':: more info: http://horus-ner.org/')
        self.config.logger.info('==================================================================')
        self.config.logger.info('loading components...')
        self.util = Util(self.config)
        self.tools = NLPTools(self.config)
        self.translator = BingTranslator(self.config)
        self.config.logger.info('loading extractors...')
        self.image_cnn = None
        self.image_sift = None
        self.text_bow = None
        self.text_tm = None
        self.dlib_cnn = DLib_Classifier(self.config)
        if load_cnn==1: self.image_cnn = InceptionCV(self.config)
        if load_sift==1: self.image_sift = SIFT(self.config)
        if load_tfidf==1: self.text_bow = BowTfidf(self.config)
        if load_topic_modeling==1: self.text_tm = TopicModelingShortCNN(self.config)
        self.config.logger.info('database connecting ...')
        self.conn = sqlite3.connect(self.config.database_db)
        if bool(int(self.config.models_force_download)) is True:
            self.config.logger.info('downloading NLTK data...')
            try:
                nltk.data.find('averaged_perceptron_tagger.zip')
            except LookupError: nltk.download('averaged_perceptron_tagger')
            try:
                nltk.data.find('punkt.zip')
            except LookupError: nltk.download('punkt')
            try:
                nltk.data.find('maxent_ne_chunker.zip')
            except LookupError: nltk.download('maxent_ne_chunker')
            try:
                nltk.data.find('universal_tagset.zip')
            except LookupError: nltk.download('universal_tagset')
            try:
                nltk.data.find('words.zip')
            except LookupError: nltk.download('words')

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.conn.close()
        except:
            pass

    def jsonDefault(object):
        return object.__dict__

    def __export_data(self, path, format):
        try:
            self.config.logger.info(':: exporting metadata to: ' + path + "." + format)
            if format == 'json':
                with open(path + '.json', 'wb') as outfile:
                    json.dump(self.horus_matrix, outfile) #indent=4, sort_keys=True
            elif format == 'csv':
                writer = csv.writer(open(path + '.csv', 'wb'), quoting=csv.QUOTE_ALL)
                writer.writerow(definitions.HORUS_MATRIX_HEADER)
                # writer.writerow([s.encode('utf8') if type(s) is unicode else s for s in self.horus_matrix])
                writer.writerows(self.horus_matrix)
            elif format == 'tsv':
                writer = csv.writer(open(path + '.tsv', 'wb'), dialect="excel", delimiter="\t", skipinitialspace=True)
                writer.writerow(definitions.HORUS_MATRIX_HEADER)
                writer.writerows(self.horus_matrix)
            else:
                raise Exception('format not implemented')
        except Exception as e:
            raise(e)

    def __get_horus_matrix_and_basic_statistics(self, sent_tokenize_list):

        df = pd.DataFrame(sent_tokenize_list)

        self.config.logger.info(':: %s sentence(s) cached' % str(len(sent_tokenize_list)))
        tot_sentences_with_entity = len(df.loc[df[0] == 1])
        tot_others = len(df.loc[df[0] == -1])
        self.config.logger.info(':: %s sentence(s) with entity' % tot_sentences_with_entity)
        self.config.logger.info(':: %s sentence(s) without entity' % tot_others)
        self.horus_matrix = self.util.sentence_to_horus_matrix(sent_tokenize_list)

        hm = pd.DataFrame(self.horus_matrix)
        self.config.logger.info(':: basic POS statistics')
        a = len(hm)  # all
        a2 = len(hm[(hm[7] == 0)])  # all excluding compounds
        plo = hm[(hm[7] == 0) & (hm[0] == 1)]  # all PLO entities (not compound)
        not_plo = hm[(hm[7] == 0) & (hm[0] == 0)]  # all PLO entities (not compound)

        pos_ok_plo = plo[(plo[5].isin(definitions.POS_NOUN_TAGS))]
        pos_not_ok_plo = plo[(~plo[5].isin(definitions.POS_NOUN_TAGS))]
        pos_noun_but_not_entity = not_plo[(not_plo[5].isin(definitions.POS_NOUN_TAGS))]

        self.config.logger.info(':: [basic statistics]')
        self.config.logger.info(':: -> ALL terms: %s ' % a)
        self.config.logger.info(':: -> ALL tokens (no compounds): %s (%.2f)' % (a2, (a2 / float(a))))
        self.config.logger.info(':: -> ALL NNs (no compounds nor entities): %s ' % len(pos_noun_but_not_entity))
        self.config.logger.info(':: [test dataset statistics]')
        self.config.logger.info(':: -> PLO entities (no compounds): %s (%.2f)' % (len(plo), len(plo) / float(a2)))
        self.config.logger.info(':: -> PLO entities correctly classified as NN (POS says is NOUN): %s (%.2f)' %
                          (len(pos_ok_plo), len(pos_ok_plo) / float(len(plo)) if len(plo) != 0 else 0))
        self.config.logger.info(':: -> PLO entities misclassified (POS says is NOT NOUN): %s (%.2f)' %
                          (len(pos_not_ok_plo), len(pos_not_ok_plo) / float(len(plo)) if len(plo) != 0 else 0))

    def __detect_and_translate(self, t1, t2, id, t1en, t2en):
        try:
            #if isinstance(t1, str):
            #    t1 = unicode(t1, "utf-8")
            #if isinstance(t2, str):
            #    t2 = unicode(t2, "utf-8")

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

            c.close()

            merged = ''
            if t1en is not None:
                merged = t1en.encode('ascii','ignore')
            if t2en is not None:
                merged = merged + ' ' + t2en.encode('ascii','ignore')

            return merged

        except Exception as e:
            raise e

    def __get_number_classes_in_embeedings(self, w, k):
        '''
        returns the number of terms existing in the set of returned words similar to w
        :param w: an input word
        :param k: a set of (predicted) classes
        :return: the number of elements from both k and the embeedings function
        '''
        try:
            most_similar = set(self.tools.word2vec_google.most_similar(positive=w, topn=5))
            return len(most_similar.union(set(k)))
        except:
            raise

    def extract_features(self):
        """
        receives a horus_matrix and iterate over the tokens, detecting objects for each image/document
        in a set of images/documents related to a given token
        :param horus_matrix: the horus_matrix
        :return: an updated horus matrix
        """
        self.config.logger.info(':: detecting %s objects...' % len(self.horus_matrix))
        auxi = 0
        toti = len(self.horus_matrix)
        for index in range(len(self.horus_matrix)):
            try:
                auxi += 1
                if (self.horus_matrix[index][5] in definitions.POS_NOUN_TAGS) or self.horus_matrix[index][7] == 1:

                    term = self.horus_matrix[index][3]
                    self.config.logger.info(':: token %d of %d [%s]' % (auxi, toti, term))

                    id_term_img = self.horus_matrix[index][10]
                    id_term_txt = self.horus_matrix[index][9]
                    id_ner_type = 0
                    tot_geral_faces =0
                    tot_geral_logos =0
                    tot_geral_locations =0
                    tot_geral_pos_locations =0
                    tot_geral_neg_locations =0
                    tot_geral_faces_cnn =0
                    tot_geral_logos_cnn =0
                    tot_geral_locations_cnn =0
                    tot_geral_pos_locations_cnn =0
                    tot_geral_neg_locations_cnn =0
                    tot_geral_faces_dlib = 0


                    with self.conn:
                        # -----------------------------------------------------------------
                        # image classification
                        # -----------------------------------------------------------------
                        if 1==1:
                            filesimg = []
                            cursor = self.conn.cursor()
                            cursor.execute(SQL_OBJECT_DETECTION_SEL % (id_term_img, id_ner_type))
                            rows = cursor.fetchall()
                            nr_results_img = len(rows)
                            if nr_results_img == 0:
                                self.config.logger.debug(":: term has not returned images!")
                            limit_img = min(nr_results_img, int(self.config.search_engine_tot_resources))

                            # 0 = file path | 1 = id | 2 = processed | 3=nr_faces | 4=nr_logos | 5 to 14=nr_places_1-to-10
                            # 14 = nr_faces_cnn, 15 = nr_logos_cnn, 16-25=nr_places_1-to-10_cnn
                            for i in range(limit_img):
                                img_full_path = self.config.cache_img_folder + rows[i][0]
                                if rows[i][2] == 1: # processed
                                    tot_geral_faces += rows[i][3] if rows[i][3] != None else 0
                                    tot_geral_logos += rows[i][4] if rows[i][4] != None else 0
                                    tot_geral_faces_cnn += rows[i][14] if rows[i][14] != None else 0
                                    tot_geral_logos_cnn += rows[i][15] if rows[i][15] != None else 0
                                    if (rows[i][5:14]).count(1) >= int(self.config.models_location_theta):
                                        tot_geral_locations += 1
                                    if (rows[i][16:25]).count(1) >= int(self.config.models_location_theta):
                                        tot_geral_locations_cnn += 1
                                    tot_geral_pos_locations += rows[i][5:14].count(1)
                                    tot_geral_neg_locations += (rows[i][5:14].count(0) * -1)
                                    tot_geral_pos_locations_cnn += rows[i][16:25].count(1)
                                    tot_geral_neg_locations_cnn += (rows[i][16:25].count(0) * -1)
                                    # TODO: to integrate (check index)
                                    tot_geral_faces_dlib += rows[i][26] if rows[i][26] != None else 0

                                else:
                                    tot_faces = 0
                                    tot_logos = 0
                                    res = [0] * 10
                                    tot_faces_cnn = 0
                                    tot_logos_cnn = 0
                                    res_cnn = [0] * 10
                                    tot_faces_dlib = 0

                                    if self.image_sift is not None:
                                        # ----- face recognition -----
                                        tot_faces = self.image_sift.detect_faces(rows[i][0])
                                        if tot_faces > 0:
                                            tot_geral_faces += 1
                                            self.config.logger.debug(":: found {0} faces!".format(tot_faces))
                                        # ----- logo recognition -----
                                        tot_logos = self.image_sift.detect_logo(rows[i][0])
                                        if tot_logos > 0:
                                            tot_geral_logos += 1
                                            self.config.logger.debug(":: found {0} logo(s)!".format(1))
                                        # ----- place recognition -----
                                        res = self.image_sift.detect_place(rows[i][0])
                                        tot_geral_pos_locations += res.count(1)
                                        tot_geral_neg_locations += (res.count(0) * -1)

                                    if res.count(1) >= int(self.config.models_location_theta):
                                        tot_geral_locations += 1
                                        self.config.logger.debug(":: found {0} place(s)!".format(1))

                                    if self.image_cnn is not None:
                                        image = self.image_cnn.preprocess_image(rows[i][0])
                                        if image is not False:
                                            # ----- face recognition -----
                                            tot_faces_cnn = self.image_cnn.detect_faces(image)
                                            if tot_faces_cnn > 0:
                                                tot_geral_faces_cnn += 1
                                                self.config.logger.debug(":: found {0} faces!".format(tot_faces))
                                            # ----- logo recognition -----
                                            tot_logos_cnn = self.image_cnn.detect_logo(image)
                                            if tot_logos_cnn > 0:
                                                tot_geral_logos_cnn += 1
                                                self.config.logger.debug(":: found {0} logo(s)!".format(1))
                                            # ----- place recognition -----
                                            res_cnn = self.image_cnn.detect_place(image)
                                            tot_geral_pos_locations_cnn += res_cnn.count(1)
                                            tot_geral_neg_locations_cnn += (res_cnn.count(0) * -1)

                                            if res_cnn.count(1) >= T:
                                                tot_geral_locations_cnn += 1
                                                self.config.logger.debug(":: found {0} place(s)!".format(1))

                                    if self.dlib_cnn is not None:
                                        tot_geral_faces_dlib += self.dlib_cnn.detect_faces_cnn(img_full_path)

                                    param = []
                                    param.append(tot_faces)
                                    param.append(tot_logos)
                                    param.extend(res)
                                    param.append(tot_faces_cnn)
                                    param.append(tot_logos_cnn)
                                    param.extend(res_cnn)
                                    param.append(rows[i][1])
                                    cursor.execute(SQL_OBJECT_DETECTION_UPD, param)

                            self.conn.commit()

                            outs = [tot_geral_locations, tot_geral_logos, tot_geral_faces]
                            maxs_cv = heapq.nlargest(2, outs)
                            dist_cv_indicator = max(maxs_cv) - min(maxs_cv)
                            place_cv_indicator = tot_geral_pos_locations + tot_geral_neg_locations

                            outs_cnn = [tot_geral_locations_cnn, tot_geral_logos_cnn, tot_geral_faces_cnn]
                            maxs_cv_cnn = heapq.nlargest(2, outs_cnn)
                            dist_cv_indicator_cnn = max(maxs_cv_cnn) - min(maxs_cv_cnn)
                            place_cv_indicator_cnn = tot_geral_pos_locations_cnn + tot_geral_neg_locations_cnn

                            self.horus_matrix[index][11] = limit_img
                            self.horus_matrix[index][12] = tot_geral_locations  # 1
                            self.horus_matrix[index][13] = tot_geral_logos  # 2
                            self.horus_matrix[index][14] = tot_geral_faces  # 3
                            self.horus_matrix[index][15] = dist_cv_indicator  # 4
                            self.horus_matrix[index][16] = place_cv_indicator  # 5
                            self.horus_matrix[index][17] = nr_results_img  # 5

                            self.horus_matrix[index][32] = tot_geral_locations_cnn  # 1
                            self.horus_matrix[index][33] = tot_geral_logos_cnn  # 2
                            self.horus_matrix[index][34] = tot_geral_faces_cnn  # 3
                            self.horus_matrix[index][35] = dist_cv_indicator_cnn  # 4
                            self.horus_matrix[index][36] = place_cv_indicator_cnn  # 5

                            self.config.logger.debug(':: CV statistics:[BOW: LOC=%s, ORG=%s, PER=%s, DIST=%s, PLC=%s | '
                                              'CNN: LOC=%s, ORG=%s, PER=%s, DIST=%s, PLC=%s]' %
                                              (str(tot_geral_locations).zfill(2), str(tot_geral_logos).zfill(2),
                                               str(tot_geral_faces).zfill(2), str(dist_cv_indicator).zfill(2),
                                               place_cv_indicator,
                                               str(tot_geral_locations_cnn).zfill(2), str(tot_geral_logos_cnn).zfill(2),
                                               str(tot_geral_faces_cnn).zfill(2), str(dist_cv_indicator_cnn).zfill(2),
                                               place_cv_indicator_cnn))

                            if limit_img != 0:
                                self.horus_matrix[index][18] = definitions.KLASSES[outs.index(max(outs)) + 1]
                            else:
                                self.horus_matrix[index][18] = definitions.KLASSES[4]

                        # -----------------------------------------------------------------
                        # text classification
                        # -----------------------------------------------------------------
                        if 1==1:
                            cnn_labels = []
                            tot_union_embeedings_cnn_classes = self.__get_number_classes_in_embeedings(term, cnn_labels)
                            #TODO: export this new feature

                            y_bow, y_tm = [], []
                            cursor.execute(SQL_TEXT_CLASS_SEL % (id_term_txt, id_ner_type))
                            rows = cursor.fetchall()

                            nr_results_txt = len(rows)
                            if nr_results_txt == 0:
                                self.config.logger.debug(":: term has not returned web sites!")
                            limit_txt = min(nr_results_txt, int(self.config.search_engine_tot_resources))

                            for itxt in range(limit_txt):
                                try:
                                    if rows[itxt][6] == 0 or rows[itxt][6] is None:  # not processed yet
                                        merged_en = self.__detect_and_translate(rows[itxt][2], rows[itxt][3], rows[itxt][0], rows[itxt][4], rows[itxt][5])
                                        ret_bow = [0,0,0,0,0]
                                        ret_tm = [0,0,0,0,0]
                                        if merged_en != '':
                                            if self.text_bow is not None:
                                                ret_bow = self.text_bow.detect_text_klass(merged_en)
                                            if self.text_tm is not None:
                                                ret_tm = self.text_tm.detect_text_klass(merged_en)

                                        y_bow.append(ret_bow)
                                        y_tm.append(ret_tm)

                                        cursor.execute(SQL_TEXT_CLASS_UPD % (ret_bow[0], ret_bow[1], ret_bow[2], ret_bow[3], ret_bow[4],
                                                                       ret_tm[0], ret_tm[1], ret_tm[2], ret_tm[3], ret_tm[4],
                                                                       rows[itxt][0]))
                                    else:
                                        y_bow.append(rows[itxt][7:12])
                                        y_tm.append(rows[itxt][12:18])
                                except Exception as e:
                                    self.config.logger.error(str(e.message))
                                    pass

                            self.conn.commit()

                            yyb = numpy.array(y_bow)
                            yytm = numpy.array(y_tm)

                            gpb = [numpy.count_nonzero(yyb == 1), numpy.count_nonzero(yyb == 2), numpy.count_nonzero(yyb == 3)]
                            horus_tx_ner = gpb.index(max(gpb)) + 1

                            self.horus_matrix[index][19] = limit_txt
                            self.horus_matrix[index][20] = gpb[0]
                            self.horus_matrix[index][21] = gpb[1]
                            self.horus_matrix[index][22] = gpb[2]
                            self.horus_matrix[index][23] = 0

                            self.horus_matrix[index][28] = 0 if len(y_tm) == 0 else numpy.sum(yytm, axis=0)[0]
                            self.horus_matrix[index][29] = 0 if len(y_tm) == 0 else numpy.sum(yytm, axis=0)[1]
                            self.horus_matrix[index][30] = 0 if len(y_tm) == 0 else numpy.sum(yytm, axis=0)[2]

                            maxs_tx = heapq.nlargest(2, gpb)
                            maxs_tm = 0 if len(y_tm) == 0 else heapq.nlargest(2, numpy.sum(yytm, axis=0))
                            dist_tx_indicator = max(maxs_tx) - min(maxs_tx)
                            dist_tx_indicator_tm = 0 if len(y_tm) == 0 else (max(maxs_tm) - min(maxs_tm))

                            self.horus_matrix[index][24] = dist_tx_indicator
                            self.horus_matrix[index][25] = nr_results_txt
                            self.horus_matrix[index][31] = dist_tx_indicator_tm

                            if limit_txt != 0:
                                self.horus_matrix[index][26] = definitions.KLASSES[horus_tx_ner]
                            else:
                                self.horus_matrix[index][26] = definitions.KLASSES[4]

                        self.config.logger.debug(':: TX statistics:'
                                           '[BoW: LOC=%s, ORG=%s, PER=%s, DIST=%s | ' 'TM: LOC=%s, ORG=%s, PER=%s, DIST=%s]' %
                                           (str(self.horus_matrix[index][20]).zfill(2), str(self.horus_matrix[index][21]).zfill(2),
                                            str(self.horus_matrix[index][22]).zfill(2), str(dist_tx_indicator).zfill(2),
                                            "{:.2f}".format(self.horus_matrix[index][28]).zfill(2), "{:.2f}".format(self.horus_matrix[index][29]).zfill(2),
                                            "{:.2f}".format(self.horus_matrix[index][30]).zfill(2), "{:.2f}".format(dist_tx_indicator_tm).zfill(2)))
                        self.config.logger.debug('-------------------------------------------------------------')


            except Exception as e:
                self.config.logger.error(repr(e))
                self.conn.rollback()
                self.horus_matrix[index] = [0] * int(definitions.HORUS_TOT_FEATURES)

        return self.horus_matrix

    def extract_features_from_text(self, text):
        """
        extracts the HORUS features for a given input text
        :param text: the input text
        :return: the features
        """
        try:
            if text is None:
                raise Exception("Provide an input text")
            self.config.logger.info(':: text: ' + text)
            self.config.logger.info(':: tokenizing sentences ...')
            sent_tokenize_list = nltk.sent_tokenize(text, language='english')
            self.config.logger.info(':: processing ' + str(len(sent_tokenize_list)) + ' sentence(s).')
            sentences = []
            for sentence in sent_tokenize_list:
                sentences.append(self.util.process_and_save_sentence(-1, sentence))

            self.util.sentence_to_horus_matrix(sentences)
            self.util.download_and_cache_results(self.horus_matrix)
            self.extract_features()
            self.__export_data(self.config.dir_output + 'outfeatures.horus', 'json')

            return self.horus_matrix

        except Exception as error:
            self.config.logger.error('extract_features1() error: ' + repr(error))

    def extract_features_from_conll(self, file, out_subfolder, label=None, token_index=0, ner_index=1):
        """
        generates the feature_extraction data for HORUS
        do not use the config file to choose the models, exports all features (self.detect_objects())
        :param file: a dataset (CoNLL format)
        :param label: a dataset label
        :param token_index: column index of the token (word)
        :param ner_index: column index if the target class (NER)
        :return: the feature file
        """
        try:
            if file is None:
                raise Exception("Provide an input file format to be annotated")
            self.config.logger.info(':: processing CoNLL format -> %s' % label)
            file = self.config.dir_datasets + file
            sent_tokenize_list = self.util.process_ds_conll_format(file, label, token_index, ner_index, '')
            self.__get_horus_matrix_and_basic_statistics(sent_tokenize_list)
            if len(self.horus_matrix) > 0:
                self.util.download_and_cache_results(self.horus_matrix)
                self.extract_features()
                filename = self.util.path_leaf(file) + ".horus"

                path = self.config.dir_output + out_subfolder + filename
                self.__export_data(path, 'tsv')
            else:
                self.config.logger.warn(':: well, nothing to do today...')

            return self.horus_matrix

        except Exception as error:
            self.config.logger.error('extract_features2() error: ' + repr(error))


if __name__ == "__main__":
    if len(sys.argv) not in (1,2,3,4):
        print("please inform: 1: data set and 2: column indexes ([1, .., n])")
    else:
        config = HorusConfig()
        # args[0], args[1], args[2], args[3]
        tot_args = 1 #len(sys.argv)
        data = 'coNLL2003/coNLL2003.eng.testb'  # args[0]
        data = 'paris hilton was once the toast of the town' #args[0]

        if tot_args == 1:
            extractor = FeatureExtraction(config)
            out = extractor.extract_features_from_text(data)
            #outjson = json.dumps(out)
            print(out)
            #print(outjson)
        else:
            exp_folder = 'EXP_002/' #
            extractor = FeatureExtraction(config, load_sift=1, load_tfidf=1, load_cnn=0, load_topic_modeling=0)
            extractor.extract_features_from_conll('Ritter/ner.txt', exp_folder, label='ritter')
            # extractor.extract_features('Ritter/ner_one_sentence.txt', exp_folder, 'ritter_sample')
            # extractor.extract_features('wnut/2016.conll.freebase.ascii.txt', exp_folder, 'wnut15')
            # extractor.extract_features('wnut/2015.conll.freebase', exp_folder, 'wnut16')
            ## attention: change POS tag lib in the HORUS.ini to NLTK before run this
            # extractor.extract_features('coNLL2003/nodocstart_coNLL2003.eng.testA', exp_folder, 'conll03', 0, 3)
            #extractor.extract_features_conll(data, exp_folder, 'conll03b', 0, 3)