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
import logging
import sys
import sqlite3
import nltk
import numpy
import pandas as pd
from horus.core.config import HorusConfig
from horus.core.feature_extraction.object_detection.cnn import CNN
from horus.core.feature_extraction.object_detection.sift import SIFT
from horus.core.feature_extraction.util import Util
from horus.core.util.nlp_tools import NLPTools
from horus.core.util.systemlog import SystemLog
from horus.core.util.definitions_sql import *
# print cv2.__version__
from horus.core.feature_extraction.text_classification.bow_tfidf import BowTfidf
from horus.core.feature_extraction.text_classification.topic_modeling import TopicModeling
from horus.core.translation.bingtranslation import BingTranslator
from horus.core.util import definitions


class FeatureExtraction(object):
    """ Description:
            A core module for feature_extraction the algorithm
        Attributes:
            None
    """
    def __init__(self):
        self.logging = SystemLog("horus.log", logging.DEBUG, logging.DEBUG)
        self.horus_matrix = []
        self.config = HorusConfig()
        self.logging.log.info('------------------------------------------------------------------')
        self.logging.log.info('::                       HORUS ' + self.config.version + '                            ::')
        self.logging.log.info('------------------------------------------------------------------')
        self.logging.log.info(':: loading components...')
        self.util = Util()
        #self.tools = NLPTools()
        #self.translator = BingTranslator()
        self.image_cnn = CNN()
        self.image_sift = SIFT()
        self.text_bow = BowTfidf()
        self.text_tm = TopicModeling()
        self.conn = sqlite3.connect(self.config.database_db)

        if bool(int(self.config.models_force_download)) is True:
            try:
                nltk.data.find('averaged_perceptron_tagger.zip')
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
            try:
                nltk.data.find('punkt.zip')
            except LookupError:
                nltk.download('punkt')
            try:
                nltk.data.find('maxent_ne_chunker.zip')
            except LookupError:
                nltk.download('maxent_ne_chunker')
            try:
                nltk.data.find('universal_tagset.zip')
            except LookupError:
                nltk.download('universal_tagset')
            try:
                nltk.data.find('words.zip')
            except LookupError:
                nltk.download('words')

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.conn.close()
        except:
            pass

    def __export_data(self, file, subfolder, format):
        temp = self.config.output_path + subfolder + '/' + file
        self.logging.log.info(':: exporting metadata to: ' + temp + "." + format)
        if format == 'json':
            with open(temp + '.json', 'wb') as outfile:
                json.dump(self.horus_matrix, outfile)
        elif format == 'csv':
            writer = csv.writer(open(temp + '.csv', 'wb'), quoting=csv.QUOTE_ALL)
            writer.writerow(definitions.HORUS_MATRIX_HEADER)
            # writer.writerow([s.encode('utf8') if type(s) is unicode else s for s in self.horus_matrix])
            writer.writerows(self.horus_matrix)
        elif format == 'tsv':
            writer = csv.writer(temp + '.tsv', dialect="excel", delimiter="\t", skipinitialspace=True)
            writer.writerow(definitions.HORUS_MATRIX_HEADER)
            writer.writerows(self.horus_matrix)
        else:
            raise Exception('format not implemented')

    def __get_horus_matrix_and_basic_statistics(self, sent_tokenize_list):

        df = pd.DataFrame(sent_tokenize_list)

        self.logging.log.info(':: %s sentence(s) cached' % str(len(sent_tokenize_list)))
        tot_sentences_with_entity = len(df.loc[df[0] == 1])
        tot_others = len(df.loc[df[0] == -1])
        self.logging.log.info(':: %s sentence(s) with entity' % tot_sentences_with_entity)
        self.logging.log.info(':: %s sentence(s) without entity' % tot_others)
        self.horus_matrix = self.util.sentence_to_horus_matrix(sent_tokenize_list)

        hm = pd.DataFrame(self.horus_matrix)
        self.logging.log.info(':: basic POS statistics')
        a = len(hm)  # all
        a2 = len(hm[(hm[7] == 0)])  # all excluding compounds
        plo = hm[(hm[7] == 0) & (hm[0] == 1)]  # all PLO entities (not compound)
        not_plo = hm[(hm[7] == 0) & (hm[0] == 0)]  # all PLO entities (not compound)

        pos_ok_plo = plo[(plo[5].isin(definitions.POS_NOUN_TAGS))]
        pos_not_ok_plo = plo[(~plo[5].isin(definitions.POS_NOUN_TAGS))]
        pos_noun_but_not_entity = not_plo[(not_plo[5].isin(definitions.POS_NOUN_TAGS))]

        self.logging.log.info(':: [basic statistics]')
        self.logging.log.info(':: -> ALL terms: %s ' % a)
        self.logging.log.info(':: -> ALL tokens (no compounds): %s (%.2f)' % (a2, (a2 / float(a))))
        self.logging.log.info(':: -> ALL NNs (no compounds nor entities): %s ' % len(pos_noun_but_not_entity))
        self.logging.log.info(':: [test dataset statistics]')
        self.logging.log.info(':: -> PLO entities (no compounds): %s (%.2f)' % (len(plo), len(plo) / float(a2)))
        self.logging.log.info(':: -> PLO entities correctly classified as NN (POS says is NOUN): %s (%.2f)' %
                          (len(pos_ok_plo), len(pos_ok_plo) / float(len(plo)) if len(plo) != 0 else 0))
        self.logging.log.info(':: -> PLO entities misclassified (POS says is NOT NOUN): %s (%.2f)' %
                          (len(pos_not_ok_plo), len(pos_not_ok_plo) / float(len(plo)) if len(plo) != 0 else 0))

    def detect_objects(self, matrix):
        """
        receives a horus_matrix and iterate over the tokens, detecting objects for each image/document
        in a set of images/documents related to a given token
        :param horus_matrix: the horus_matrix
        :return: an updated horus matrix
        """
        horus_matrix = matrix
        if self.config.object_detection_type not in (-1, 0,1):
            raise Exception('parameter value not implemented: ' + str(self.config.object_detection_type))
        if self.config.text_classification_type not in (-1, 0, 1):
            raise Exception('parameter value not implemented: ' + str(self.config.text_classification_type))
        self.logging.log.info(':: detecting %s objects...' % len(horus_matrix))
        auxi = 0
        toti = len(horus_matrix)
        for index in range(len(horus_matrix)):
            auxi += 1
            if (horus_matrix[index][5] in definitions.POS_NOUN_TAGS) or horus_matrix[index][7] == 1:

                term = horus_matrix[index][3]
                self.logging.log.info(':: token %d of %d [%s]' % (auxi, toti, term))

                id_term_img = horus_matrix[index][10]
                id_term_txt = horus_matrix[index][9]
                id_ner_type = 0

                tot_geral_faces, tot_geral_logos, tot_geral_locations, tot_geral_pos_locations, \
                    tot_geral_neg_locations = 0

                tot_geral_faces_cnn, tot_geral_logos_cnn, tot_geral_locations_cnn, tot_geral_pos_locations_cnn, \
                    tot_geral_neg_locations_cnn = 0

                T = int(self.config.models_location_theta)  # location threshold

                # -----------------------------------------------------------------
                # image classification
                # -----------------------------------------------------------------

                filesimg = []
                with self.conn:
                    cursor = self.conn.cursor()
                    cursor.execute(SQL_OBJECT_DETECTION_SQL % (id_term_img, id_ner_type))
                    rows = cursor.fetchall()
                    nr_results_img = len(rows)
                    if nr_results_img == 0:
                        self.logging.log.debug(":: term has not returned images!")
                    limit_img = min(nr_results_img, int(self.config.search_engine_tot_resources))

                    # 0 = file path | 1 = id | 2 = processed | 3=nr_faces | 4=nr_logos | 5 to 14=nr_places_1-to-10
                    # 14 = nr_faces_cnn, 15 = nr_logos_cnn, 16-25=nr_places_1-to-10_cnn
                    for i in range(limit_img):
                        temp=[]
                        temp.append(self.config.cache_img_folder + rows[i][0])
                        temp.extend(rows[i][1:25])
                        filesimg.append(temp)

                for ifeat in filesimg:
                    if ifeat[2] == 1: # processed
                        tot_geral_faces += ifeat[3]
                        tot_geral_logos += ifeat[4]
                        tot_geral_faces_cnn += ifeat[14]
                        tot_geral_logos_cnn += ifeat[15]
                        if (ifeat[5:14]).count(1) >= int(T):
                            tot_geral_locations += 1
                        if (ifeat[16:25]).count(1) >= int(T):
                            tot_geral_locations_cnn += 1
                        tot_geral_pos_locations += ifeat[5:14].count(1)
                        tot_geral_neg_locations += (ifeat[5:14].count(-1) * -1)
                        tot_geral_pos_locations_cnn += ifeat[16:25].count(1)
                        tot_geral_neg_locations_cnn += (ifeat[16:25].count(-1) * -1)

                    else:
                        if self.config.object_detection_type in (-1,0):
                            # ----- face recognition -----
                            tot_faces = self.image_sift.detect_faces(ifeat[0])
                            if tot_faces > 0:
                                tot_geral_faces += 1
                                self.logging.log.debug(":: found {0} faces!".format(tot_faces))
                            # ----- logo recognition -----
                            tot_logos = self.image_sift.detect_logo(ifeat[0])
                            if tot_logos[0] == 1:
                                tot_geral_logos += 1
                                self.logging.log.debug(":: found {0} logo(s)!".format(1))
                            # ----- place recognition -----
                            res = self.image_sift.detect_place(ifeat[0])
                            tot_geral_pos_locations += res.count(1)
                            tot_geral_neg_locations += (res.count(-1) * -1)

                            if res.count(1) >= T:
                                tot_geral_locations += 1
                                self.logging.log.debug(":: found {0} place(s)!".format(1))

                        elif self.config.object_detection_type in (-1,1):
                            image = self.image_cnn.preprocess_image(ifeat[0])
                            # ----- face recognition -----
                            tot_faces = self.image_cnn.detect_faces(image)
                            if tot_faces > 0:
                                tot_geral_faces_cnn += 1
                                self.logging.log.debug(":: found {0} faces!".format(tot_faces))
                            # ----- logo recognition -----
                            tot_logos = self.image_cnn.detect_logo_cnn(image)
                            if tot_logos[0] == 1:
                                tot_geral_logos_cnn += 1
                                self.logging.log.debug(":: found {0} logo(s)!".format(1))
                            # ----- place recognition -----
                            res = self.image_cnn.detect_place_cnn(image)
                            tot_geral_pos_locations_cnn += res.count(1)
                            tot_geral_neg_locations_cnn += (res.count(0) * -1)

                            if res.count(1) >= T:
                                tot_geral_locations_cnn += 1
                                self.logging.log.debug(":: found {0} place(s)!".format(1))

                        # updating results
                        if self.config.object_detection_type == 0:  # SIFT
                            _sql = sql_object_0_upd
                        elif self.config.object_detection_type == 1:  # CNN
                            _sql = sql_object_1_upd
                        elif self.config.object_detection_type == -1: #ALL
                            _sql = sql_object_upd
                        else:
                            raise Exception('parameter value not implemented: ' + str(self.config.object_detection_type))

                        param = []
                        param.append(tot_faces)
                        param.append(tot_logos[0]) if tot_logos[0] == 1 else param.append(0)
                        param.extend(res)
                        param.append(ifeat[1])
                        cursor.execute(sql_object_upd, param)

                self.conn.commit()

                outs = [tot_geral_locations, tot_geral_logos, tot_geral_faces]
                maxs_cv = heapq.nlargest(2, outs)
                dist_cv_indicator = max(maxs_cv) - min(maxs_cv)
                place_cv_indicator = tot_geral_pos_locations + tot_geral_neg_locations

                horus_matrix[index][11] = limit_img
                horus_matrix[index][12] = tot_geral_locations  # 1
                horus_matrix[index][13] = tot_geral_logos  # 2
                horus_matrix[index][14] = tot_geral_faces  # 3
                horus_matrix[index][15] = dist_cv_indicator  # 4
                horus_matrix[index][16] = place_cv_indicator  # 5
                horus_matrix[index][17] = nr_results_img  # 5

                self.logging.log.debug(':: CV statistics:'
                                   '(LOC=%s, ORG=%s, PER=%s, DIST=%s, PLC=%s)' %
                                   (str(tot_geral_locations).zfill(2), str(tot_geral_logos).zfill(2),
                                    str(tot_geral_faces).zfill(2), str(dist_cv_indicator).zfill(2), place_cv_indicator))

                if limit_img != 0:
                    horus_matrix[index][18] = definitions.KLASSES[outs.index(max(outs)) + 1]
                else:
                    horus_matrix[index][18] = definitions.KLASSES[4]

                # -----------------------------------------------------------------
                # text classification
                # -----------------------------------------------------------------
                y = []
                with self.conn:
                    cursor = self.conn.cursor()
                    if self.config.text_classification_type == 0:  # SIFT
                        _sql = sql_text_0_sel
                    elif self.config.text_classification_type == 1:  # CNN
                        _sql = sql_text_1_sel

                    cursor.execute(_sql % (id_term_txt, id_ner_type))
                    rows = cursor.fetchall()

                    nr_results_txt = len(rows)
                    if nr_results_txt == 0:
                        self.logging.log.debug(":: term has not returned web sites!")
                    limit_txt = min(nr_results_txt, int(self.config.search_engine_tot_resources))

                    tot_err = 0
                    for itxt in range(limit_txt):

                        if rows[itxt][6] == 0 or rows[itxt][6] is None:  # not processed yet
                            text_merged = rows[itxt][2], rows[itxt][3], rows[itxt][0], rows[itxt][4], rows[itxt][5]
                            text_en = self.util.translate(text_merged)

                            if self.config.text_classification_type == 0:
                                ret = self.text_bow.detect_text_klass(text_en)
                                _sql = sql_text_0_upd
                            elif self.config.text_classification_type == 1:
                                ret = self.text_tm.detect_text_klass(text_en)
                                _sql = sql_text_1_upd

                            y.append(ret)
                            sql = _sql % (ret[0], ret[1], ret[2], ret[3], ret[4], rows[itxt][0])
                            cursor.execute(sql)
                            if ret[0] == -1 or ret[1] == -1 or ret[2] == -1 or ret[3] == -1 or ret[4] == -1:
                                tot_err += 1
                        else:
                            y.append(rows[itxt][7:12])

                    self.conn.commit()

                    yy = numpy.array(y)
                    gp = [numpy.count_nonzero(yy == 1), numpy.count_nonzero(yy == 2), numpy.count_nonzero(yy == 3)]
                    horus_tx_ner = gp.index(max(gp)) + 1

                    horus_matrix[index][19] = limit_txt
                    horus_matrix[index][20] = gp[0]
                    horus_matrix[index][21] = gp[1]
                    horus_matrix[index][22] = gp[2]
                    horus_matrix[index][23] = float(tot_err)

                    maxs_tx = heapq.nlargest(2, gp)
                    dist_tx_indicator = max(maxs_tx) - min(maxs_tx)

                    horus_matrix[index][24] = dist_tx_indicator
                    horus_matrix[index][25] = nr_results_txt

                    self.logging.log.debug(':: TX statistics:'
                                       '(LOC=%s, ORG=%s, PER=%s, DIST=%s, ERR.TRANS=%s)' %
                                       (str(gp[0]).zfill(2), str(gp[1]).zfill(2), str(gp[2]).zfill(2),
                                        str(dist_tx_indicator).zfill(2),
                                        str(tot_err / float(limit_txt)) if limit_txt > 0 else 0))
                    self.logging.log.debug('-------------------------------------------------------------')

                    if limit_txt != 0:
                        horus_matrix[index][26] = definitions.KLASSES[horus_tx_ner]
                    else:
                        horus_matrix[index][26] = definitions.KLASSES[4]

                    # checking final NER based on:
                    #  -> theta
                    if horus_matrix[index][15] >= int(self.config.models_distance_theta):
                        horus_matrix[index][36] = horus_matrix[index][18]  # CV is the final decision
                        horus_matrix[index][39] = horus_matrix[index][36]  # compound prediction initial
                    elif horus_matrix[index][24] >= int(self.config.models_distance_theta):
                        horus_matrix[index][36] = horus_matrix[index][26]  # TX is the final decision
                        horus_matrix[index][39] = horus_matrix[index][36]  # compound prediction initial
                    #  -> theta+1
                    if horus_matrix[index][15] >= int(self.config.models_distance_theta) + 1:
                        horus_matrix[index][37] = horus_matrix[index][18]  # CV is the final decision
                    elif horus_matrix[index][24] >= int(self.config.models_distance_theta) + 1:
                        horus_matrix[index][37] = horus_matrix[index][26]  # TX is the final decision
                    #  -> theta+2
                    if horus_matrix[index][15] >= int(self.config.models_distance_theta) + 2:
                        horus_matrix[index][38] = horus_matrix[index][18]  # CV is the final decision
                    elif horus_matrix[index][24] >= int(self.config.models_distance_theta) + 2:
                        horus_matrix[index][38] = horus_matrix[index][26]  # TX is the final decision
        return horus_matrix

    def extract_features(self, file, out_subfolder, label=None, token_index=0, ner_index=1):
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
            print self.version_label
            if file is None:
                raise Exception("Provide an input file format to be annotated")
            self.logging.log.info(':: processing CoNLL format -> %s' % label)
            file = self.config.dataset_path + file
            sent_tokenize_list = self.util.process_ds_conll_format(file, label, token_index, ner_index, '')
            self.__get_horus_matrix_and_basic_statistics(sent_tokenize_list)
            if len(self.horus_matrix) > 0:
                self.util.download_and_cache_results()
                self.horus_matrix = self.detect_objects(self.horus_matrix)
                outfilename = self.util.path_leaf(file) + ".horus"
                self.__export_data(outfilename, out_subfolder, 'tsv')
                self.logging.log.info(':: feature extraction completed! filename = ' + outfilename)
            else:
                self.logging.log.warn(':: nothing to do...')

            return self.horus_matrix

        except Exception as error:
            self.logging.log.error('caught this error here: ' + repr(error))


if __name__ == "__main__":
    if len(sys.argv) not in (1,2,3,4):
        print "please inform: 1: data set and 2: column indexes ([1, .., n])"
    else:
        #args[0], args[1], args[2], args[3]
        FeatureExtraction().extract_features()