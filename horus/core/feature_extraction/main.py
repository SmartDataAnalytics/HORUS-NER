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
from horus.core.util.systemlog import SysLogger
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
        self.horus_matrix = []
        self.config = HorusConfig()
        self.logger = SysLogger().getLog()
        self.logger.info('------------------------------------------------------------------')
        self.logger.info('::                       HORUS ' + self.config.version + '                            ::')
        self.logger.info('------------------------------------------------------------------')
        self.logger.info(':: loading components...')
        self.util = Util(self.config)
        #self.tools = NLPTools()
        #self.translator = BingTranslator(self.config)
        self.translator = None
        self.logger.info(':: loading CNN')
        self.image_cnn = CNN(self.config)
        self.logger.info(':: loading SIFT')
        self.image_sift = SIFT(self.config)
        self.logger.info(':: loading BoW')
        self.text_bow = BowTfidf(self.config)
        self.logger.info(':: loading TM')
        self.text_tm = TopicModeling(self.config)
        self.logger.info(':: database connecting ...')
        self.conn = sqlite3.connect(self.config.database_db)

        if bool(int(self.config.models_force_download)) is True:
            self.logger.info(':: downloading NLTK data...')
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
        self.logger.info(':: exporting metadata to: ' + temp + "." + format)
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

        self.logger.info(':: %s sentence(s) cached' % str(len(sent_tokenize_list)))
        tot_sentences_with_entity = len(df.loc[df[0] == 1])
        tot_others = len(df.loc[df[0] == -1])
        self.logger.info(':: %s sentence(s) with entity' % tot_sentences_with_entity)
        self.logger.info(':: %s sentence(s) without entity' % tot_others)
        self.horus_matrix = self.util.sentence_to_horus_matrix(sent_tokenize_list)

        hm = pd.DataFrame(self.horus_matrix)
        self.logger.info(':: basic POS statistics')
        a = len(hm)  # all
        a2 = len(hm[(hm[7] == 0)])  # all excluding compounds
        plo = hm[(hm[7] == 0) & (hm[0] == 1)]  # all PLO entities (not compound)
        not_plo = hm[(hm[7] == 0) & (hm[0] == 0)]  # all PLO entities (not compound)

        pos_ok_plo = plo[(plo[5].isin(definitions.POS_NOUN_TAGS))]
        pos_not_ok_plo = plo[(~plo[5].isin(definitions.POS_NOUN_TAGS))]
        pos_noun_but_not_entity = not_plo[(not_plo[5].isin(definitions.POS_NOUN_TAGS))]

        self.logger.info(':: [basic statistics]')
        self.logger.info(':: -> ALL terms: %s ' % a)
        self.logger.info(':: -> ALL tokens (no compounds): %s (%.2f)' % (a2, (a2 / float(a))))
        self.logger.info(':: -> ALL NNs (no compounds nor entities): %s ' % len(pos_noun_but_not_entity))
        self.logger.info(':: [test dataset statistics]')
        self.logger.info(':: -> PLO entities (no compounds): %s (%.2f)' % (len(plo), len(plo) / float(a2)))
        self.logger.info(':: -> PLO entities correctly classified as NN (POS says is NOUN): %s (%.2f)' %
                          (len(pos_ok_plo), len(pos_ok_plo) / float(len(plo)) if len(plo) != 0 else 0))
        self.logger.info(':: -> PLO entities misclassified (POS says is NOT NOUN): %s (%.2f)' %
                          (len(pos_not_ok_plo), len(pos_not_ok_plo) / float(len(plo)) if len(plo) != 0 else 0))

    def __detect_and_translate(self, t1, t2, id, t1en, t2en):
        try:

            t1final = t1
            t2final = t1
            if isinstance(t1, str):
                t1 = unicode(t1, "utf-8")
            if isinstance(t2, str):
                t2 = unicode(t2, "utf-8")
            c = self.conn.cursor()
            if t1en is None:
                lt1 = self.translator.detect_language(t1)
                if lt1 != 'en':
                    t1final = self.translator.translate(t1, 'en')
                sql = """UPDATE HORUS_SEARCH_RESULT_TEXT SET result_title_en = ? WHERE id = ?"""
                c.execute(sql, (t1final.encode("utf-8"), id))
            else:
                t1final = t1en

            if t2en is None:
                lt2 = self.translator.detect_language(t2)
                if lt2 != 'en':
                    t2final = self.translator.translate(t2, 'en')
                sql = """UPDATE HORUS_SEARCH_RESULT_TEXT SET result_description_en = ? WHERE id = ?"""
                c.execute(sql, (t2final.encode("utf-8"), id))
            else:
                t2final = t2en

            c.close()

            return "{} {}".format(t1final.encode("utf-8"), t2final.encode("utf-8"))

        except Exception as e:
            self.sys.log.error(':: Error: ' + str(e))
            raise

    def detect_objects(self, matrix):
        """
        receives a horus_matrix and iterate over the tokens, detecting objects for each image/document
        in a set of images/documents related to a given token
        :param horus_matrix: the horus_matrix
        :return: an updated horus matrix
        """
        horus_matrix = matrix
        #if self.config.object_detection_type not in (-1, 0,1):
        #    raise Exception('parameter value not implemented: ' + str(self.config.object_detection_type))
        #if self.config.text_classification_type not in (-1, 0, 1):
        #    raise Exception('parameter value not implemented: ' + str(self.config.text_classification_type))
        self.logger.info(':: detecting %s objects...' % len(horus_matrix))
        auxi = 0
        toti = len(horus_matrix)
        for index in range(len(horus_matrix)):
            auxi += 1
            if (horus_matrix[index][5] in definitions.POS_NOUN_TAGS) or horus_matrix[index][7] == 1:

                term = horus_matrix[index][3]
                self.logger.info(':: token %d of %d [%s]' % (auxi, toti, term))

                id_term_img = horus_matrix[index][10]
                id_term_txt = horus_matrix[index][9]
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

                T = int(self.config.models_location_theta)  # location threshold

                # -----------------------------------------------------------------
                # image classification
                # -----------------------------------------------------------------

                filesimg = []
                with self.conn:
                    cursor = self.conn.cursor()
                    cursor.execute(SQL_OBJECT_DETECTION_SEL % (id_term_img, id_ner_type))
                    rows = cursor.fetchall()
                    nr_results_img = len(rows)
                    if nr_results_img == 0:
                        self.logger.debug(":: term has not returned images!")
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
                        tot_geral_faces += ifeat[3] if ifeat[3] != None else 0
                        tot_geral_logos += ifeat[4] if ifeat[4] != None else 0
                        tot_geral_faces_cnn += ifeat[14] if ifeat[14] != None else 0
                        tot_geral_logos_cnn += ifeat[15] if ifeat[15] != None else 0
                        if (ifeat[5:14]).count(1) >= int(T):
                            tot_geral_locations += 1
                        if (ifeat[16:25]).count(1) >= int(T):
                            tot_geral_locations_cnn += 1
                        tot_geral_pos_locations += ifeat[5:14].count(1)
                        tot_geral_neg_locations += (ifeat[5:14].count(0) * -1)
                        tot_geral_pos_locations_cnn += ifeat[16:25].count(1)
                        tot_geral_neg_locations_cnn += (ifeat[16:25].count(0) * -1)

                    else:

                        # ----- face recognition -----
                        tot_faces = self.image_sift.detect_faces(ifeat[0])
                        if tot_faces > 0:
                            tot_geral_faces += 1
                            self.logger.debug(":: found {0} faces!".format(tot_faces))
                        # ----- logo recognition -----
                        tot_logos = self.image_sift.detect_logo(ifeat[0])
                        if tot_logos > 0:
                            tot_geral_logos += 1
                            self.logger.debug(":: found {0} logo(s)!".format(1))
                        # ----- place recognition -----
                        res = self.image_sift.detect_place(ifeat[0])
                        tot_geral_pos_locations += res.count(1)
                        tot_geral_neg_locations += (res.count(0) * -1)

                        if res.count(1) >= T:
                            tot_geral_locations += 1
                            self.logger.debug(":: found {0} place(s)!".format(1))


                        image = self.image_cnn.preprocess_image(ifeat[0])

                        # ----- face recognition -----
                        tot_faces_cnn = self.image_cnn.detect_faces(image)
                        if tot_faces_cnn > 0:
                            tot_geral_faces_cnn += 1
                            self.logger.debug(":: found {0} faces!".format(tot_faces))
                        # ----- logo recognition -----
                        tot_logos_cnn = self.image_cnn.detect_logo_cnn(image)
                        if tot_logos_cnn > 0:
                            tot_geral_logos_cnn += 1
                            self.logger.debug(":: found {0} logo(s)!".format(1))
                        # ----- place recognition -----
                        res_cnn = self.image_cnn.detect_place_cnn(image)
                        tot_geral_pos_locations_cnn += res_cnn.count(1)
                        tot_geral_neg_locations_cnn += (res_cnn.count(0) * -1)

                        if res_cnn.count(1) >= T:
                            tot_geral_locations_cnn += 1
                            self.logger.debug(":: found {0} place(s)!".format(1))

                        param = []
                        param.append(tot_faces)
                        param.append(tot_logos)
                        param.extend(res)
                        param.append(tot_faces_cnn)
                        param.append(tot_logos_cnn)
                        param.extend(res_cnn)
                        param.append(ifeat[1])
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

                horus_matrix[index][11] = limit_img
                horus_matrix[index][12] = tot_geral_locations_cnn  # 1
                horus_matrix[index][13] = tot_geral_logos_cnn  # 2
                horus_matrix[index][14] = tot_geral_faces_cnn  # 3
                horus_matrix[index][15] = dist_cv_indicator_cnn  # 4
                horus_matrix[index][16] = place_cv_indicator  # 5
                horus_matrix[index][17] = nr_results_img  # 5

                horus_matrix[index][32] = tot_geral_locations_cnn  # 1
                horus_matrix[index][33] = tot_geral_logos_cnn  # 2
                horus_matrix[index][34] = tot_geral_faces_cnn  # 3
                horus_matrix[index][35] = dist_cv_indicator_cnn  # 4
                horus_matrix[index][36] = place_cv_indicator_cnn  # 5

                self.logger.debug(':: CV statistics:'
                                   '(BOW: LOC=%s, ORG=%s, PER=%s, DIST=%s, PLC=%s)'
                                       '(CNN: LOC=%s, ORG=%s, PER=%s, DIST=%s, PLC=%s)' %
                                   (str(tot_geral_locations).zfill(2), str(tot_geral_logos).zfill(2),
                                    str(tot_geral_faces).zfill(2), str(dist_cv_indicator).zfill(2), place_cv_indicator,
                                    str(tot_geral_locations_cnn).zfill(2), str(tot_geral_logos_cnn).zfill(2),
                                    str(tot_geral_faces_cnn).zfill(2), str(dist_cv_indicator_cnn).zfill(2), place_cv_indicator_cnn
                                    ))

                if limit_img != 0:
                    horus_matrix[index][18] = definitions.KLASSES[outs.index(max(outs)) + 1]
                else:
                    horus_matrix[index][18] = definitions.KLASSES[4]

                # -----------------------------------------------------------------
                # text classification
                # -----------------------------------------------------------------
                y_bow, y_tm = [], []
                with self.conn:
                    cursor = self.conn.cursor()
                    cursor.execute(SQL_TEXT_CLASS_SEL % (id_term_txt, id_ner_type))
                    rows = cursor.fetchall()

                    nr_results_txt = len(rows)
                    if nr_results_txt == 0:
                        self.logger.debug(":: term has not returned web sites!")
                    limit_txt = min(nr_results_txt, int(self.config.search_engine_tot_resources))

                    for itxt in range(limit_txt):

                        if rows[itxt][6] == 0 or rows[itxt][6] is None:  # not processed yet
                            merged_en = self.__detect_and_translate(rows[itxt][2], rows[itxt][3], rows[itxt][0], rows[itxt][4], rows[itxt][5])
                            if self.config.text_classification_type in (-1, 0):
                                ret_bow = self.text_bow.detect_text_klass(merged_en)
                            else: ret_bow = [0,0,0,0,0]
                            if self.config.text_classification_type in (-1, 1):
                                ret_tm = self.text_tm.detect_text_klass(merged_en)
                            else: ret_tm = [0,0,0,0,0]

                            y_bow.append(ret_bow)
                            y_tm.append(ret_tm)

                            cursor.execute(SQL_TEXT_CLASS_UPD % (ret_bow[0], ret_bow[1], ret_bow[2], ret_bow[3], ret_bow[4],
                                                           ret_tm[0], ret_tm[1], ret_tm[2], ret_tm[3], ret_tm[4],
                                                           rows[itxt][0]))
                        else:
                            y_bow.append(rows[itxt][7:12])
                            y_tm.append(rows[itxt][13:18])

                    self.conn.commit()

                    yyb = numpy.array(y_bow)
                    yytm = numpy.array(y_tm)
                    gpb = [numpy.count_nonzero(yyb == 1), numpy.count_nonzero(yyb == 2), numpy.count_nonzero(yyb == 3)]
                    horus_tx_ner = gpb.index(max(gpb)) + 1

                    horus_matrix[index][19] = limit_txt
                    horus_matrix[index][20] = gpb[0]
                    horus_matrix[index][21] = gpb[1]
                    horus_matrix[index][22] = gpb[2]
                    horus_matrix[index][23] = 0
                    horus_matrix[index][28] = yytm[0]
                    horus_matrix[index][29] = yytm[1]
                    horus_matrix[index][30] = yytm[2]

                    maxs_tx = heapq.nlargest(2, gpb)
                    maxs_tm = heapq.nlargest(2, yytm)
                    dist_tx_indicator = max(maxs_tx) - min(maxs_tx)
                    dist_tx_indicator_tm = max(maxs_tm) - min(maxs_tm)
                    horus_matrix[index][31] = dist_tx_indicator_tm
                    horus_matrix[index][24] = dist_tx_indicator
                    horus_matrix[index][25] = nr_results_txt

                    self.logger.debug(':: TX statistics:'
                                       '(LOC=%s, ORG=%s, PER=%s, DIST=%s, LOC_TM=%s, ORG_TM=%s, PER_TM=%s)' %
                                       (str(gpb[0]).zfill(2), str(gpb[1]).zfill(2), str(gpb[2]).zfill(2),
                                        str(dist_tx_indicator).zfill(2),
                                        str(yytm[0]).zfill(2), str(yytm[1]).zfill(2), str(yytm[2]).zfill(2)))
                    self.logger.debug('-------------------------------------------------------------')

                    if limit_txt != 0:
                        horus_matrix[index][26] = definitions.KLASSES[horus_tx_ner]
                    else:
                        horus_matrix[index][26] = definitions.KLASSES[4]

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
            if file is None:
                raise Exception("Provide an input file format to be annotated")
            self.logger.info(':: processing CoNLL format -> %s' % label)
            file = self.config.dataset_path + file
            sent_tokenize_list = self.util.process_ds_conll_format(file, label, token_index, ner_index, '')
            self.__get_horus_matrix_and_basic_statistics(sent_tokenize_list)
            if len(self.horus_matrix) > 0:
                self.util.download_and_cache_results(self.horus_matrix)
                self.detect_objects(self.horus_matrix)
                outfilename = self.util.path_leaf(file) + ".horus"
                self.__export_data(outfilename, out_subfolder, 'tsv')
                self.logger.info(':: feature extraction completed! filename = ' + outfilename)
            else:
                self.logger.warn(':: nothing to do...')

            return self.horus_matrix

        except Exception as error:
            self.logger.error('caught this error here: ' + repr(error))


if __name__ == "__main__":
    if len(sys.argv) not in (1,2,3,4):
        print "please inform: 1: data set and 2: column indexes ([1, .., n])"
    else:
        #args[0], args[1], args[2], args[3]
        FeatureExtraction().extract_features()