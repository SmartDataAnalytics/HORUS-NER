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
import sqlite3
from time import gmtime, strftime

import cv2
import langdetect
import nltk
import numpy
import requests
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from microsofttranslator import Translator
from nltk.tokenize import sent_tokenize
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer

import definitions
from bingAPI1 import bing_api2
from config import HorusConfig
from horus.components.systemlog import SystemLog
from horus.postagger import CMUTweetTagger

print cv2.__version__


class Core(object):
    """ Description:
            A core module for config.
        Attributes:
            name: A string representing the customer's name.
            balance: A float tracking the current balance of the customer's account.
    """

    # static methods
    version = "0.1"
    version_label = "HORUS 0.1 alpha"

    def __init__(self,force_download,trees):
        """Return a HORUS object"""
        self.sys = SystemLog("horus.log", logging.INFO, logging.INFO)
        self.config = HorusConfig()

        self.sys.log.info('------------------------------------------------------------------')
        self.sys.log.info('::                            HORUS                             ::')
        self.sys.log.info('------------------------------------------------------------------')
        self.sys.log.info(':: loading components...')

        self.english_vocab = None
        self.translator = Translator(self.config.translation_id, self.config.translation_secret)
        self.tfidf_transformer = TfidfTransformer()
        self.detect = cv2.xfeatures2d.SIFT_create()
        self.extract = cv2.xfeatures2d.SIFT_create()
        self.flann_params = dict(algorithm=1, trees=trees)
        self.flann = cv2.FlannBasedMatcher(self.flann_params, {})
        self.extract_bow = cv2.BOWImgDescriptorExtractor(self.extract, self.flann)
        self.svm_logo = joblib.load(self.config.models_cv_org)
        self.voc_org = joblib.load(self.config.models_cv_org_dict)
        self.svm_loc1 = joblib.load(self.config.models_cv_loc1)
        self.svm_loc2 = joblib.load(self.config.models_cv_loc2)
        self.svm_loc3 = joblib.load(self.config.models_cv_loc3)
        self.svm_loc4 = joblib.load(self.config.models_cv_loc4)
        self.svm_loc5 = joblib.load(self.config.models_cv_loc5)
        self.svm_loc6 = joblib.load(self.config.models_cv_loc6)
        self.svm_loc7 = joblib.load(self.config.models_cv_loc7)
        self.svm_loc8 = joblib.load(self.config.models_cv_loc8)
        self.svm_loc9 = joblib.load(self.config.models_cv_loc9)
        self.svm_loc10 = joblib.load(self.config.models_cv_loc10)
        self.voc_loc_1 = joblib.load(self.config.models_cv_loc_1_dict)
        self.voc_loc_2 = joblib.load(self.config.models_cv_loc_2_dict)
        self.voc_loc_3 = joblib.load(self.config.models_cv_loc_3_dict)
        self.voc_loc_4 = joblib.load(self.config.models_cv_loc_4_dict)
        self.voc_loc_5 = joblib.load(self.config.models_cv_loc_5_dict)
        self.voc_loc_6 = joblib.load(self.config.models_cv_loc_6_dict)
        self.voc_loc_7 = joblib.load(self.config.models_cv_loc_7_dict)
        self.voc_loc_8 = joblib.load(self.config.models_cv_loc_8_dict)
        self.voc_loc_9 = joblib.load(self.config.models_cv_loc_9_dict)
        self.voc_loc_10 = joblib.load(self.config.models_cv_loc_10_dict)
        self.text_checking_model = joblib.load(self.config.models_text)
        self.conn = sqlite3.connect(self.config.database_db)
        self.horus_matrix = []
        if force_download is True:
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

    def get_cv_annotation(self):
        x = numpy.array(self.horus_matrix)
        return x[:, [3, 4, 12, 13, 14, 15, 16, 17]]

    def annotate(self, input_text, input_file, ds_format, output_file, output_format):
        # 0 = text (parameter of reading file) / 1 = ritter
        if int(ds_format) == 0:
            text = ''
            if input_text is not None:
                text = input_text.strip('"\'')
                self.sys.log.info(':: processing text')
            elif input_file is not None:
                f = open(input_file, 'r')
                text = f.readlines()
                self.sys.log.info(':: processing input file')
            else:
                raise Exception("err: missing text to be annotated")
            sent_tokenize_list = self.process_input_text(text)

        elif int(ds_format) == 1:  # ritter
            if input_file is None:
                raise Exception("Provide an input file (ritter format) to be annotated")
            else:
                self.sys.log.info(':: loading Ritter ds')
                sent_tokenize_list = self.process_ritter_ds(input_file)

        self.sys.log.info(':: caching %s sentence(s)' % str(len(sent_tokenize_list)))
        # hasEntityNER (1=yes,0=dunno,-1=no), sentence, words[], tags_NER[], tags_POS[], tags_POS_UNI[]
        self.cache_sentence(int(ds_format), sent_tokenize_list)
        self.sys.log.info(':: done!')

        self.sys.log.info(':: caching results...')
        self.cache_results()
        self.sys.log.info(':: done!')

        #  updating components matrix
        # 0 = is_entity?,    1 = index_sent,   2 = index_word, 3 = word/term,
        # 4 = pos_universal, 5 = pos,          6 = ner       , 7 = compound? ,
        # 8 = compound_size, 9 = id_term_txt, 10 = id_term_img
        self.sys.log.info(':: detecting %s objects...' % len(self.horus_matrix))
        self.detect_objects()
        self.sys.log.info(':: done!')

        self.conn.close()

        self.sys.log.info(':: applying rules...')
        self.update_rules_cv_predictions()
        self.sys.log.info(':: updating compounds...')
        self.update_compound_predictions()
        self.sys.log.info(':: done!')

        # self.sys.log.info(horus_matrix)
        header = ["IS_ENTITY?", "ID_SENT", "ID_WORD", "WORD/TERM", "POS_UNI", "POS", "NER", "COMPOUND", "COMPOUND_SIZE",
                  "ID_TERM_TXT", "ID_TERM_IMG",
                  "TOT_IMG", "TOT_CV_LOC", "TOT_CV_ORG", "TOT_CV_PER", "DIST_CV_I", "PL_CV_I", "CV_KLASS", "TOT_RESULTS_TX",
                  "TOT_TX_LOC", "TOT_TX_ORG",
                  "TOT_TX_PER", "TOT_ERR_TRANS", "DIST_TX_I", "TX_KLASS", "HORUS_KLASS"]

        if int(ds_format) == 0:
            self.print_annotated_sentence()

        self.sys.log.info(':: exporting metadata...')
        if output_file == '':
            output_file = 'noname'
        if output_format == 'json':
            with open(output_file + '.json', 'wb') as outfile:
                json.dump(self.horus_matrix, outfile)
        elif output_format == 'csv':
            horus_csv = open(output_file + '.csv', 'wb')
            wr = csv.writer(horus_csv, quoting=csv.QUOTE_ALL)
            wr.writerow(header)
            wr.writerows(self.horus_matrix)

        return self.horus_matrix

    def print_annotated_sentence(self):
        '''
        read the components matrix and prints the annotated sentences
        :param horus_matrix:
        :return: output of annotated sentence
        '''
        x = ''
        id_sent_aux = self.horus_matrix[0][1]
        for term in self.horus_matrix:
            if term[7] == 0:
                if id_sent_aux != term[1]:
                    #self.sys.log.info(':: sentence ' + str(id_sent_aux) + ': ' + x)
                    self.sys.log.info(':: sentence: ' + x)
                    id_sent_aux = term[1]
                    x = ' ' + str(term[3]) + '/' + str(term[25])
                else:
                    x += ' ' + str(term[3]) + '/' + str(term[4]) + '/' + str(term[5]) + '/' + str(term[25])
                    # x += ' ' + str(term[3]) + '/' + str(term[25])

        self.sys.log.info(':: sentence: ' + x)

    def cache_sentence_ritter(self,sentence_list):
        self.sys.log.debug(':: caching Ritter dataset...:')
        i_sent, i_word = 1, 1
        compound, prev_tag = '', ''
        sent_with_ner = 0
        token_ok = 0
        compound_ok = 0
        for sent in sentence_list:

            self.sys.log.info(':: processing sentence: ' + sent[1])

            # processing compounds
            if sent[0] == 1:
                sent_with_ner += 1
                for tag in sent[3]:  # list of NER tags
                    word = sent[2][i_word - 1]
                    if tag in definitions.NER_RITTER:  # only desired tags
                        if prev_tag.replace('B-', '').replace('I-', '') == tag.replace('B-', '').replace('I-', ''):
                            compound += prev_word + ' ' + word + ' '
                    prev_word = word
                    prev_tag = tag
                    i_word += 1
                compound = compound[:-1]

                if compound != '':
                    compound_ok+=1
                    self.horus_matrix.append([1, i_sent, i_word - len(compound.split(' ')), compound, '', '', '', 1, len(compound.split(' '))])
                    compound = ''
                prev_tag = ''
                prev_word = ''

            # processing tokens

            #  transforming to components matrix
            # 0 = is_entity?,    1 = index_sent, 2 = index_word, 3 = word/term,
            # 4 = pos_universal, 5 = pos,        6 = ner       , 7 = compound? ,
            # 8 = compound_size

            i_word = 1
            for k in range(len(sent[2])): # list of NER tags
                is_entity = 1 if sent[3] in definitions.NER_RITTER else 0

                self.horus_matrix.append([is_entity, i_sent, i_word, sent[2][k], sent[5][k], sent[4][k], sent[3][k], 0, 0])
                i_word += 1
                if is_entity:
                    token_ok += 1

            self.db_save_sentence(sent[1], '-', '-', str(sent[3]))
            i_sent += 1
            i_word = 1

        self.sys.log.debug(':: done! total of sentences = %s, tokens = %s and compounds = %s'
                     % (str(sent_with_ner), str(token_ok), str(compound_ok)))

    def tokenize_and_pos(self, sentence):
        if self.config.models_pos_tag_lib == 1:
            tokens = sentence
            if type(sentence) is not list:
                tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            return tokens, tagged, nltk.pos_tag(tokens, tagset="universal")
        elif self.config.models_pos_tag_lib == 2:
            if type(sentence) is not list:
                return self.tokenize_and_pos_twitter([sentence])
            return self.tokenize_and_pos_twitter_list(sentence)

    def cache_sentence(self,sentence_format, sentence_list):
        if sentence_format == 0:
            self.cache_sentence_free_text(sentence_list)
        elif sentence_format == 1:
            self.cache_sentence_ritter(sentence_list)

    def cache_sentence_free_text(self,sentence_list):

        i_sent, i_word = 1, 1
        self.sys.log.debug(':: chunking pattern ...')
        pattern = """
                NP:
                   {<JJ>*<NN|NNS|NNP|NNPS><CC>*<NN|NNS|NNP|NNPS>+}
                   {<NN|NNS|NNP|NNPS><IN>*<NN|NNS|NNP|NNPS>+}
                   {<JJ>*<NN|NNS|NNP|NNPS>+}
                   {<NN|NNP|NNS|NNPS>+}
                """
        cp = nltk.RegexpParser(pattern)
        compounds = '|'

        for sent in sentence_list:
            #tokens, tagged = tokenize_and_pos(sent[1])
            #self.sys.log.info(':: tags: ' + str(tagged))
            ## entities = nltk.chunk.ne_chunk(tagged)

            #  add compounds of given sentence

            aux = 0
            toparse = []
            for obj in sent[2]:
                toparse.append(tuple([obj, sent[4][aux]]))
                aux+=1
            t = cp.parse(toparse)
            i_word = 1
            for item in t:
                is_entity = 1 if (sent[0] == 1 and sent[3][i_word - 1] != 'O') else -1
                if type(item) is nltk.Tree:  # that's a compound
                    compound = ''
                    for tk in item:
                        compound += tk[0] + ' '
                        i_word += 1
                    if len(item) > 1:
                        self.horus_matrix.append([is_entity, i_sent, i_word - len(item), compound[:len(compound) - 1], '', '', '', 1, len(item)])
                        compounds += compound[:len(compound) - 1] + '|'
                    compound = ''

            #  transforming to components matrix
            # 0 = is_entity?,    1 = index_sent, 2 = index_word, 3 = word/term,
            # 4 = pos_universal, 5 = pos,        6 = ner       , 7 = compound? ,
            # 8 = compound_size
            i_word = 1
            for k in range(len(sent[2])):
                is_entity = 1 if (sent[0] == 1 and sent[3][k] != 'O') else -1
                self.horus_matrix.append([is_entity, i_sent, i_word, sent[2][k], sent[5][k], sent[4][k], sent[3][k], 0, 0])
                i_word += 1

            self.db_save_sentence(sent[1], ' '.join(sent[5]), compounds, ' '.join(sent[2]))

            i_sent += 1

    def db_save_sentence(self,sent, tagged, compound, tokens):
        c = self.conn.cursor()
        self.conn.text_factory = str
        sql = """SELECT id FROM HORUS_SENTENCES WHERE sentence = ? """
        c.execute(sql, (sent,))
        res_sent = c.fetchone()
        if res_sent is None:
            self.sys.log.info(':: caching sentence: ' + sent)
            #buffer(zlib.compress
            row = (str(sent), str(tagged), str(compound), str(tokens))
            sql = """INSERT INTO HORUS_SENTENCES(sentence, tagged, compounds, tokens)
                             VALUES(?,?,?,?)"""
            self.sys.log.debug(sql)
            c.execute(sql, row)
            self.conn.commit()
            self.sys.log.debug(':: done: ' + sent)
        else:
            self.sys.log.debug(':: sentence is already cached')

    def download_image_local(self,image_url, image_type, thumbs_url, thumbs_type, term_id, id_ner_type, seq):
        val = URLValidator()
        auxtype = None
        try:
            val(thumbs_url)
            try:
                img_data = requests.get(thumbs_url).content
                with open('%s%s_%s_%s.%s' % (self.config.cache_img_folder, term_id, id_ner_type, seq, thumbs_type.split('/')[1]), 'wb') as handler:
                    handler.write(img_data)
                    auxtype = thumbs_type.split('/')[1]
            except Exception as error:
                print('-> error: ' + repr(error))
        except ValidationError, e:
            self.sys.log.error('No thumbs img here...', e)
            try:
                img_data = requests.get(image_url).content
                with open('%s%s_%s_%s.%s' % (self.config.cache_img_folder, term_id, id_ner_type, seq, image_type.split('/')[1]), 'wb') as handler:
                    auxtype = image_type.split('/')[1]
                    handler.write(img_data)
            except Exception as error:
                print('-> error: ' + repr(error))
        return auxtype

    def cache_results(self):
        try:
            auxc = 1
            self.sys.log.info(':: caching horus_matrix: ' + str(len(self.horus_matrix)))

            for item in self.horus_matrix:
                self.sys.log.info(':: item %s - %s ' % (str(auxc), str(len(self.horus_matrix))))
                term = item[3]
                if item[4] == 'NOUN' or item[7] == 1:
                    self.sys.log.debug(':: caching [%s] ...' % term)
                    c = self.conn.cursor()

                    # checking if we have searched that before (might be the case of had used
                    # different configurations, or different search engine, for instance.
                    sql = """SELECT id FROM HORUS_TERM WHERE term = ?"""
                    c.execute(sql, (term,))
                    res_term = c.fetchone()
                    if res_term is None:
                        self.sys.log.info(':: [%s] has not been cached before!' % term)
                        cterm = self.conn.execute("""INSERT INTO HORUS_TERM(term) VALUES(?)""", (term,))
                    else:
                        cterm = res_term[0]

                    # check if term (text) has been cached before
                    # in case components is extended to accept more than 1 search engine, this table should also
                    # have it defined
                    values = (term, self.config.search_engine_api, 1, self.config.search_engine_features_text)
                    sql = """SELECT id
                             FROM HORUS_TERM_SEARCH
                             WHERE term = ? AND
                                   id_search_engine = ? AND
                                   id_search_type = ? AND
                                   search_engine_features = ?"""
                    c.execute(sql, values)
                    res = c.fetchone()
                    if res is None:
                        self.sys.log.info(':: [%s] caching - text' % term)
                        values = (term, cterm.lastrowid, self.config.search_engine_api, 1,
                                  self.config.search_engine_features_text,
                                  str(strftime("%Y-%m-%d %H:%M:%S", gmtime())),
                                  self.config.search_engine_tot_resources)
                        c = self.conn.execute("""INSERT into HORUS_TERM_SEARCH(term, id_term, id_search_engine, id_search_type,
                                                                          search_engine_features, query_date,
                                                                          query_tot_resource)
                                                 VALUES(?, ?, ?, ?, ?, ?, ?)""", values)

                        id_term_search = c.lastrowid
                        item.extend([id_term_search])  # updating matrix
                        seq = 0
                        # get text
                        metaquery, result = bing_api2(term, api=self.config.search_engine_key, source_type="Web",
                                                     top=self.config.search_engine_tot_resources, format='json', market='en-US')
                        for web_result in result['d']['results']:
                            seq+=1
                            row = (id_term_search,
                                   0,
                                   web_result['ID'],
                                   seq,
                                   web_result['Url'],
                                   web_result['Title'],
                                   web_result['Description'],
                                   '')
                            c.execute("""INSERT INTO HORUS_SEARCH_RESULT_TEXT (id_term_search,
                                                                                    id_ner_type,
                                                                                    search_engine_resource_id,
                                                                                    result_seq,
                                                                                    result_url,
                                                                                    result_title,
                                                                                    result_description,
                                                                                    result_html_text)
                                              VALUES(?,?,?,?,?,?,?,?)""", row)

                            c.execute("""UPDATE HORUS_TERM_SEARCH
                                              SET metaquery = '%s'
                                              WHERE id = %s""" % (metaquery, id_term_search))

                        if seq == 0:
                            row = (id_term_search,
                                   0,
                                   '',
                                   seq,
                                   '',
                                   '',
                                   '',
                                   '')
                            c.execute("""INSERT INTO HORUS_SEARCH_RESULT_TEXT (id_term_search,
                                                                               id_ner_type,
                                                                               search_engine_resource_id,
                                                                               result_seq,
                                                                               result_url,
                                                                               result_title,
                                                                               result_description,
                                                                               result_html_text)
                                                                      VALUES(?,?,?,?,?,?,?,?)""", row)

                            c.execute("""UPDATE HORUS_TERM_SEARCH
                                         SET metaquery = '%s'
                                         WHERE id = %s""" % (metaquery, id_term_search))

                        self.sys.log.debug(':: term [%s] cached (text)!' % term)
                        self.conn.commit()
                    else:
                        item.extend(res)  # updating matrix
                        self.sys.log.debug(':: term %s is already cached (text)!' % term)

                    values = (term, self.config.search_engine_api, 2, self.config.search_engine_features_text)
                    c = self.conn.execute("""SELECT id
                                        FROM HORUS_TERM_SEARCH
                                        WHERE term = ? AND
                                              id_search_engine = ? AND
                                              id_search_type = ? AND
                                              search_engine_features = ?""", values)
                    res = c.fetchone()
                    if res is None:
                        self.sys.log.info(':: [%s] caching - image' % term)
                        values = (term, cterm.lastrowid if type(cterm) is not int else cterm, self.config.search_engine_api, 2,
                                   self.config.search_engine_features_img,
                                   str(strftime("%Y-%m-%d %H:%M:%S", gmtime())), self.config.search_engine_tot_resources)
                        sql = """INSERT into HORUS_TERM_SEARCH(term, id_term, id_search_engine, id_search_type,
                                                               search_engine_features, query_date, query_tot_resource)
                                                 VALUES(?,?,?,?,?,?,?)"""
                        c.execute(sql, values)
                        id_term_img = c.lastrowid
                        item.extend([id_term_img])  # updating matrix
                        seq = 0
                        # get images
                        metaquery, result = bing_api2(item[3], api=self.config.search_engine_key, source_type="Image",
                                                     top=self.config.search_engine_tot_resources, format='json')
                        for web_img_result in result['d']['results']:
                            self.sys.log.debug(':: downloading image [%s]' % (web_img_result['Title']))
                            seq += 1
                            auxtype = self.download_image_local(web_img_result['MediaUrl'],
                                                 web_img_result['ContentType'],
                                                 web_img_result['Thumbnail']['MediaUrl'],
                                                 web_img_result['Thumbnail']['ContentType'],
                                                 id_term_img,
                                                 0,
                                                 seq)
                            self.sys.log.debug(':: caching image result ...')
                            fname = ('%s_%s_%s.%s' % (str(id_term_img),  str(0),  str(seq),  str(auxtype)))
                            row = (id_term_img,
                                   0,
                                   web_img_result['ID'],
                                   seq,
                                   web_img_result['MediaUrl'],
                                   web_img_result['Title'],
                                   web_img_result['ContentType'],
                                   web_img_result['Height'],
                                   web_img_result['Width'],
                                   web_img_result['Thumbnail']['MediaUrl'],
                                   web_img_result['Thumbnail']['ContentType'],
                                   fname)

                            sql = """INSERT INTO HORUS_SEARCH_RESULT_IMG (id_term_search,
                                                                          id_ner_type,
                                                                          search_engine_resource_id,
                                                                          result_seq,
                                                                          result_media_url,
                                                                          result_media_title,
                                                                          result_media_content_type,
                                                                          result_media_height,
                                                                          result_media_width,
                                                                          result_media_thumb_media_url,
                                                                          result_media_thumb_media_content_type,
                                                                          filename)
                                              VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"""
                            c.execute(sql, row)

                            c.execute("""UPDATE HORUS_TERM_SEARCH
                                              SET metaquery = '%s'
                                              WHERE id = %s""" % (metaquery, id_term_img))

                        self.sys.log.debug(':: term [%s] cached (img)!' % term)
                        self.conn.commit()
                    else:
                        self.sys.log.debug(':: term %s is already cached (img)!' % term)
                        item.extend(res)  # updating matrix

                auxc+=1
        except Exception as e:
            self.sys.log.error(':: an error has occurred: ', e)
            raise

    def detect_faces(self,img):
        try:
            # print cv2.__version__
            face_cascade = cv2.CascadeClassifier(self.config.models_cv_per)
            image = cv2.imread(img)
            if image is None:
                self.sys.log.error('could not load the image: ' + img)
                return -1
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
            return len(faces)
            # cv2.CV_HAAR_SCALE_IMAGE #
            # minSize=(30, 30)

            ## Draw a rectangle around the faces
            # for (x, y, w, h) in faces:
            #    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow("Faces found", image)
            # cv2.waitKey(0)

        except Exception as error:
            self.sys.log.error(':: error: ' + repr(error))
            return -1

    def bow_features(self,fn, ner_type):
        im = cv2.imread(fn, 0)
        if ner_type == 'ORG_1':
            self.extract_bow.setVocabulary(self.voc_org)
        elif ner_type == 'LOC_1':
            self.extract_bow.setVocabulary(self.voc_loc_1)
        elif ner_type == 'LOC_2':
            self.extract_bow.setVocabulary(self.voc_loc_2)
        elif ner_type == 'LOC_3':
            self.extract_bow.setVocabulary(self.voc_loc_3)
        elif ner_type == 'LOC_4':
            self.extract_bow.setVocabulary(self.voc_loc_4)
        elif ner_type == 'LOC_5':
            self.extract_bow.setVocabulary(self.voc_loc_5)
        elif ner_type == 'LOC_6':
            self.extract_bow.setVocabulary(self.voc_loc_6)
        elif ner_type == 'LOC_7':
            self.extract_bow.setVocabulary(self.voc_loc_7)
        elif ner_type == 'LOC_8':
            self.extract_bow.setVocabulary(self.voc_loc_8)
        elif ner_type == 'LOC_9':
            self.extract_bow.setVocabulary(self.voc_loc_9)
        elif ner_type == 'LOC_10':
            self.extract_bow.setVocabulary(self.voc_loc_10)

        return self.extract_bow.compute(im, self.detect.detect(im))

    def detect_logo(self,img):
        f = self.bow_features(img, 'ORG_1');
        if f is None:
            self.sys.log.warn(':: feature extraction error!')
            p = [0]
        else:
            p = self.svm_logo.predict(f)
        self.sys.log.debug('predicted class -> ' + str(p))
        return p

    def detect_place(self,img):
        self.sys.log.debug(':: detecting places...')
        ret = []
        f = self.bow_features(img, 'LOC_1');
        if f is None:
            self.sys.log.warn(':: feature extraction error!')
            ret.append(-1)
        else:
            ret.append(self.svm_loc1.predict(f)[0])

        f = self.bow_features(img, 'LOC_2');
        if f is None:
            self.sys.log.warn(':: feature extraction error!')
            ret.append(-1)
        else:
            ret.append(self.svm_loc2.predict(f)[0])

        f = self.bow_features(img, 'LOC_3');
        if f is None:
            self.sys.log.warn(':: feature extraction error!')
            ret.append(-1)
        else:
            ret.append(self.svm_loc3.predict(f)[0])

        f = self.bow_features(img, 'LOC_4');
        if f is None:
            self.sys.log.warn(':: feature extraction error!')
            ret.append(-1)
        else:
            ret.append(self.svm_loc4.predict(f)[0])

        f = self.bow_features(img, 'LOC_5');
        if f is None:
            self.sys.log.warn(':: feature extraction error!')
            ret.append(-1)
        else:
            ret.append(self.svm_loc5.predict(f)[0])

        f = self.bow_features(img, 'LOC_6');
        if f is None:
            self.sys.log.warn(':: feature extraction error!')
            ret.append(-1)
        else:
            ret.append(self.svm_loc6.predict(f)[0])

        f = self.bow_features(img, 'LOC_7');
        if f is None:
            self.sys.log.warn(':: feature extraction error!')
            ret.append(-1)
        else:
            ret.append(self.svm_loc7.predict(f)[0])

        f = self.bow_features(img, 'LOC_8');
        if f is None:
            self.sys.log.warn(':: feature extraction error!')
            ret.append(-1)
        else:
            ret.append(self.svm_loc8.predict(f)[0])

        f = self.bow_features(img, 'LOC_9');
        if f is None:
            self.sys.log.warn(':: feature extraction error!')
            ret.append(-1)
        else:
            ret.append(self.svm_loc9.predict(f)[0])

        f = self.bow_features(img, 'LOC_10');
        if f is None:
            self.sys.log.warn(':: feature extraction error!')
            ret.append(-1)
        else:
            ret.append(self.svm_loc10.predict(f)[0])

        return ret

    def detect_text_klass(self,t1, t2, id, t1en, t2en):

        self.sys.log.debug(':: text analysis component launched')

        try:
            from translate import Translator

            #https://pypi.python.org/pypi/translate (alternative 1000 per day)
            #https://www.microsoft.com/en-us/translator/getstarted.aspx
            #https://github.com/openlabs/Microsoft-Translator-Python-API

            c = self.conn.cursor()
            t1final = t1
            t2final = t2

            # need to save to components db
            if t1en is None:
                lt1 = langdetect.detect(t1)
                if lt1 != 'en':
                    try:
                        t1final = self.translator.translate(t1, 'en')
                    except Exception as e1:
                        self.sys.log.error(':: Error, trying another service: ' + str(e1))
                        try:
                            translator2 = Translator(from_lang=lt1, to_lang="en")
                            t1final = translator2.translate(t1)
                        except Exception as e2:
                            self.sys.log.error(':: Error at service 2: ' + str(e2))
                            return [-1]
                            # updating

                t1final = 'u'+t1final # .encode('ascii', 'ignore')
                sql = """UPDATE HORUS_SEARCH_RESULT_TEXT
                         SET result_title_en = ? WHERE id = ?"""
                c.execute(sql, (t1final, id))
                self.conn.commit()
            else:
                t1final = t1en

            if t2en is None:
                lt2 = langdetect.detect(t2)
                if lt2 != 'en':
                    try:
                        t2final = self.translator.translate(t2, 'en')
                    except Exception as e1:
                        self.sys.log.error(':: Error, trying another service: ' + str(e1))
                        try:
                            translator2 = Translator(from_lang=lt2, to_lang="en")
                            t2final = translator2.translate(t2)
                        except Exception as e2:
                            self.sys.log.error(':: Error at service 2: ' + str(e2))
                            return [-1]
                            # updating

                #t2final = t2final.encode('ascii', 'ignore')
                t2final = 'u'+t2final
                sql = """UPDATE HORUS_SEARCH_RESULT_TEXT
                                SET result_description_en = ? WHERE id = ?"""
                c.execute(sql, (t2final, id))
                self.conn.commit()
            else:
                t2final = t2en

            c.close()
            docs = ["{} {}".format(t1final, t2final)]
            predicted = self.text_checking_model.predict(docs)
            return predicted

            #blob = TextBlob(t2)
            #t22 = blob.translate(to='en')
            # text_vocab = set(w.lower() for w in t2 if w.lower().isalpha())
            # unusual = text_vocab.difference(english_vocab)

        except Exception as e:
            self.sys.log.error(':: Error: ' + str(e))
            return [-1]

    def detect_objects(self):     # id_term_img, id_term_txt, id_ner_type, term
        auxi = 0
        toti = len(self.horus_matrix)
        for item in self.horus_matrix:
            auxi += 1
            if item[4] == 'NOUN' or item[7] == 1:

                self.sys.log.info(':: processing item %d of %d' % (auxi, toti))

                id_term_img = item[10]
                id_term_txt = item[9]
                id_ner_type = 0
                term = item[3]

                tot_geral_faces = 0
                tot_geral_logos = 0
                tot_geral_locations = 0
                tot_geral_pos_locations = 0
                tot_geral_neg_locations = 0
                T = int(self.config.models_location_theta)  # location threshold

                filesimg = []
                metadata = []
                with self.conn:
                    cursor = self.conn.cursor()
                    cursor.execute("""SELECT filename,
                                                 id,
                                                 processed,
                                                 nr_faces,
                                                 nr_logos,
                                                 nr_place_1,
                                                 nr_place_2,
                                                 nr_place_3,
                                                 nr_place_4,
                                                 nr_place_5,
                                                 nr_place_6,
                                                 nr_place_7,
                                                 nr_place_8,
                                                 nr_place_9,
                                                 nr_place_10
                                          FROM HORUS_SEARCH_RESULT_IMG
                                          WHERE id_term_search = %s AND id_ner_type = %s""" % (id_term_img, id_ner_type))
                    rows = cursor.fetchall()
                    tot_img = len(rows)

                    for row in rows:  # 0 = file path | 1 = id | 2 = processed | 3=nr_faces | 4=nr_logos | 5 to 13=nr_places_1 to 9
                        filesimg.append((self.config.cache_img_folder + row[0],
                                         row[1],
                                         row[2],
                                         row[3],
                                         row[4],
                                         row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13],
                                         row[14]))

                for image_term in filesimg:
                    if image_term[2] == 1:
                        tot_geral_faces += image_term[3]
                        tot_geral_logos += image_term[4]
                        if (image_term[5:13]).count(1) >= int(T):
                            tot_geral_locations += 1
                        tot_geral_pos_locations += image_term[5:13].count(1)
                        tot_geral_neg_locations += (image_term[5:13].count(-1) * -1)
                    else:
                        # ----- face recognition -----
                        tot_faces = self.detect_faces(image_term[0])
                        if tot_faces > 0:
                            tot_geral_faces += 1
                            self.sys.log.debug(":: found {0} faces!".format(tot_faces))

                        # ----- logo recognition -----
                        tot_logos = self.detect_logo(image_term[0])
                        if tot_logos[0] == 1:
                            tot_geral_logos += 1
                            self.sys.log.debug(":: found {0} logo(s)!".format(1))

                        # ----- place recognition -----
                        res = self.detect_place(image_term[0])
                        tot_geral_pos_locations += res.count(1)
                        tot_geral_neg_locations += (res.count(-1) * -1)

                        if res.count(1) >= T:
                            tot_geral_locations += 1
                            self.sys.log.debug(":: found {0} place(s)!".format(1))

                        # updating results
                        sql = """UPDATE HORUS_SEARCH_RESULT_IMG
                                     SET nr_faces = ?, nr_logos = ?, nr_place_1 = ?, nr_place_2 = ?, nr_place_3 = ?,
                                         nr_place_4 = ?, nr_place_5 = ?, nr_place_6 = ?, nr_place_7 = ?, nr_place_8 = ?,
                                         nr_place_9 = ?, nr_place_10 = ?, processed = 1
                                     WHERE id = ?"""
                        param = []
                        param.append(tot_faces)
                        param.append(tot_logos[0]) if tot_logos[0] == 1 else param.append(0)
                        param.extend(res)
                        param.append(image_term[1])
                        cursor.execute(sql, param)

                self.conn.commit()

                outs = [tot_geral_locations, tot_geral_logos, tot_geral_faces]
                maxs_cv = heapq.nlargest(2, outs)
                dist_cv_indicator = max(maxs_cv) - min(maxs_cv)
                place_cv_indicator = tot_geral_pos_locations + tot_geral_neg_locations

                # 0 to 5
                metadata.append(tot_img)
                metadata.append(tot_geral_locations)  # 1
                metadata.append(tot_geral_logos)  # 2
                metadata.append(tot_geral_faces)  # 3
                metadata.append(dist_cv_indicator)  # 4
                metadata.append(place_cv_indicator)  # 5

                self.sys.log.debug('-------------------------------------------------------------')
                self.sys.log.debug(':: [checking related visual information for [%s]]' % term)
                self.sys.log.debug('')
                self.sys.log.debug('-> CV_LOC  indicator: %f %%' % (float(tot_geral_locations) / tot_img)) if tot_img > 0 \
                    else self.sys.log.debug('-> CV_LOC  indicator: err no img retrieved')
                self.sys.log.debug('-> CV_ORG  indicator: %f %%' % (float(tot_geral_logos) / tot_img)) if tot_img > 0 \
                    else self.sys.log.debug('-> CV_ORG  indicator: err no img retrieved')
                self.sys.log.debug('-> CV_PER  indicator: %f %%' % (float(tot_geral_faces) / tot_img)) if tot_img > 0 \
                    else self.sys.log.debug('-> CV_PER  indicator: err no img retrieved')
                self.sys.log.debug('-> CV_DIST indicator: %s' % (str(dist_cv_indicator)))
                self.sys.log.debug('-> CV_PLC  indicator: %s' % (str(place_cv_indicator)))

                metadata.append(definitions.KLASSES[outs.index(max(outs)) + 1])

                # text classification
                self.sys.log.debug(':: [checking related textual information ...]')
                y = []
                with self.conn:
                    cursor = self.conn.cursor()
                    cursor.execute("""SELECT id, result_seq, result_title, result_description, result_title_en,
                                                 result_description_en, processed, text_klass
                                          FROM HORUS_SEARCH_RESULT_TEXT
                                          WHERE id_term_search = %s AND id_ner_type = %s""" % (id_term_txt, id_ner_type))
                    rows = cursor.fetchall()
                    tot_err = 0
                    for row in rows:
                        if row[6] == 0 or row[6] is None:
                            ret = self.detect_text_klass(row[2], row[3], row[0], row[4], row[5])
                            if ret[0] != -1:
                                y.append(ret)
                                sql = """UPDATE HORUS_SEARCH_RESULT_TEXT SET text_klass = %s , processed = 1
                                         WHERE id = %s""" % (ret[0], row[0])
                                self.sys.log.debug(':: ' + sql)
                                cursor.execute(sql)
                            else:
                                tot_err += 1
                        else:
                            y.append(row[7])

                    self.conn.commit()

                    gp = [y.count(1), y.count(2), y.count(3)]
                    horus_tx_ner = gp.index(max(gp)) + 1

                    self.sys.log.debug(':: final textual checking statistics for term [%s] '
                                    '(1-LOC = %s, 2-ORG = %s and 3-PER = %s)' % (term, str(y.count(1)), str(y.count(2)),
                                                                                 str(y.count(3))))
                    # 7 to 11
                    metadata.append(len(rows))
                    metadata.append(y.count(1))
                    metadata.append(y.count(2))
                    metadata.append(y.count(3))
                    metadata.append(float(tot_err))

                    maxs_tx = heapq.nlargest(2, gp)
                    dist_tx_indicator = max(maxs_tx) - min(maxs_tx)

                    # 12, 13
                    metadata.append(dist_tx_indicator)
                    metadata.append(definitions.KLASSES[horus_tx_ner])

                    if len(rows) != 0:
                        self.sys.log.debug('-> TX_LOC  indicator: %f %%' % (float(y.count(1)) / len(rows)))
                        self.sys.log.debug('-> TX_ORG  indicator: %f %%' % (float(y.count(2)) / len(rows)))
                        self.sys.log.debug('-> TX_PER  indicator: %f %%' % (float(y.count(3)) / len(rows)))
                        self.sys.log.debug('-> TX_DIST indicator: %s' % (str(dist_tx_indicator)))
                        self.sys.log.debug(':: number of trans. errors -> ' + str(tot_err) + ' over ' + str(len(rows)))
                        self.sys.log.debug(':: most likely class -> ' + definitions.KLASSES[horus_tx_ner])
                    else:
                        self.sys.log.debug(':: there was a problem searching this term..please try to index it again...')

                    # checking final NER - 14
                    if metadata[4] >= int(self.config.models_distance_theta):
                        metadata.append(metadata[6])  # CV is the final decision
                    else:
                        metadata.append(metadata[13])  # TX is the final decision

                item.extend(metadata)
            else:
                item.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def update_rules_cv_predictions(self):
        '''
        updates the predictions based on inner rules
        :return:
        '''
        self.sys.log.info(':: updating predictions based on rules')
        for i in range(len(self.horus_matrix)):
            initial = self.horus_matrix[i][17]
            # get nouns or compounds
            if self.horus_matrix[i][4] == u'NOUN' or self.horus_matrix[i][7] == 1:
                # do not consider classifications below a theta
                if self.horus_matrix[i][15] < int(self.config.models_distance_theta):
                    self.horus_matrix[i][17] = "*"
                # ignore LOC classes having iPLC negative
                if bool(self.config.models_distance_theta_high_bias) is True:
                    if initial == "LOC":
                        if self.horus_matrix[i][16] < int(self.config.models_limit_min_loc):
                            self.horus_matrix[i][17] = "*"
                        elif self.horus_matrix[i][16] < 0 and self.horus_matrix[i][15] > \
                                int(self.config.models_safe_interval):
                            self.horus_matrix[i][17] = initial



    def update_compound_predictions(self):
        '''
        updates the predictions based on the compound
        :return:
        '''
        self.sys.log.info(':: updating compounds predictions')
        i_sent = None
        i_first_word = None
        for i in range(len(self.horus_matrix)):
            if self.horus_matrix[i][7] == 1:
                y = self.horus_matrix[i][25]
                i_sent = self.horus_matrix[i][1]
                i_first_word = self.horus_matrix[i][2]
                c_size = int(self.horus_matrix[i][8])
            if self.horus_matrix[i][7] == 0 and i_sent is not None:
                if self.horus_matrix[i][1] == i_sent and self.horus_matrix[i][2] == i_first_word:
                    for k in range(c_size):
                        self.horus_matrix[i + k][25] = y
        self.sys.log.info(':: done')

    def process_input_text(self, text):
        self.sys.log.info(':: text: ' + text)
        self.sys.log.info(':: tokenizing sentences ...')
        sent_tokenize_list = sent_tokenize(text)
        self.sys.log.info(':: processing ' + str(len(sent_tokenize_list)) + ' sentence(s).')
        sentences = []
        w = []
        ner = []
        pos = []
        hasNER = -1
        for sentence in sent_tokenize_list:
            tokens, pos_taggers, pos_universal = self.tokenize_and_pos(sentence)
            chunked = nltk.ne_chunk(pos_taggers)
            for ch in chunked:
                if type(ch) is nltk.Tree:
                    w.append(ch[0][0])
                    pos.append(ch[0][1])
                    hasNER = 1
                    if ch._label == 'GPE' or ch._label == 'LOCATION':
                        ner.append('LOC')
                    elif ch._label == 'PERSON':
                        ner.append('PER')
                    elif ch._label == 'ORGANIZATION':
                        ner.append('ORG')
                    else:
                        ner.append('O')
                else:
                    w.append(ch[0])
                    pos.append(ch[1])
                    ner.append('O')

            _pos = list(zip(*pos_universal)[1])
            sentences.append([hasNER, sentence, tokens, ner, pos, _pos])
            pos = []
            ner = []
            w = []
        return sentences


    def process_ritter_ds(self,dspath):
        '''
        return a set of sentences
        :param dspath: path to Ritter dataset
        :return: sentence contains any entity?, sentence, words, NER tags
        '''
        sentences = []
        w = []
        t = []
        s = ''
        has3NER = -1
        with open(dspath) as f:
            for line in f:
                token = line.split('\t')[0]
                tag = line.split('\t')[1].replace('\r','').replace('\n','')
                if token == '':
                    if len(w) != 0:
                        pos_uni = nltk.pos_tag(w, tagset='universal')
                        pos = nltk.pos_tag(w)
                        _pos = zip(*pos)[1]
                        _pos_uni = zip(*pos_uni)[1]
                        sentences.append([has3NER, s, w, t, list(_pos), list(_pos_uni)])
                        w = []
                        t = []
                        s = ''
                        has3NER = -1
                else:
                    s += token + ' '
                    w.append(token)
                    t.append(tag)
                    if tag in definitions.NER_RITTER:
                        has3NER = 1

        #just in case of tweetNLP
        if self.config.models_pos_tag_lib == 2:
            #Communicating once the shell process is opened rather than closing comms it's definitely more sensible, so I have done it...
            list_shell = []
            for item in sentences:
                list_shell.append(item[1])
            self.sys.log.info(':: opening shell once to tweetNLP pos tagger ...')
            tokens, pos, pos_uni = self.tokenize_and_pos(list_shell)
            cache_sentences = []
            index = 0
            for item in sentences:
                _pos = zip(*pos[index])[1]
                _pos_uni = zip(*pos_uni[index])[1]
                cache_sentences.append([item[0], item[1], item[2], item[3], list(_pos), list(_pos_uni)])
                index +=1
            return cache_sentences
        else:
            return sentences

    def tokenize_and_pos_twitter(self, text):
        tokens = []
        tagged = []
        pos_universal = []
        pos_token_tag_sentence = CMUTweetTagger.runtagger_parse(text)

        for sequence_tag in pos_token_tag_sentence:
            for token_tag in sequence_tag:
                tokens.append(token_tag[0])
                tagged.append(self.convert_cmu_tags(token_tag[1]))
                pos_universal.append(self.convert_cmu_tags(token_tag[1], tagset="universal"))
        return tokens, zip(tokens, tagged), zip(tokens, pos_universal)

    def tokenize_and_pos_twitter_list(self, text):
        token_list = []
        tagged_list = []
        pos_universal_list = []
        pos_token_tag_sentence = CMUTweetTagger.runtagger_parse(text)

        for sequence_tag in pos_token_tag_sentence:
            tokens = []
            tagged = []
            pos_universal = []
            for token_tag in sequence_tag:
                tokens.append(token_tag[0])
                tagged.append(self.convert_cmu_tags(token_tag[1]))
                pos_universal.append(self.convert_cmu_tags(token_tag[1], tagset="universal"))

            token_list.append(tokens)
            tagged_list.append(zip(tokens, tagged))
            pos_universal_list.append(zip(tokens, pos_universal))
        return token_list, tagged_list, pos_universal_list

    def convert_cmu_tags(self, tag, tagset=None):
        if tagset == "universal":
            return self.convert_cmu_to_universal_tags(tag)
        return self.convert_cmu_to_peen_tags(tag)

    @staticmethod
    def convert_cmu_to_peen_tags(cmu_tag):
        for item in definitions.CMU_PENN_TAGS:
            if item[0] == cmu_tag:
                return item[1]
        return cmu_tag

    @staticmethod
    def convert_cmu_to_universal_tags(cmu_tag):
        for item in definitions.CMU_UNI_TAGS:
            if item[0] == cmu_tag:
                return item[1]
        return "X"