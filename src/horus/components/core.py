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
import codecs
import csv
import heapq
import json
import logging
import sqlite3
import string
from time import gmtime, strftime
import re
import chardet
import cv2
import langdetect
import nltk
import numpy
import requests
import unicodedata
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from microsofttranslator import Translator
from nltk.tokenize import sent_tokenize
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import tree, preprocessing
from bingAPI1 import bing_api2, bing_api5
from config import HorusConfig
from horus import definitions
from horus.components.systemlog import SystemLog
from horus.components.util.nlp_tools import NLPTools
import pandas as pd
import cgi

#print cv2.__version__


class Core(object):
    """ Description:
            A core module for config.
        Attributes:
            name: A string representing the customer's name.
            balance: A float tracking the current balance of the customer's account.
    """

    # static methods
    version = "0.1.5"
    version_label = "HORUS 0.1.5"

    def deletar_depois(self):
        try:
            c = self.conn.cursor()
            c2 = self.conn.cursor()
            self.conn.text_factory = str
            sql = """SELECT id, annotator_nltk_compounds, annotator_stanford_compounds, annotator_tweetNLP_compounds
                     FROM HORUS_SENTENCES"""
            c.execute(sql)
            res = c.fetchall()
            if not res is None:
                for reg in res:
                    id = reg[0]
                    u0 = json.loads(reg[1])
                    u1 = json.loads(reg[2])
                    u2 = json.loads(reg[3])
                    for item in u0:
                        item[0] = int(item[0]) + 1
                    for item in u1:
                        item[0] = int(item[0]) + 1
                    for item in u2:
                        item[0] = int(item[0]) + 1
                    sql = """UPDATE HORUS_SENTENCES SET annotator_nltk_compounds = ?,
                              annotator_stanford_compounds = ?, annotator_tweetNLP_compounds = ?
                            WHERE id = ?"""
                    c2.execute(sql, (json.dumps(u0), json.dumps(u1), json.dumps(u2), id))
            #self.conn.commit() -> ja fiz o que tinha que fazer...
        except Exception as e:
            print e
            self.conn.rollback()

    def __init__(self,force_download=False,trees=5):
        """Return a HORUS object"""
        self.sys = SystemLog("horus.log", logging.DEBUG, logging.INFO)
        self.config = HorusConfig()

        self.sys.log.info('------------------------------------------------------------------')
        self.sys.log.info('::                       HORUS ' + self.version + '                            ::')
        self.sys.log.info('------------------------------------------------------------------')
        self.sys.log.info(':: loading components...')
        self.tools = NLPTools()
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
        self.text_checking_model_1 = joblib.load(self.config.models_1_text)
        self.text_checking_model_2 = joblib.load(self.config.models_2_text)
        self.text_checking_model_3 = joblib.load(self.config.models_3_text)
        self.text_checking_model_4 = joblib.load(self.config.models_4_text)
        self.text_checking_model_5 = joblib.load(self.config.models_5_text)
        self.final = joblib.load(self.config.model_final)
        self.final_encoder = joblib.load(self.config.model_final_encoder)

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

    html_escape_table = {
        "&": "&amp;",
        '"': "&quot;",
        "'": "&apos;",
        ">": "&gt;",
        "<": "&lt;",
    }

    def html_escape(self, text):
        return "".join(self.html_escape_table.get(c, c) for c in text)

    def get_ner_mapping_simple(self, y, x, ix, starty):
        try:
            index = -1
            for k in range(starty, len(y)):
                base = y[k]
                for i in range(ix, len(x)):
                    term = x[i]
                    if self.config.models_pos_tag_lib == 1:  # nltk
                        term = x[i].replace('``', u'"')

                    swap = ''
                    if self.config.models_pos_tag_lib != 3:
                        if term == "''": swap = '"'
                        if term == '"': swap = "''"
                    # tweetNLP
                    #if u'&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;lt' == x[i]:
                    #    term = term.replace(u'&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;lt', u'&lt;')
                    #elif u'&' in x[i]:
                    #    term = term.replace(u'&', u'&amp;')
                    #elif u'<' in x[i]:
                    #    term = term.replace(u'<', u'&lt;')
                    #elif u'>' in x[i]:
                    #    term = term.replace(u'>', u'&gt;')

                    if self.config.models_pos_tag_lib == 3:
                        base = re.sub("&amp;", "&", base)
                        base = re.sub("&quot;", '"', base)
                        base = re.sub("&apos;", "'", base)
                        base = re.sub("&gt;", ">", base)
                        base = re.sub("&lt;", "<", base)
                        term = re.sub("&amp;", "&", term)
                        term = re.sub("&quot;", '"', term)
                        term = re.sub("&apos;", "'", term)
                        term = re.sub("&apos", "'", term) #trick
                        term = re.sub("&gt;", ">", term)
                        term = re.sub("&lt;", "<", term)

                    if term in base or (swap in base if swap != '' else False):
                        index = k
                        if i == ix:
                            return index
            raise Exception

        except Exception as error:
            print error
            exit(-1)

    def get_ner_mapping_slice(self, y, x, ix):

        try:
            for i in range(len(x)):
                x[i] = x[i].replace('``', u'"')
                #x[i] = x[i].replace("''", u'"')




            ##################################################
            # cases (|count(left)| + x + |count(right)| = 7)
            #################################################
            # d
            term = x[ix]
            # d + 6
            merged_aft_7 = x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[ix + 5] + x[ix + 6] + x[ix + 7] \
                if ix + 7 < len(x) else ''
            merged_aft_6 = x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[ix + 5] + x[ix + 6] \
                if ix + 6 < len(x) else ''
            merged_aft_5 = x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[ix + 5] if ix + 5 < len(x) else ''
            merged_aft_4 = x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] if ix + 4 < len(x) else ''
            merged_aft_3 = x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] if ix + 3 < len(x) else ''
            merged_aft_2 = x[ix] + x[ix + 1] + x[ix + 2] if ix + 2 < len(x) else ''
            merged_aft_1 = x[ix] + x[ix + 1] if ix + 1 < len(x) else ''

            # d - 7
            merged_bef_7 = x[ix - 7] + x[ix - 6] + x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] \
                if ix >= 7 else ''
            merged_bef_6 = x[ix - 6] + x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] \
                if ix >= 6 else ''
            merged_bef_5 = x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] \
                if ix >= 5 else ''
            merged_bef_4 = x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] \
                if ix >= 4 else ''
            merged_bef_3 = x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] \
                if ix >= 3 else ''
            merged_bef_2 = x[ix - 2] + x[ix - 1] + x[ix] \
                if ix >= 2 else ''
            merged_bef_1 = x[ix - 1] + x[ix] \
                if ix >= 1 else ''

            # -1 d +1
            merged_bef_1_aft_1 = x[ix - 1] + x[ix] + x[ix + 1] \
                if (ix + 1 < len(x) and ix >= 1) else ''
            # -2 d +2
            merged_bef_2_aft_2 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] \
                if (ix + 2 < len(x) and ix >= 2) else ''
            # -3 d +3
            merged_bef_3_aft_3 = x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] \
                if (ix + 3 < len(x) and ix >= 3) else ''

            # -1 d +2..5
            merged_bef_1_aft_2 = x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] \
                if (ix + 2 < len(x) and ix >= 1) else ''
            merged_bef_1_aft_3 = x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] \
                if (ix + 3 < len(x) and ix >= 1) else ''
            merged_bef_1_aft_4 = x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] \
                if (ix + 4 < len(x) and ix >= 1) else ''
            merged_bef_1_aft_5 = x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[ix + 5] \
                if (ix + 5 < len(x) and ix >= 1) else ''
            merged_bef_1_aft_6 = x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[ix + 5] + x[ix + 6] \
                if (ix + 6 < len(x) and ix >= 1) else ''


            # -2..5 d +1
            merged_bef_2_aft_1 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] \
                if (ix + 1 < len(x) and ix >= 2) else ''
            merged_bef_3_aft_1 = x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] \
                if (ix + 1 < len(x) and ix >= 3) else ''
            merged_bef_4_aft_1 = x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] \
                if (ix + 1 < len(x) and ix >= 4) else ''
            merged_bef_5_aft_1 = x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] \
                if (ix + 1 < len(x) and ix >= 5) else ''
            merged_bef_6_aft_1 = x[ix - 6] + x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] \
                if (ix + 1 < len(x) and ix >= 6) else ''
            merged_bef_5_aft_2 = x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] \
                if (ix + 2 < len(x) and ix >= 5) else ''

            # -2 d +3..5
            merged_bef_2_aft_3 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] \
                if (ix + 3 < len(x) and ix >= 2) else ''
            merged_bef_2_aft_4 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] \
                if (ix + 4 < len(x) and ix >= 2) else ''
            merged_bef_2_aft_5 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[ix + 5] \
                if (ix + 5 < len(x) and ix >= 2) else ''


            # -3..4 d +2
            merged_bef_3_aft_2 = x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] \
                if (ix + 2 < len(x) and ix >= 3) else ''
            merged_bef_3_aft_4 = x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] \
                if (ix + 4 < len(x) and ix >= 3) else ''
            merged_bef_4_aft_2 = x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2]  \
                if (ix + 2 < len(x) and ix >= 4) else ''
            merged_bef_4_aft_3 = x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] \
                + x[ix + 3] if (ix + 3 < len(x) and ix >= 4) else ''

            seq = [[term, -1, 1],
                   [merged_aft_1, -1, 2],
                   [merged_aft_2, -1, 3],
                   [merged_aft_3, -1, 4],
                   [merged_aft_4, -1, 5],
                   [merged_aft_5, -1, 6],
                   [merged_aft_6, -1, 7],
                   [merged_aft_7, -1, 8],
                   [merged_bef_1, -2, 1],
                   [merged_bef_2, -3, 1],
                   [merged_bef_3, -4, 1],
                   [merged_bef_4, -5, 1],
                   [merged_bef_5, -6, 1],
                   [merged_bef_6, -7, 1],
                   [merged_bef_7, -8, 1],
                   [merged_bef_1_aft_1, -2, 2],
                   [merged_bef_2_aft_2, -3, 3],
                   [merged_bef_3_aft_3, -4, 4],
                   [merged_bef_1_aft_2, -2, 3],
                   [merged_bef_1_aft_3, -2, 4],
                   [merged_bef_1_aft_4, -2, 5],
                   [merged_bef_1_aft_5, -2, 6],
                   [merged_bef_1_aft_6, -2, 7],
                   [merged_bef_2_aft_1, -3, 2],
                   [merged_bef_3_aft_1, -4, 2],
                   [merged_bef_4_aft_1, -5, 2],
                   [merged_bef_5_aft_1, -6, 2],
                   [merged_bef_2_aft_3, -3, 4],
                   [merged_bef_2_aft_4, -3, 5],
                   [merged_bef_2_aft_5, -3, 6],
                   [merged_bef_3_aft_2, -4, 3],
                   [merged_bef_4_aft_2, -5, 3],
                   [merged_bef_4_aft_3, -5, 4],
                   [merged_bef_6_aft_1, -7, 2],
                   [merged_bef_5_aft_2, -6, 3],
                   [merged_bef_3_aft_4, -4, 5]]

            for s in seq:
                xbefore1 = x[ix + s[1]] if (ix + s[1]) >= 0 else ''

                xbefore2 = x[ix + s[1] - 1] + x[ix + s[1]]  \
                    if (ix + s[1] - 1) >= 0 else ''

                xbefore3 = x[ix + s[1] - 2] + x[ix + s[1] - 1] + x[ix + s[1]]  \
                    if (ix + s[1] - 2) >= 0 else ''

                xbefore4 = x[ix + s[1] - 3] + x[ix + s[1] - 2] + x[ix + s[1] - 1] + x[ix + s[1]] \
                    if (ix + s[1] - 3) >= 0 else ''

                xbefore5 = x[ix + s[1] - 4] + x[ix + s[1] - 3] + x[ix + s[1] - 2] + x[ix + s[1] - 1] + x[ix + s[1]] \
                    if (ix + s[1] - 4) >= 0 else ''

                xafter4 = x[ix + s[2]] + x[ix + s[2] + 1] + x[ix + s[2] + 2] + x[ix + s[2] + 3] \
                    if (ix + s[2] + 3) < len(x) else ''

                xafter3 = x[ix + s[2]] + x[ix + s[2] + 1] + x[ix + s[2] + 2] \
                    if (ix + s[2] + 2) < len(x) else ''

                xafter2 = x[ix + s[2]] + x[ix + s[2] + 1]  \
                    if (ix + s[2] + 1) < len(x) else ''

                xafter1 = x[ix + s[2]] \
                    if (ix + s[2]) < len(x) else ''

                for iy in range(len(y)):
                    ybefore = y[iy - 1] if iy > 0 else ''
                    yafter = y[iy + 1] if iy + 1 < len(y) else ''
                    print '    ybefore: %s, y: %s, yafter: %s' % (ybefore, y[iy], yafter)
                    if (y[iy] == s[0] or y[iy] == s[0].replace(u'"', u"''")) and (ybefore == xbefore1 or
                                          ybefore == xbefore2 or
                                          ybefore == xbefore3 or
                                          ybefore == xbefore4 or
                                          ybefore == xbefore5 or
                                          yafter == xafter1 or
                                          yafter == xafter2 or
                                          yafter == xafter3 or
                                          yafter == xafter4):
                        return iy

            print 'index not found'
            exit(-1)
        except Exception as error:
            exit(-1)

    def get_ner_mapping2_loop(self, x, y, ix, term):

        index_token = -1

        try:
            # http://stackoverflow.com/questions/32185072/nltk-word-tokenize-behaviour-for-double-quotation-marks-is-confusing
            if self.config.models_pos_tag_lib == 1:
                if term == '``' or term == '\'\'':
                    term = u'"'

            for i in range(len(x)):
                x[i] = x[i].replace('``', u'"')
                # x[i] = x[i].replace('\'\'', u'"')

            xpos = (x[ix + 1] if ix + 1 < len(x) else '')
            xpre = (x[ix - 1] if ix > 0 else '')
            xpos2 = (x[ix + 1] + x[ix + 2] if ix + 2 < len(x) else '')
            xpre2 = (x[ix - 2] + x[ix - 1] if ix > 1 else '')

            print '================================================================'
            print 'x = %s' % (x)
            print 'y = %s' % (y)

            print 'ix = %s, term = %s' % (ix, term)
            print 'xpre2 = %s, xpre = %s, xpos = %s, xpos2 = %s' % (xpre2, xpre, xpos, xpos2)

            q = True
            # optimization trick
            start = 0
            #if i >= 14:
            #    start = i - 14
            #elif i >= 13:
            #    start = i - 13
            #elif i >= 12:
            #    start = i - 12
            # tries to get a single not aligned token
            for z in range(start, len(y)):
                try:

                    ypos = (y[z + 1] if z + 1 < len(y) else '')
                    ypre = (y[z - 1] if z > 0 else '')
                    ypos2 = (y[z + 1] + y[z + 2] if z + 2 < len(y) else '')
                    ypre2 = (y[z - 2] + y[z - 1] if z > 1 else '')

                    print '----------------------------------------------------------'
                    print 'ypre2 = %s, ypre = %s, ypos = %s, ypos2 = %s' % (ypre2, ypre, ypos, ypos2)
                    print 'z: y[z] = %s [%s]' % (z, y[z])

                    fine1 = (xpos == ypos2)
                    fine2 = (xpos == ypos)
                    fine3 = (xpos2 == ypos2)
                    fine4 = (xpos2 == ypos)

                    fine5 = (xpre == ypre2)
                    fine6 = (xpre == ypre)
                    fine7 = (xpre2 == ypre2)
                    fine8 = (xpre2 == ypre)

                    p = '_'
                    if ix + 1 < len(x):
                            p = (term + x[ix+1])
                    if (y[z] == term or y[z] == p) and (fine1 or fine2 or fine3 or fine4 or fine5 or fine6 or fine7 or fine8):
                        #  ok, is the correct one by value and position
                        index_token = y.index(y[z])
                        q = False
                        break
                except Exception:
                    continue
            # start to merge stuff and try to locate it
            merged=''
            print '-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-'
            ntimes = len(x) - start
            print 'ntimes = %s' % (ntimes)
            while q is True:
                for slide in range(ntimes):
                    print 'slide = %s' % (slide)
                    merged = ''
                    if q is False:
                        break
                    for m in range(start, len(x)): #start, len(x)
                        xm = x[m].replace(u'``', u'"').replace('\'\'', u'"')
                        merged = merged + xm
                        print 'm = %s, xm = %s, merged = %s' % (m, xm, merged)
                        try:
                            index_token = y.index(merged)
                            af = (x[ix + 1] if ix + 1 < len(x) else '')
                            bf = (x[ix - 1] if ix > 0 else '')

                            af = af.replace(u'``', u'"') #.replace('\'\'', u'"')
                            bf = bf.replace(u'``', u'"')

                            print 'af = %s, bf = %s' % (af, bf)

                            if term in merged and (af in merged or bf in merged):  # if it is merged, at least 2 MUST be included
                                q = False
                                break
                        except Exception:
                            continue
                    start+=1
                if q is True:
                    return None

        except Exception as e:
            self.sys.log.error(':: error on get ner: %s' % e)
            exit(-1)

        return index_token

    def get_ner_mapping2(self, y, x, t, i):
        if i+1 < len(y):
            if y[i] == t:
                return i
        index_token = self.get_ner_mapping2_loop(x, y, i, t)
        if index_token is None:
            # means that we looked over all combinations in x that could exist in y
            # if we enter here, means that it is not possible and thus, the given token t has not been
            # tokenized in x, but in y yes! try other way around...
            index_token = self.get_ner_mapping2_loop(y, x, i, t)
            if index_token is None:
                print 'it should never ever happen!!! maiden maiden!'
                exit(-1)
        return index_token

    def get_ner_mapping(self, listy, listx, termx, itermx):
        index_ner_y = -1
        try:
            # lists are aligned
            if listy[itermx] == listx[itermx]:
                return itermx
            else:
                # simplest solution
                for itermy in range(len(listy)-1):
                    if listy[itermy] == termx and (listy[itermy-1] == listx[itermx-1]
                                                    or listy[itermy+1] == listx[itermx+1]):
                        index_ner_y = itermy
                        break
                if index_ner_y != -1:
                    return index_ner_y
                else:
                    try:
                        # from this point on, it' more tricky, so let' start
                        # if t isn't there, automatically, t+1 will not be there also!
                        # Thus, try to catch t+2, i.e.,
                        # we gonna have t that has been tokenized in two parts
                        # (excepting if it' about the last item)
                        if itermx + 2 <= len(listx):
                            next_2_term = listx[itermx + 2]
                            listy.index(next_2_term)  # dummy var - this will raise an error in case of problems
                        # if worked, then merging the 2 before, will make it work
                        term_merged = termx + listx[itermx + 1]
                        index_token = listy.index(term_merged)
                        index_ner_y = index_token
                    except:
                        # checking if current is last
                        try:
                            term_merged = listx[itermx - 1] + termx
                            index_token = listy.index(term_merged)
                            index_ner_y = index_token
                        except:  # try now i + i+1 + i+2!
                            try:
                                term_merged = termx + listx[itermx + 1] + listx[itermx + 2]
                                index_token = listy.index(term_merged)
                                index_ner_y = index_token
                            except:
                                # checking if current is last
                                try:
                                    term_merged = listx[itermx - 2] + listx[itermx - 1] + termx
                                    index_token = listy.index(term_merged)
                                    index_ner_y = index_token
                                except:
                                    print 'maiden maiden...!!!!!'
                                    print termx, itermx
                                    exit(-1)

        except Exception as error:
            self.sys.log.error(':: error on get ner: %s' % error)

        return index_ner_y

    def convert_dataset_to_horus_matrix(self, sentences):
        '''
        converts the list to horus_matrix
        :param sentences
        :return: horus_matrix
        '''
        self.sys.log.info(':: starting conversion to horus_matrix based on system parameters')
        converted = []
        sent_index = 0
        try:
            for sent in sentences:
                sent_index+=1
                for c in range(len(sent[6][self.config.models_pos_tag_lib])):
                    word_index_ref = sent[6][self.config.models_pos_tag_lib][c][0]
                    compound = sent[6][self.config.models_pos_tag_lib][c][1]
                    compound_size = sent[6][self.config.models_pos_tag_lib][c][2]
                    temp = [0, sent_index, word_index_ref, compound, '', '', definitions.KLASSES[4], 1, compound_size]
                    temp.extend(self.populate_matrix_new_columns())
                    temp.extend([definitions.KLASSES[4]])
                    converted.append(temp)
                word_index = 0
                starty = 0
                for i in range(len(sent[2][self.config.models_pos_tag_lib])):
                    term = sent[2][self.config.models_pos_tag_lib][i]
                    if len(sent[2][0]) > 0:
                        ind_ner_real = self.get_ner_mapping_simple(sent[2][0], sent[2][self.config.models_pos_tag_lib], i, starty)
                        starty = ind_ner_real
                        #ind_ner = self.get_ner_mapping_slice(sent[2][0], sent[2][self.config.models_pos_tag_lib], i)
                        #ind_ner = self.get_ner_mapping2(sent[2][0], sent[2][self.config.models_pos_tag_lib], term, i)
                        is_entity = 1 if sent[3][0][ind_ner_real] in definitions.NER_TAGS else 0
                    else:
                        is_entity = -1
                    tag_ner = sent[3][self.config.models_pos_tag_lib][i] if len(sent[3][self.config.models_pos_tag_lib]) > 0 else ''
                    tag_pos = sent[4][self.config.models_pos_tag_lib][i] if len(sent[4][self.config.models_pos_tag_lib]) > 0 else ''
                    tag_pos_uni = sent[5][self.config.models_pos_tag_lib][i] if len(sent[5][self.config.models_pos_tag_lib]) > 0 else ''
                    word_index += 1
                    # we do not know if they have the same alignment, so test it to get the correct tag
                    if len(sent[3][0]) > 0:
                        tag_ner_y = sent[3][0][ind_ner_real]
                        if tag_ner_y in definitions.NER_TAGS_LOC:
                            tag_ner_y = definitions.KLASSES[1]
                        elif tag_ner_y in definitions.NER_TAGS_ORG:
                            tag_ner_y = definitions.KLASSES[2]
                        elif tag_ner_y in definitions.NER_TAGS_PER:
                            tag_ner_y = definitions.KLASSES[3]
                        else:
                            tag_ner_y = definitions.KLASSES[4]
                    else:
                        tag_ner_y = definitions.KLASSES[4]

                    if tag_ner in definitions.NER_TAGS_LOC:
                        tag_ner = definitions.KLASSES[1]
                    elif tag_ner in definitions.NER_TAGS_ORG:
                        tag_ner = definitions.KLASSES[2]
                    elif tag_ner in definitions.NER_TAGS_PER:
                        tag_ner = definitions.KLASSES[3]
                    else:
                        tag_ner = definitions.KLASSES[4]

                    temp = [is_entity, sent_index, word_index, term, tag_pos_uni, tag_pos, tag_ner, 0, 0] # 0-8
                    temp.extend(self.populate_matrix_new_columns())
                    temp.extend([tag_ner_y])
                    converted.append(temp)

        except Exception as error:
            self.sys.log.error(':: Erro! %s' % str(error))
            exit(-1)

        return converted

    def populate_matrix_new_columns(self):
        temp = [] #receives 0=8
        temp.extend([0] * 9)  # 9-17
        temp.extend([definitions.KLASSES[4]])  # 18
        temp.extend([0] * 7)  # 19-25
        temp.extend([definitions.KLASSES[4]])  # 26
        temp.extend([0] * 9)  # 27-35
        temp.extend([definitions.KLASSES[4]] * 15)  # 36-50
        return temp

    def annotate(self, input_text, input_file=None, ds_format=0, output_file='horus_out', output_format="csv", ds_name=None):
        try:
            # 0 = text (parameter of reading file) / 1 = ritter
            if int(ds_format) == 0:
                text = ''
                if input_text is not None:
                    self.sys.log.info(':: processing text')
                    text = input_text.strip('"\'')
                elif input_file is not None:
                    self.sys.log.info(':: processing input file')
                    f = open(input_file, 'r')
                    text = f.readlines()
                else:
                    raise Exception("err: missing text to be annotated")

                sent_tokenize_list = self.process_input_text(text)

            elif int(ds_format) == 1: #CoNLL format
                if input_file is None:
                    raise Exception("Provide an input file format to be annotated")
                self.sys.log.info(':: processing CoNLL format -> %s' % ds_name)
                sent_tokenize_list = self.process_ds_conll_format(input_file, ds_name)

            df = pd.DataFrame(sent_tokenize_list)

            self.sys.log.info(':: %s sentence(s) cached' % str(len(sent_tokenize_list)))
            tot_sentences_with_entity = len(df.loc[df[0] == 1])
            tot_others = len(df.loc[df[0] == -1])
            self.sys.log.info(':: %s sentence(s) with entity' % tot_sentences_with_entity)
            self.sys.log.info(':: %s sentence(s) without entity' % tot_others)
            self.horus_matrix = self.convert_dataset_to_horus_matrix(sent_tokenize_list)


            hm = pd.DataFrame(self.horus_matrix)
            self.sys.log.info(':: basic POS statistics')
            a = len(hm)  # all
            a2 = len(hm[(hm[7] == 0)])  # all excluding compounds
            plo = hm[(hm[7] == 0) & (hm[0] == 1)]  # all PLO entities (not compound)
            not_plo = hm[(hm[7] == 0) & (hm[0] == 0)]  # all PLO entities (not compound)

            pos_ok_plo = plo[(plo[5].isin(definitions.POS_NOUN_TAGS))]
            pos_not_ok_plo = plo[(~plo[5].isin(definitions.POS_NOUN_TAGS))]
            pos_noun_but_not_entity = not_plo[(not_plo[5].isin(definitions.POS_NOUN_TAGS))]

            self.sys.log.info(':: [basic statistics]')
            self.sys.log.info(':: -> ALL terms: %s ' % a)
            self.sys.log.info(':: -> ALL tokens (no compounds): %s (%.2f)' % (a2, (a2 / float(a))))
            self.sys.log.info(':: -> ALL NNs (no compounds nor entities): %s ' % len(pos_noun_but_not_entity))
            self.sys.log.info(':: [test dataset statistics]')
            self.sys.log.info(':: -> PLO entities (no compounds): %s (%.2f)' % (len(plo), len(plo) / float(a2)))
            self.sys.log.info(':: -> PLO entities correctly classified as NN (POS says is NOUN): %s (%.2f)' %
                              (len(pos_ok_plo), len(pos_ok_plo) / float(len(plo)) if len(plo)!=0 else 0))
            self.sys.log.info(':: -> PLO entities misclassified (POS says is NOT NOUN): %s (%.2f)' %
                              (len(pos_not_ok_plo), len(pos_not_ok_plo) / float(len(plo)) if len(plo)!=0 else 0))

            if len(self.horus_matrix) > 0:
                self.download_and_cache_results()
                self.detect_objects()
                self.update_compound_predictions()
                #self.run_final_classifier()
                self.export_data(output_file, output_format)
                if int(ds_format) == 0:
                    self.print_annotated_sentence()

            self.conn.close()
            return self.horus_matrix

        except Exception as error:
            print('caught this error: ' + repr(error))

    def export_data(self, output_file, output_format):
        self.sys.log.info(':: exporting metadata to: ' + self.config.output_path + output_file + "." + output_format)

        if output_file == '':
            output_file = 'noname'
        if output_format == 'json':
            with open(self.config.output_path + output_file + '.json', 'wb') as outfile:
                json.dump(self.horus_matrix, outfile)
        elif output_format == 'csv':
            horus_csv = open(self.config.output_path + output_file + '.csv', 'wb')
            wr = csv.writer(horus_csv, quoting=csv.QUOTE_ALL)
            wr.writerow(definitions.HORUS_MATRIX_HEADER)
            #wr.writerow([s.encode('utf8') if type(s) is unicode else s for s in self.horus_matrix])
            wr.writerows(self.horus_matrix)

    def convert_unicode(s):
        # u'abc'.encode('utf-8') -> unicode to str
        # 'abc'.decode('utf-8') -> str to unicode
        if isinstance(s, str):
            return s.decode('utf8') #unicode(s, 'utf8 )
        elif isinstance(s, unicode):
            return s
        else:
            raise Exception ("that's not a string!")


    def run_final_classifier(self):
        self.sys.log.info(':: running final classifier...')
        try:
            for index in range(len(self.horus_matrix)):
                features = []
                pos_bef = ''
                pos_aft = ''
                if index > 1:
                    pos_bef = self.horus_matrix[index - 1][5]
                if index + 1 < len(self.horus_matrix):
                    pos_aft = self.horus_matrix[index + 1][5]

                one_char_token = 1 if len(self.horus_matrix[index][3]) == 1 else 0
                special_char = 1 if len(re.findall('(http://\S+|\S*[^\w\s]\S*)', self.horus_matrix[index][3])) > 0 else 0
                first_capitalized = 1 if self.horus_matrix[index][3][0].isupper() else 0
                capitalized = 1 if self.horus_matrix[index][3].isupper() else 0
                '''
                    pos-1; pos; pos+1; cv_loc; cv_org; cv_per; cv_dist; cv_plc; 
                    tx_loc; tx_org; tx_per; tx_err; tx_dist; 
                    one_char; special_char; first_cap; cap
                '''
                features.append((pos_bef, self.horus_matrix[index][5], pos_aft, int(self.horus_matrix[index][12]),
                                 int(self.horus_matrix[index][13]), int(self.horus_matrix[index][14]),
                                 int(self.horus_matrix[index][15]),
                                 int(self.horus_matrix[index][16]), #int(self.horus_matrix[index][17])
                                 int(self.horus_matrix[index][20]), int(self.horus_matrix[index][21]),
                                 int(self.horus_matrix[index][22]), float(self.horus_matrix[index][23]),
                                 int(self.horus_matrix[index][24]), #int(self.horus_matrix[index][25])
                                 one_char_token, special_char, first_capitalized, capitalized))

                features = numpy.array(features)
                features[0][0] = self.final_encoder.transform(features[0][0])
                features[0][1] = self.final_encoder.transform(features[0][1])
                features[0][2] = self.final_encoder.transform(features[0][2])
                self.horus_matrix[index][40] = definitions.KLASSES[self.final.predict(features)[0]]

        except Exception as error:
            raise error

    def print_annotated_sentence(self):
        '''
        reads the components matrix and prints the annotated sentences
        :: param horus_matrix:
        :: return: output of annotated sentence
        '''
        x1, x2, x3, x4, x5 = '','','','',''
        id_sent_aux = self.horus_matrix[0][1]
        for token in self.horus_matrix:
            if token[7] == 0:
                if id_sent_aux != token[1]:
                    id_sent_aux = token[1]
                    x1 = ' ' + str(token[3]) + '/' + str(token[36])
                    x2 = ' ' + str(token[3]) + '/' + str(token[37])
                    x3 = ' ' + str(token[3]) + '/' + str(token[38])
                    x4 = ' ' + str(token[3]) + '/' + str(token[39])
                    x5 = ' ' + str(token[3]) + '/' + str(token[40])
                else:
                    x1 += ' ' + str(token[3]) + '/' + str(token[4]) + '/' + str(token[36])
                    x2 += ' ' + str(token[3]) + '/' + str(token[4]) + '/' + str(token[37])
                    x3 += ' ' + str(token[3]) + '/' + str(token[4]) + '/' + str(token[38])
                    x4 += ' ' + str(token[3]) + '/' + str(token[4]) + '/' + str(token[39])
                    x5 += ' ' + str(token[3]) + '/' + str(token[4]) + '/' + str(token[40])

        self.sys.log.info(':: sentence annotated :: ')
        self.sys.log.info(':: KLASS 1 -->: ' + x1)
        self.sys.log.info(':: KLASS 2 -->: ' + x2)
        self.sys.log.info(':: KLASS 3 -->: ' + x3)
        self.sys.log.info(':: KLASS 4 -->: ' + x4)
        self.sys.log.info(':: KLASS 5 -->: ' + x5)

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

            i_sent += 1
            i_word = 1

        self.sys.log.debug(':: done! total of sentences = %s, tokens = %s and compounds = %s'
                     % (str(sent_with_ner), str(token_ok), str(compound_ok)))

    def cache_sentence_conll(self,sentence_list):
        self.sys.log.debug(':: caching coNLL 2003 dataset...:')
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
                for chunck_tag in sent[6]:  # list of chunck tags
                    word = sent[2][i_word - 1]
                    if chunck_tag in "I-NP":  # only NP chunck
                        if prev_tag.replace('I-NP', 'NP').replace('B-NP', 'NP') == chunck_tag.replace('I-NP', 'NP').replace('B-NP', 'NP'):
                            if compound == "":
                                compound += prev_word + ' ' + word + ' '
                            else:
                                compound += word + ' '
                    elif compound != "":
                        prev_tag = ''
                        prev_word = ''
                        compound_ok += 1
                        compound = compound[:-1]
                        self.horus_matrix.append([1, i_sent, i_word - len(compound.split(' ')), compound, '', '', '', 1,
                                                  len(compound.split(' '))])
                        compound = ''
                    prev_word = word
                    prev_tag = chunck_tag
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
                is_entity = 1 if sent[3] in definitions.NER_CONLL else 0

                self.horus_matrix.append([is_entity, i_sent, i_word, sent[2][k], sent[5][k], sent[4][k], sent[3][k], 0, 0])
                i_word += 1
                if is_entity:
                    token_ok += 1

            self.db_save_sentence(sent[1], '-', '-', str(sent[3]))
            i_sent += 1
            i_word = 1

        self.sys.log.debug(':: done! total of sentences = %s, tokens = %s and compounds = %s'
                     % (str(sent_with_ner), str(token_ok), str(compound_ok)))

    def tokenize_and_pos(self, sentence, annotator_id):
        # NLTK
        if annotator_id == 1:
            return self.tools.tokenize_and_pos_nltk(sentence)
        # Stanford
        elif annotator_id == 2:
            return self.tools.tokenize_and_pos_stanford(sentence)
        # TwitterNLP
        elif annotator_id == 3:
            if type(sentence) is not list:
                return self.tools.tokenize_and_pos_twitter(sentence)
            return self.tools.tokenize_and_pos_twitter_list(sentence)

    def cache_sentence(self,sentence_format, sentence_list):
        if sentence_format == 1:
            self.cache_sentence_ritter(sentence_list)
        elif sentence_format == 2:
            self.cache_sentence_conll(sentence_list)

    def update_database_compound(self, sentence_str, compounds):
        c = self.conn.cursor()
        col = "annotator_nltk_compounds"
        if self.config.models_pos_tag_lib == 2:
            col = "annotator_stanford_compounds"
        elif self.config.models_pos_tag_lib == 3:
            col = "annotator_tweetNLP_compounds"
        sql = """UPDATE HORUS_SENTENCES SET ? = ? WHERE sentence = ?"""
        return c.execute(sql, (col, compounds, sentence_str))

    def create_matrix_and_compounds(self, sentence_list):

        i_sent, i_word = 1, 1
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
            #  add compounds of given sentence
            aux = 0
            toparse = []

            for token in sent[2][self.config.models_pos_tag_lib]:
                toparse.append(tuple([token, sent[4][self.config.models_pos_tag_lib][aux]]))
                aux+=1
            t = cp.parse(toparse)

            i_word = 0
            for item in t:
                if type(item) is nltk.Tree:
                    is_entity = 1 if (sent[0] == 1 and sent[3][0][i_word] != 'O') else -1
                    i_word += len(item)
                    if len(item) > 1:  # that's a compound
                        compound = ''
                        for tk in item:
                            compound += tk[0] + ' '

                        self.horus_matrix.append([is_entity, i_sent, i_word - len(item),
                                                  compound[:len(compound) - 1], '', '', '', 1, len(item)])
                        compounds += compound[:len(compound) - 1] + '|'
                        compound = ''
                else:
                    i_word += 1

            # update the database with compounds for given sentence
            upd = self.update_database_compound(sent[1][0], compounds)
            #  transforming to components matrix
            # 0 = is_entity?,    1 = index_sent, 2 = index_word, 3 = word/term,
            # 4 = pos_universal, 5 = pos,        6 = ner       , 7 = compound? ,
            # 8 = compound_size
            i_word = 1
            for k in range(len(sent[2])):
                is_entity = 1 if (sent[0] == 1 and sent[3][k] != 'O') else -1
                self.horus_matrix.append([is_entity, i_sent, i_word, sent[2][k], sent[5][k], sent[4][k], sent[3][k], 0, 0])
                i_word += 1

            i_sent += 1

        # commit updates (compounds)
        self.conn.commit()

    def db_save_sentence(self, sent, corpus):
        try:
            c = self.conn.cursor()
            self.conn.text_factory = str
            sentence = [corpus, sent[0], sent[1][0], sent[1][1], sent[1][2], sent[1][3],
                    json.dumps(sent[2][0]), json.dumps(sent[2][1]), json.dumps(sent[2][2]), json.dumps(sent[2][3]),
                    json.dumps(sent[3][0]), json.dumps(sent[3][1]), json.dumps(sent[3][2]), json.dumps(sent[3][3]),
                    json.dumps(sent[4][0]), json.dumps(sent[4][1]), json.dumps(sent[4][2]), json.dumps(sent[4][3]),
                    json.dumps(sent[5][0]), json.dumps(sent[5][1]), json.dumps(sent[5][2]), json.dumps(sent[5][3]),
                    json.dumps(sent[6][1]), json.dumps(sent[6][2]), json.dumps(sent[6][3])]

            sql = """INSERT INTO HORUS_SENTENCES(corpus_name, sentence_has_NER, sentence,
                            same_tokenization_nltk, same_tokenization_stanford, same_tokenization_tweetNLP,
                            corpus_tokens, annotator_nltk_tokens, annotator_stanford_tokens, annotator_tweetNLP_tokens,
                            corpus_ner_y, annotator_nltk_ner, annotator_stanford_ner, annotator_tweetNLP_ner,
                            corpus_pos_y, annotator_nltk_pos, annotator_stanford_pos, annotator_tweetNLP_pos,
                            corpus_pos_uni_y, annotator_nltk_pos_universal, annotator_stanford_pos_universal, annotator_tweetNLP_pos_universal,
                            annotator_nltk_compounds, annotator_stanford_compounds, annotator_tweetNLP_compounds)
                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
            id = c.execute(sql, sentence)
            self.conn.commit()
            return id.lastrowid

        except Exception as e:
            self.sys.log.error(':: an error has occurred: ', e)
            raise


    def download_image_local(self,image_url, image_type, thumbs_url, thumbs_type, term_id, id_ner_type, seq):
        val = URLValidator()
        auxtype = None
        try:
            val(thumbs_url)
            try:
                img_data = requests.get(thumbs_url).content
                with open('%s%s_%s_%s.%s' % (self.config.cache_img_folder, term_id, id_ner_type, seq, thumbs_type), 'wb') as handler:
                    handler.write(img_data)
                    auxtype = thumbs_type
            except Exception as error:
                print('-> error: ' + repr(error))
        except ValidationError, e:
            self.sys.log.error('No thumbs img here...', e)
            try:
                img_data = requests.get(image_url).content
                with open('%s%s_%s_%s.%s' % (self.config.cache_img_folder, term_id, id_ner_type, seq, image_type), 'wb') as handler:
                    auxtype = image_type
                    handler.write(img_data)
            except Exception as error:
                print('-> error: ' + repr(error))
        return auxtype

    def download_and_cache_results(self):
        try:
            self.sys.log.info(':: caching results...')
            auxc = 1
            c = self.conn.cursor()
            for index in range(len(self.horus_matrix)):
                term = self.horus_matrix[index][3]
                self.sys.log.debug(':: processing term %s - %s [%s]' % (str(auxc), str(len(self.horus_matrix)), term))
                if (self.horus_matrix[index][5] in definitions.POS_NOUN_TAGS) or self.horus_matrix[index][7] == 1:
                    #self.sys.log.debug(':: caching [%s] ...' % term)
                    sql = """SELECT id FROM HORUS_TERM WHERE term = ?"""
                    c.execute(sql, (term,))
                    res_term = c.fetchone()

                    # checking point
                    if res_term is None:
                        self.sys.log.info(':: [%s] has not been cached before!' % term)
                        ret_id_term = c.execute("""INSERT INTO HORUS_TERM(term) VALUES(?)""", (term,))
                        ret_id_term = ret_id_term.lastrowid
                    else:
                        ret_id_term = res_term[0]

                    #we could use a flag here based on last check, but we got a database constraint failure
                    #thus, need to check that. will be corrected next version
                    values = (term, self.config.search_engine_api, self.config.search_engine_features_text)
                    sql = """SELECT id, id_search_type FROM HORUS_TERM_SEARCH WHERE term = ? 
                             AND id_search_engine = ? AND search_engine_features = ? ORDER BY id_search_type ASC"""
                    c.execute(sql, values)
                    res = c.fetchall()
                    if res is None or len(res) != 2: #just one of those is cached
                        whatdowehave = 0 #we need to cache both
                        if len(res) == 1:
                            whatdowehave = res[0][1]
                            if whatdowehave == 1:
                                self.horus_matrix[index][9] = res[0][0]
                            elif whatdowehave == 2:
                                self.horus_matrix[index][10] = res[0][0]
                            else:
                                raise Exception("that should not happen")
                        elif res is not None and len(res) != 0:
                            raise Exception("that should not happen")

                        self.sys.log.info(':: [%s] querying the web ->' % term)
                        metaquery, result_txts, result_imgs = bing_api5(term, key=self.config.search_engine_key, market='en-US')

                        OK = False
                        # we got image (or nothing), thus we need to cache websites (text)
                        if whatdowehave == 2 or whatdowehave == 0:
                            OK = True
                            '''
                            -------------------------------------
                            Web Sites
                            -------------------------------------
                            '''
                            self.sys.log.info(':: [%s] caching (web site) ->' % term)
                            values = (term, ret_id_term, self.config.search_engine_api, 1, self.config.search_engine_features_text,
                            str(strftime("%Y-%m-%d %H:%M:%S", gmtime())), self.config.search_engine_tot_resources,
                            len(result_txts), metaquery)
                            c.execute("""INSERT into HORUS_TERM_SEARCH(term, id_term, id_search_engine, id_search_type,
                                                                 search_engine_features, query_date, query_tot_resource, tot_results_returned, metaquery)
                                                                 VALUES(?,?,?,?,?,?,?,?,?)""", values)
                            id_term_search = c.lastrowid
                            self.horus_matrix[index][9] = id_term_search  # updating matrix

                            seq = 0
                            for web_result in result_txts:
                                seq += 1
                                row = (
                                id_term_search, 0, web_result['id'], seq, web_result['displayUrl'], web_result['name'],
                                web_result['snippet'], '')

                                c.execute("""INSERT INTO HORUS_SEARCH_RESULT_TEXT (id_term_search, id_ner_type,
                                                                     search_engine_resource_id, result_seq, result_url, result_title,
                                                                     result_description, result_html_text) VALUES(?,?,?,?,?,?,?,?)""",
                                          row)
                        # we got text (or nothing), thus we need to cache images
                        if whatdowehave == 1 or whatdowehave == 0:
                            OK = True
                            '''
                            -------------------------------------
                            Images
                            -------------------------------------
                            '''
                            self.sys.log.info(':: [%s] caching - image' % term)
                            values = (
                            term, ret_id_term, self.config.search_engine_api, 2, self.config.search_engine_features_img,
                            str(strftime("%Y-%m-%d %H:%M:%S", gmtime())), self.config.search_engine_tot_resources,
                            len(result_imgs), metaquery)

                            sql = """INSERT into HORUS_TERM_SEARCH(term, id_term, id_search_engine, id_search_type,
                                                             search_engine_features, query_date, query_tot_resource, tot_results_returned, metaquery) VALUES(?,?,?,?,?,?,?,?,?)"""
                            c.execute(sql, values)
                            id_term_img = c.lastrowid
                            self.horus_matrix[index][10] = id_term_img  # updating matrix
                            seq = 0
                            for web_img_result in result_imgs:
                                self.sys.log.debug(':: downloading image [%s]' % (web_img_result['name']))
                                seq += 1
                                auxtype = self.download_image_local(web_img_result['contentUrl'],
                                                                    web_img_result['encodingFormat'],
                                                                    web_img_result['thumbnailUrl'],
                                                                    web_img_result['encodingFormat'], id_term_img, 0, seq)
                                self.sys.log.debug(':: caching image result ...')
                                fname = ('%s_%s_%s.%s' % (str(id_term_img), str(0), str(seq), str(auxtype)))
                                row = (id_term_img, 0, seq, seq, web_img_result['contentUrl'],
                                       web_img_result['name'], web_img_result['encodingFormat'], web_img_result['height'],
                                       web_img_result['width'], web_img_result['thumbnailUrl'],
                                       web_img_result['encodingFormat'], fname)

                                sql = """INSERT INTO HORUS_SEARCH_RESULT_IMG (id_term_search, id_ner_type, 
                                                                 search_engine_resource_id, result_seq, result_media_url, result_media_title,
                                                                 result_media_content_type, result_media_height, result_media_width, 
                                                                 result_media_thumb_media_url, result_media_thumb_media_content_type, filename)
                                                                 VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"""
                                c.execute(sql, row)
                                self.sys.log.debug(':: term [%s] cached (img)!' % term)
                        if OK is False:
                            raise Exception ("that should not happen!")
                        self.conn.commit()
                        OK = False

                    else: #we got both
                        self.horus_matrix[index][9] = res[0][0]
                        self.horus_matrix[index][10] = res[1][0]

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

        from translate import Translator
        self.sys.log.debug(':: text analysis component launched')

        try:

            ret_error = [-1, -1, -1, -1, -1]

            if isinstance(t1, str):
                t1 = unicode(t1, "utf-8")
            if isinstance(t2, str):
                t2 = unicode(t2, "utf-8")

            #print t1.encode("utf-8")
            #print t2.encode("utf-8")

            #t1 = t1.decode('utf-8')
            #t2 = t2.decode('utf-8')

            #content = unicode(t1.strip(codecs.BOM_UTF8), 'utf-8')

            #print self.remove_accents(t1)
            #t1 = self.remove_non_ascii(t1)
            #t2 = self.remove_non_ascii(t2)

            t1final = t1
            t2final = t1

            #https://pypi.python.org/pypi/translate (alternative 1000 per day)
            #https://www.microsoft.com/en-us/translator/getstarted.aspx
            #https://github.com/openlabs/Microsoft-Translator-Python-API

            c = self.conn.cursor()

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
                            return ret_error
                            # updating

                sql = """UPDATE HORUS_SEARCH_RESULT_TEXT
                         SET result_title_en = ? WHERE id = ?"""
                c.execute(sql, (t1final.encode("utf-8"), id))
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
                            return ret_error
                            # updating

                sql = """UPDATE HORUS_SEARCH_RESULT_TEXT
                                SET result_description_en = ? WHERE id = ?"""
                c.execute(sql, (t2final.encode("utf-8"), id))
                self.conn.commit()
            else:
                t2final = t2en

            c.close()

            docs = ["{} {}".format(t1final.encode("utf-8"), t2final.encode("utf-8"))]
            predictions = [self.text_checking_model_1.predict(docs)[0],
                           self.text_checking_model_2.predict(docs)[0],
                           self.text_checking_model_3.predict(docs)[0],
                           self.text_checking_model_4.predict(docs)[0],
                           self.text_checking_model_5.predict(docs)[0]]

            return predictions

            #blob = TextBlob(t2)
            #t22 = blob.translate(to='en')
            # text_vocab = set(w.lower() for w in t2 if w.lower().isalpha())
            # unusual = text_vocab.difference(english_vocab)

        except Exception as e:
            self.sys.log.error(':: Error: ' + str(e))
            return ret_error

    def remove_accents(self, data):
        return ' '.join(x for x in unicodedata.normalize('NFKD', data) if x in string.ascii_letters).lower()

    def remove_non_ascii(self, t):
        import string
        printable = set(string.printable)
        temp = filter(lambda x: x in printable, t)
        return temp
        #return "".join(i for i in temp if ord(i) < 128)

    def detect_objects(self):     # id_term_img, id_term_txt, id_ner_type, term
        self.sys.log.info(':: detecting %s objects...' % len(self.horus_matrix))
        auxi = 0
        toti = len(self.horus_matrix)
        for index in range(len(self.horus_matrix)):
            auxi += 1
            if (self.horus_matrix[index][5] in definitions.POS_NOUN_TAGS) or self.horus_matrix[index][7] == 1:

                term = self.horus_matrix[index][3]
                self.sys.log.info(':: token %d of %d [%s]' % (auxi, toti, term))

                id_term_img = self.horus_matrix[index][10]
                id_term_txt = self.horus_matrix[index][9]
                id_ner_type = 0


                tot_geral_faces = 0
                tot_geral_logos = 0
                tot_geral_locations = 0
                tot_geral_pos_locations = 0
                tot_geral_neg_locations = 0
                T = int(self.config.models_location_theta)  # location threshold

                # -----------------------------------------------------------------
                # image classification
                # -----------------------------------------------------------------

                filesimg = []
                with self.conn:
                    cursor = self.conn.cursor()
                    cursor.execute("""SELECT filename, id, processed, nr_faces, nr_logos, nr_place_1, nr_place_2,
                                             nr_place_3, nr_place_4, nr_place_5, nr_place_6, nr_place_7, nr_place_8,
                                             nr_place_9, nr_place_10 FROM HORUS_SEARCH_RESULT_IMG 
                                      WHERE id_term_search = %s AND id_ner_type = %s """ % (id_term_img, id_ner_type))
                    rows = cursor.fetchall()

                    nr_results_img = len(rows)
                    if nr_results_img == 0:
                        self.sys.log.debug(":: term has not returned images!")
                    limit_img = min(nr_results_img, int(self.config.search_engine_tot_resources))

                    for indeximgs in range(limit_img):  # 0 = file path | 1 = id | 2 = processed | 3=nr_faces | 4=nr_logos | 5 to 13=nr_places_1 to 9
                        filesimg.append((self.config.cache_img_folder + rows[indeximgs][0], rows[indeximgs][1], rows[indeximgs][2],
                                         rows[indeximgs][3], rows[indeximgs][4], rows[indeximgs][5], rows[indeximgs][6], rows[indeximgs][7],
                                         rows[indeximgs][8], rows[indeximgs][9], rows[indeximgs][10], rows[indeximgs][11], rows[indeximgs][12],
                                         rows[indeximgs][13], rows[indeximgs][14]))

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
                        sql = """UPDATE HORUS_SEARCH_RESULT_IMG SET nr_faces = ?, nr_logos = ?, nr_place_1 = ?, 
                                 nr_place_2 = ?, nr_place_3 = ?, nr_place_4 = ?, nr_place_5 = ?, nr_place_6 = ?, 
                                 nr_place_7 = ?, nr_place_8 = ?, nr_place_9 = ?, nr_place_10 = ?, processed = 1
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

                self.horus_matrix[index][11] = limit_img
                self.horus_matrix[index][12] = tot_geral_locations  # 1
                self.horus_matrix[index][13] = tot_geral_logos  # 2
                self.horus_matrix[index][14] = tot_geral_faces  # 3
                self.horus_matrix[index][15] = dist_cv_indicator  # 4
                self.horus_matrix[index][16] = place_cv_indicator  # 5
                self.horus_matrix[index][17] = nr_results_img  # 5

                self.sys.log.debug(':: CV statistics:'
                                   '(LOC=%s, ORG=%s, PER=%s, DIST=%s, PLC=%s)' %
                                   (str(tot_geral_locations).zfill(2), str(tot_geral_logos).zfill(2),
                                    str(tot_geral_faces).zfill(2), str(dist_cv_indicator).zfill(2), place_cv_indicator))

                if limit_img != 0:
                    self.horus_matrix[index][18] = definitions.KLASSES[outs.index(max(outs)) + 1]
                else:
                    self.horus_matrix[index][18] = definitions.KLASSES[4]

                # -----------------------------------------------------------------
                # text classification
                # -----------------------------------------------------------------
                y = []
                with self.conn:
                    cursor = self.conn.cursor()
                    cursor.execute("""SELECT id, result_seq, result_title, result_description, result_title_en,
                                      result_description_en, processed, text_1_klass, text_2_klass, text_3_klass, 
                                      text_4_klass, text_5_klass FROM HORUS_SEARCH_RESULT_TEXT
                                      WHERE id_term_search = %s AND id_ner_type = %s""" % (id_term_txt, id_ner_type))
                    rows = cursor.fetchall()

                    nr_results_txt = len(rows)
                    if nr_results_txt == 0:
                        self.sys.log.debug(":: term has not returned web sites!")
                    limit_txt = min(nr_results_txt, int(self.config.search_engine_tot_resources))

                    tot_err = 0
                    for itxt in range(limit_txt):
                        if rows[itxt][6] == 0 or rows[itxt][6] is None: # not processed yet
                            ret = self.detect_text_klass(rows[itxt][2], rows[itxt][3], rows[itxt][0], rows[itxt][4], rows[itxt][5])
                            y.append(ret)
                            sql = """UPDATE HORUS_SEARCH_RESULT_TEXT SET processed = 1, text_1_klass = %s, text_2_klass = %s,
                                     text_3_klass = %s, text_4_klass = %s, text_5_klass = %s
                                     WHERE id = %s""" % (ret[0], ret[1], ret[2], ret[3], ret[4], rows[itxt][0])
                            cursor.execute(sql)
                            if ret[0] == -1 or ret[1] == -1 or ret[2] == -1 or ret[3] == -1 or ret[4] == -1:
                                tot_err += 1
                        else:
                            y.append(rows[itxt][7:12])

                    self.conn.commit()

                    yy = numpy.array(y)
                    gp = [numpy.count_nonzero(yy == 1), numpy.count_nonzero(yy == 2), numpy.count_nonzero(yy == 3)]
                    horus_tx_ner = gp.index(max(gp)) + 1

                    self.horus_matrix[index][19] = limit_txt
                    self.horus_matrix[index][20] = gp[0]
                    self.horus_matrix[index][21] = gp[1]
                    self.horus_matrix[index][22] = gp[2]
                    self.horus_matrix[index][23] = float(tot_err)

                    maxs_tx = heapq.nlargest(2, gp)
                    dist_tx_indicator = max(maxs_tx) - min(maxs_tx)

                    self.horus_matrix[index][24] = dist_tx_indicator
                    self.horus_matrix[index][25] = nr_results_txt

                    self.sys.log.debug(':: TX statistics:'
                                       '(LOC=%s, ORG=%s, PER=%s, DIST=%s, ERR.TRANS=%s)' %
                                       (str(gp[0]).zfill(2), str(gp[1]).zfill(2), str(gp[2]).zfill(2),
                                        str(dist_tx_indicator).zfill(2), str(tot_err/float(limit_txt)) if limit_txt >0 else 0))
                    self.sys.log.debug('-------------------------------------------------------------')

                    if limit_txt != 0:
                        self.horus_matrix[index][26] = definitions.KLASSES[horus_tx_ner]
                    else:
                        self.horus_matrix[index][26] = definitions.KLASSES[4]

                    # checking final NER based on:
                    #  -> theta
                    if self.horus_matrix[index][15] >= int(self.config.models_distance_theta):
                        self.horus_matrix[index][36] = self.horus_matrix[index][18]   # CV is the final decision
                        self.horus_matrix[index][39] = self.horus_matrix[index][36]   # compound prediction initial
                    elif self.horus_matrix[index][24] >= int(self.config.models_distance_theta):
                        self.horus_matrix[index][36] = self.horus_matrix[index][26]   # TX is the final decision
                        self.horus_matrix[index][39] = self.horus_matrix[index][36]   # compound prediction initial
                    #  -> theta+1
                    if self.horus_matrix[index][15] >= int(self.config.models_distance_theta)+1:
                        self.horus_matrix[index][37] = self.horus_matrix[index][18]  # CV is the final decision
                    elif self.horus_matrix[index][24] >= int(self.config.models_distance_theta)+1:
                        self.horus_matrix[index][37] = self.horus_matrix[index][26]  # TX is the final decision
                    #  -> theta+2
                    if self.horus_matrix[index][15] >= int(self.config.models_distance_theta)+2:
                        self.horus_matrix[index][38] = self.horus_matrix[index][18]  # CV is the final decision
                    elif self.horus_matrix[index][24] >= int(self.config.models_distance_theta)+2:
                        self.horus_matrix[index][38] = self.horus_matrix[index][26]  # TX is the final decision

    def update_rules_cv_predictions(self):
        '''
        updates the predictions based on inner rules
        :return:
        '''
        self.sys.log.info(':: updating predictions based on rules')
        for i in range(len(self.horus_matrix)):
            initial = self.horus_matrix[i][17]
            # get nouns or compounds
            if self.horus_matrix[i][4] == 'NOUN' or \
                            self.horus_matrix[i][4] == 'PROPN' or self.horus_matrix[i][7] == 1:
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
        pre-requisite: the matrix should start with the sentence compounds at the beginning.
        '''
        self.sys.log.info(':: updating compounds predictions')
        i_y, i_sent, i_first_word, i_c_size = [], [], [], []
        for i in range(len(self.horus_matrix)):
            if self.horus_matrix[i][7] == 1:
                i_y.append(self.horus_matrix[i][36]) # KLASS_1
                i_sent.append(self.horus_matrix[i][1])
                i_first_word.append(self.horus_matrix[i][2])
                i_c_size.append(int(self.horus_matrix[i][8]))
            if self.horus_matrix[i][7] == 0:
                for z in range(len(i_y)):
                    if i_sent[z] == self.horus_matrix[i][1] and i_first_word[z] == self.horus_matrix[i][2]:
                        for k in range(i_c_size[z]):
                            self.horus_matrix[i+k][39] = i_y[z] # KLASS_4

    def process_input_text(self, text):
        self.sys.log.info(':: text: ' + text)
        self.sys.log.info(':: tokenizing sentences ...')
        sent_tokenize_list = sent_tokenize(text)
        self.sys.log.info(':: processing ' + str(len(sent_tokenize_list)) + ' sentence(s).')
        sentences = []
        for sentence in sent_tokenize_list:
            sentences.append(self.process_and_save_sentence(-1, sentence))

        return sentences

    def sentence_cached_before(self, corpus, sentence):
        """This method caches the structure of HORUS in db
        in order to speed things up. The average pre-processing time is about to 4-5sec
        for EACH sentence due to attached components (eg.: stanford tools). If the sentence
        has already been cached, we load and convert strings to list, appending the content
        directly to the matrix, thus optimizing a lot this phase.
        """
        sent = []
        try:
            self.conn.text_factory = str
            sSql = """SELECT sentence_has_NER, 
            sentence, same_tokenization_nltk, same_tokenization_stanford, same_tokenization_tweetNLP,
            corpus_tokens, annotator_nltk_tokens, annotator_stanford_tokens, annotator_tweetNLP_tokens,
            corpus_ner_y, annotator_nltk_ner, annotator_stanford_ner, annotator_tweetNLP_ner,
            corpus_pos_y, annotator_nltk_pos, annotator_stanford_pos, annotator_tweetNLP_pos,
            corpus_pos_uni_y, annotator_nltk_pos_universal, annotator_stanford_pos_universal, annotator_tweetNLP_pos_universal,
            annotator_nltk_compounds, annotator_stanford_compounds, annotator_tweetNLP_compounds
            FROM HORUS_SENTENCES
            WHERE sentence = ? and corpus_name = ?"""
            c = self.conn.execute(sSql, (sentence, corpus))
            ret = c.fetchone()
            if ret is not None:
                sent.append(ret[0])
                sent.append([ret[1], ret[2], ret[3], ret[4]])
                sent.append([json.loads(ret[5]), json.loads(ret[6]), json.loads(ret[7]), json.loads(ret[8])])
                sent.append([json.loads(ret[9]), json.loads(ret[10]), json.loads(ret[11]), json.loads(ret[12])])
                sent.append([json.loads(ret[13]), json.loads(ret[14]), json.loads(ret[15]), json.loads(ret[16])])
                sent.append([json.loads(ret[17]), json.loads(ret[18]), json.loads(ret[19]), json.loads(ret[20])])
                sent.append([json.loads('[]'), json.loads(ret[21]), json.loads(ret[22]), json.loads(ret[23])])
        except Exception as e:
            self.sys.log.error(':: an error has occurred: ', e)
            raise
        return sent

    def get_compounds(self, tokens):
        compounds = []
        pattern = """
                                    NP:
                                       {<JJ>*<NN|NNS|NNP|NNPS><CC>*<NN|NNS|NNP|NNPS>+}
                                       {<NN|NNS|NNP|NNPS><IN>*<NN|NNS|NNP|NNPS>+}
                                       {<JJ>*<NN|NNS|NNP|NNPS>+}
                                       {<NN|NNP|NNS|NNPS>+}
                                    """
        cp = nltk.RegexpParser(pattern)
        toparse = []
        for token in tokens:
            toparse.append(tuple([token[0], token[1]]))
        t = cp.parse(toparse)

        i_word = 0
        for item in t:
            if type(item) is nltk.Tree:
                i_word += len(item)
                if len(item) > 1:  # that's a compound
                    compound = ''
                    for tk in item:
                        compound += tk[0] + ' '

                    compounds.append([i_word - len(item) + 1, compound[:len(compound) - 1], len(item)])
            else:
                i_word += 1
        return compounds

    def process_and_save_sentence(self, hasNER, s, dataset_name = '', tokens_gold_standard = [], ner_gold_standard = []):
        # that' a sentence, check if cached!
        cache_sent = self.sentence_cached_before(dataset_name, s)
        if len(cache_sent) != 0:
            return cache_sent
        else:

            _tokens_nltk, _pos_nltk, _pos_uni_nltk = self.tokenize_and_pos(s, 1)
            _tokens_st, _pos_st, _pos_uni_st = self.tokenize_and_pos(s, 2)
            _tokens_twe, _pos_twe, _pos_uni_twe = self.tokenize_and_pos(s, 3)

            _pos_nltk = numpy.array(_pos_nltk)
            _pos_uni_nltk = numpy.array(_pos_uni_nltk)
            _pos_st = numpy.array(_pos_st)
            _pos_uni_st = numpy.array(_pos_uni_st)
            _pos_twe = numpy.array(_pos_twe)
            _pos_uni_twe = numpy.array(_pos_uni_twe)

            # nltk tok has the same length of corpus tok?
            _same_tok_nltk = (len(_tokens_nltk) == len(tokens_gold_standard))

            # stanford tok has the same length of corpus tok?
            _same_tok_stanf = (len(_tokens_st) == len(tokens_gold_standard))

            # tweetNLP tok has the same length of corpus tok?
            _same_tok_tweet = (len(_tokens_twe) == len(tokens_gold_standard))

            # NLTK NER
            nernltktags = self.tools.annotate_ner_nltk(_pos_nltk)

            # stanford NER
            nerstantags = self.tools.annotate_ner_stanford(s)
            nerstantags = numpy.array(nerstantags)

            comp_nltk = self.get_compounds(_pos_nltk)
            comp_st = self.get_compounds(_pos_st)
            comp_twe = self.get_compounds(_pos_twe)

            # saving to database (pos_uni_sta not implemented yet)
            sent = [hasNER,
                    [s, 1 if _same_tok_nltk else 0, 1 if _same_tok_stanf else 0, 1 if _same_tok_tweet else 0],
                    [tokens_gold_standard, _tokens_nltk, _tokens_st, _tokens_twe],
                    [ner_gold_standard, nernltktags, nerstantags[:, 1].tolist(), []],
                    [[], _pos_nltk[:, 1].tolist(), _pos_st[:, 1].tolist(), _pos_twe[:, 1].tolist()],
                    [[], _pos_uni_nltk[:, 1].tolist(), [], _pos_uni_twe[:, 1].tolist()],
                    [[], comp_nltk, comp_st, comp_twe]
                    ]

            self.db_save_sentence(sent, dataset_name)
            return sent

    def process_ds_conll_format(self, dspath, dataset_name):
        '''
        return a set of sentences
        :param dspath: path to Ritter dataset
        :return: sentence contains any entity?, sentence, words, NER tags
        '''
        try:
            # sentences
            sentences = []
            # default corpus tokens
            tokens = []
            # correct corpus NER tags
            tags_ner_y = []
            s = ''
            has3NER = -1
            tot_sentences = 1
            self.sys.log.info(':: processing sentences...')
            with open(dspath) as f:
                for line in f:
                    if line.strip() != '':
                        token = line.split('\t')[0]
                        ner = line.split('\t')[1].replace('\r','').replace('\n','')
                    if line.strip() == '':
                        if len(tokens) != 0:
                            self.sys.log.debug(':: processing sentence %s' % str(tot_sentences))
                            sentences.append(self.process_and_save_sentence(has3NER, s, dataset_name, tokens, tags_ner_y))
                            tokens = []
                            tags_ner_y = []
                            s = ''
                            has3NER = -1
                            tot_sentences += 1
                    else:
                        s += token + ' '
                        tokens.append(token)
                        tags_ner_y.append(ner)
                        if ner in definitions.NER_RITTER:
                            has3NER = 1

            self.sys.log.info(':: %s sentences processed successfully' % str(len(sentences)))
            return sentences
        except Exception as error:
            print('caught this error: ' + repr(error))

    def processing_conll_ds(self, dspath):
        sentences = []
        w = []
        t = []
        p = []
        pu = []
        c = []
        s = ''
        has3NER = -1
        with open(dspath) as f:
            for line in f:
                text = line.split(' ')
                token = text[0].replace('\n', '')
                if token == '':
                    if len(w) != 0:
                        sentences.append([has3NER, s, w, t, p, pu, c])
                        w = []
                        t = []
                        p = []
                        pu = []
                        c = []
                        s = ''
                        has3NER = -1
                else:
                    pos_tag = text[1]
                    chunck_tag = text[2]
                    ner_tag = text[3].replace('\n', '')
                    s += token + ' '
                    w.append(token)
                    p.append(pos_tag)
                    pu.append(self.tools.convert_penn_to_universal_tags(pos_tag))
                    t.append(ner_tag)
                    c.append(chunck_tag)
                    if ner_tag in definitions.NER_CONLL:
                        has3NER = 1
        return sentences



