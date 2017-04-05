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

    def __init__(self,force_download,trees):
        """Return a HORUS object"""
        self.sys = SystemLog("horus.log", logging.DEBUG, logging.DEBUG)
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
        0 = entity (-1,0,1)
        1 = index_sentence
        2 = index_token
        3 = token
        4 = pos_universal_tag (deprecated) 
        5 = pos_tag
        6 = ner_tag
        7 = compound (1-0)
        8 = compound_size
        :param sentences:
        :return:horus_matrix
        '''
        temp = []
        sent_index = 0
        try:
            for sent in sentences:
                sent_index+=1
                for c in range(len(sent[6][self.config.models_pos_tag_lib])):
                    word_index_ref = sent[6][self.config.models_pos_tag_lib][c][0]
                    compound = sent[6][self.config.models_pos_tag_lib][c][1]
                    compound_size = sent[6][self.config.models_pos_tag_lib][c][2]
                    temp.append([1, sent_index, word_index_ref, compound, '', '', '', 1, compound_size])
                word_index = 0
                starty = 0
                for i in range(len(sent[2][self.config.models_pos_tag_lib])):
                    term = sent[2][self.config.models_pos_tag_lib][i]
                    ind_ner = self.get_ner_mapping_simple(sent[2][0], sent[2][self.config.models_pos_tag_lib], i, starty)
                    starty = ind_ner
                    #ind_ner = self.get_ner_mapping_slice(sent[2][0], sent[2][self.config.models_pos_tag_lib], i)
                    #ind_ner = self.get_ner_mapping2(sent[2][0], sent[2][self.config.models_pos_tag_lib], term, i)
                    is_entity = 1 if sent[3][0][ind_ner] in definitions.NER_RITTER else 0
                    tag_pos = sent[self.config.models_pos_tag_lib_type][self.config.models_pos_tag_lib][i]
                    word_index += 1
                    # we do not know if they have the same alignment, so test it to get the correct tag
                    tag_ner_y = sent[3][0][ind_ner]
                    temp.append([is_entity, sent_index, word_index, term, '', tag_pos, tag_ner_y, 0, 0])

        except Exception as error:
            self.sys.log.error(':: Erro! %s' % str(error))
            exit(-1)

        return temp

    def annotate(self, input_text, input_file=None, ds_format=0, output_file='horus_annotation', output_format="csv", ds_name=None):
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

            elif int(ds_format) == 1 or int(ds_format) == 2:
                if input_file is None:
                    raise Exception("Provide an input file format to be annotated")
                elif int(ds_format) == 1:  # Ritter
                    self.sys.log.info(':: processing CoNLL format -> %s' % ds_name)
                    sent_tokenize_list = self.process_ds_conll_format(input_file, ds_name)
                elif int(ds_format) == 2:  # CoNLL 2003
                    #TODO: merge this method to above
                    sent_tokenize_list = self.processing_conll_ds(input_file)

            df = pd.DataFrame(sent_tokenize_list)
            if ds_format!=0:
                self.sys.log.info(':: %s sentence(s) cached' % str(len(sent_tokenize_list)))
                tot_sentences_with_entity = len(df.loc[df[0] == 1])
                tot_others = len(df.loc[df[0] == -1])
                self.sys.log.info(':: %s sentence(s) with entity' % tot_sentences_with_entity)
                self.sys.log.info(':: %s sentence(s) without entity' % tot_others)
                self.sys.log.info(':: starting conversion to horus_matrix based on system parameters')
                self.horus_matrix = self.convert_dataset_to_horus_matrix(sent_tokenize_list)
            else:
                self.sys.log.info(':: %s sentence(s) cached' % str(len(df[1].unique())))
                self.horus_matrix = sent_tokenize_list


             # hasEntityNER (1=yes,0=dunno,-1=no), sentence, words[], tags_NER[], tags_POS[], tags_POS_UNI[]
            # self.cache_sentence(int(ds_format), sent_tokenize_list)
            #self.sys.log.info(':: done!')

            hm = pd.DataFrame(self.horus_matrix)
            self.sys.log.info(':: basic POS statistics')
            a = len(hm)  # all
            a2 = len(hm[(hm[7] == 0)])  # all excluding compounds
            plo = hm[(hm[7] == 0) & (hm[0] == 1)]  # all PLO entities (not compound)
            not_plo = hm[(hm[7] == 0) & (hm[0] == 0)]  # all PLO entities (not compound)

            pos_ok_plo = plo[(plo[5].isin(definitions.POS_NOUN_TAGS))]
            pos_not_ok_plo = plo[(~plo[5].isin(definitions.POS_NOUN_TAGS))]
            pos_noun_but_not_entity = not_plo[(not_plo[5].isin(definitions.POS_NOUN_TAGS))]

            self.sys.log.info(':: -> ALL terms: %s ' % a)
            self.sys.log.info(':: -> ALL terms (exc. compounds): %s (%.2f)' % (a2, (a2 / float(a))))
            self.sys.log.info(':: -> ALL NNs (exc. compounds and entities): %s ' % len(pos_noun_but_not_entity))
            self.sys.log.info(':: -> PLO entities (exc. compounds): %s (%.2f)' % (len(plo), len(plo) / float(a2)))
            self.sys.log.info(':: -> PLO entities correctly classified as NN (POS says is NOUN): %s (%.2f)' %
                              (len(pos_ok_plo), len(pos_ok_plo) / float(len(plo))) if len(plo)!=0 else 0)
            self.sys.log.info(':: -> PLO entities misclassified (POS says is NOT NOUN): %s (%.2f)' %
                              (len(pos_not_ok_plo), len(pos_not_ok_plo) / float(len(plo))) if len(plo)!=0 else 0)

            if len(self.horus_matrix) > 0:
                self.sys.log.info(':: caching results...')
                self.download_and_cache_results()
                self.sys.log.info(':: done!')

                self.sys.log.info(':: detecting %s objects...' % len(self.horus_matrix))
                self.detect_objects()
                self.sys.log.info(':: done!')


                #self.sys.log.info(':: applying rules...')
                #self.update_rules_cv_predictions()
                self.sys.log.info(':: updating compounds...')
                self.update_compound_predictions()
                self.sys.log.info(':: done!')

                # self.sys.log.info(horus_matrix)
                '''
                IS_ENTITY: -1: unknown; 0: no; 1:yes
                ID_SENT: sentence position
                ID_WORD: term position
                TOKEN: word or term (compound)
                POS_UNI: annotation: universal pos tag
                POS: annotation: pos tag
                NER: annotation: ner (target)
                COMPOUND: compound
                COMPOUND_SIZE: size of compound
                ID_TERM_TXT: id of the table of texts (internal control)
                ID_TERM_IMG: id of the table of images (internal control)
                TOT_IMG: total of resources (img) retrieved (top)
                TOT_CV_LOC: number of resources classified as LOC (computer vision module)
                TOT_CV_ORG: number of resources classified as ORG (computer vision module)
                TOT_CV_PER: number of resources classified as PER (computer vision module)
                DIST_CV_I: distance (subtraction) between 2 max values of (TOT_CV_LOC, TOT_CV_ORG and TOT_CV_PER) (computer vision module)
                PL_CV_I: sum of all LOC classifiers (computer vision module)
                CV_KLASS: target class (computer vision module)
                TOT_RESULTS_TX: total of resources (snippets of text) retrieved (top)
                TOT_TX_LOC: number of resources classified as LOC (text classification module)
                TOT_TX_ORG: number of resources classified as ORG (text classification module)
                TOT_TX_PER: number of resources classified as PER (text classification module)
                TOT_ERR_TRANS: number of exceptions raised by the translation module (text classification module)
                DIST_TX_I: similar to DIST_CV_I (text classification module)
                TX_KLASS: target class (text classification module)
                HORUS_KLASS: final target class
                STANFORD_NER: annotation: NER Stanford
                '''
                header = ["IS_ENTITY", "ID_SENT", "ID_WORD", "TOKEN", "POS_UNI", "POS", "NER", "COMPOUND",
                          "COMPOUND_SIZE", "ID_TERM_TXT", "ID_TERM_IMG", "TOT_IMG", "TOT_CV_LOC", "TOT_CV_ORG",
                          "TOT_CV_PER", "DIST_CV_I", "PL_CV_I", "CV_KLASS", "TOT_RESULTS_TX", "TOT_TX_LOC", "TOT_TX_ORG",
                          "TOT_TX_PER", "TOT_ERR_TRANS", "DIST_TX_I", "TX_KLASS", "HORUS_KLASS", "STANFORD_NER"]

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

            self.conn.close()
            return self.horus_matrix

        except Exception as error:
            print('caught this error: ' + repr(error))

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
        c = self.conn.cursor()
        self.conn.text_factory = str
        sql = """SELECT id FROM HORUS_SENTENCES
                 WHERE sentence = ? and corpus_name = ? """
        c.execute(sql, (str(sent[1][0]), corpus))
        res = c.fetchone()
        id = -1
        if res is None:
            self.sys.log.debug(':: caching ... ')
            #buffer(zlib.compress
            #row = (str(sent), str(tagged), str(compound), str(tokens))
            # sent[6][0] is a dummy variable!
            sentence = [sent[0], corpus, sent[1][0], sent[1][1], sent[1][2], sent[1][3],
                   json.dumps(sent[2][0]), json.dumps(sent[2][1]), json.dumps(sent[2][2]), json.dumps(sent[2][3]),
                   json.dumps(sent[3][0]), json.dumps(sent[3][1]), json.dumps(sent[3][2]), json.dumps(sent[3][3]),
                   json.dumps(sent[4][0]), json.dumps(sent[4][1]), json.dumps(sent[4][2]), json.dumps(sent[4][3]),
                   json.dumps(sent[5][0]), json.dumps(sent[5][1]), json.dumps(sent[5][2]), json.dumps(sent[5][3]),
                   json.dumps(sent[6][1]), json.dumps(sent[6][2]), json.dumps(sent[6][3])]
            #row = (sentence)
            sql = """INSERT INTO HORUS_SENTENCES(sentence_has_NER, corpus_name,
                          sentence, same_tokenization_nltk, same_tokenization_stanford, same_tokenization_tweetNLP,
                          corpus_tokens, annotator_nltk_tokens, annotator_stanford_tokens, annotator_tweetNLP_tokens,
                          corpus_ner_y, annotator_nltk_ner, annotator_stanford_ner, annotator_tweetNLP_ner,
                          corpus_pos_y, annotator_nltk_pos, annotator_stanford_pos, annotator_tweetNLP_pos,
                          corpus_pos_uni_y, annotator_nltk_pos_universal, annotator_stanford_pos_universal, annotator_tweetNLP_pos_universal,
                          annotator_nltk_compounds, annotator_stanford_compounds, annotator_tweetNLP_compounds)
                             VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
            id = c.execute(sql, sentence)
            self.conn.commit()
            id = id.lastrowid
        else:
            self.sys.log.debug(':: ... already cached')
            id = res[0]
        return id

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

    def download_and_cache_results(self):
        try:
            auxc = 1
            self.sys.log.info(':: download and cache tokens')
            c = self.conn.cursor()
            for item in self.horus_matrix:
                term_cached = False
                self.sys.log.debug(':: item %s - %s ' % (str(auxc), str(len(self.horus_matrix))))
                term = item[3]
                if (item[5] in definitions.POS_NOUN_TAGS) or item[7] == 1:
                    self.sys.log.debug(':: caching [%s] ...' % term)
                    sql = """SELECT id FROM HORUS_TERM WHERE term = ?"""
                    c.execute(sql, (term,))
                    res_term = c.fetchone()
                    if res_term is None:
                        term_cached = False
                        self.sys.log.info(':: [%s] has not been cached before!' % term)
                        ret_id_term = c.execute("""INSERT INTO HORUS_TERM(term) VALUES(?)""", (term,))
                    else:
                        term_cached = True
                        ret_id_term = res_term[0]

                    if term_cached:
                        self.sys.log.debug(':: term %s is already cached!' % term)
                        # web site
                        values = (term, self.config.search_engine_api, 1, self.config.search_engine_features_text)
                        sql = """SELECT id FROM HORUS_TERM_SEARCH WHERE term = ? AND id_search_engine = ?
                                 AND id_search_type = ? AND search_engine_features = ?"""
                        c.execute(sql, values)
                        res = c.fetchone()
                        if res is None:
                            raise Exception("err: there is a problem in the database (item cached missing txt metadata)")
                        item.extend(res)

                        # image
                        values = (term, self.config.search_engine_api, 2, self.config.search_engine_features_text)
                        sql = """SELECT id FROM HORUS_TERM_SEARCH WHERE term = ? AND id_search_engine = ? 
                                 AND id_search_type = ? AND search_engine_features = ?"""
                        c.execute(sql, values)
                        res = c.fetchone()
                        if res is None:
                            raise Exception("err: there is a problem in the database (item cached missing img metadata)")
                        item.extend(res)

                    else:
                        self.sys.log.info(':: [%s] querying the web ->' % term)
                        metaquery, result_txts, result_imgs = \
                            bing_api5(term, api=self.config.search_engine_key, top=self.config.search_engine_tot_resources,
                                      market='en-US')

                        '''
                        -------------------------------------
                        Web Sites
                        -------------------------------------
                        '''
                        self.sys.log.info(':: [%s] caching (web site) ->' % term)
                        values = (term, ret_id_term.lastrowid, self.config.search_engine_api, 1,
                                      self.config.search_engine_features_text, str(strftime("%Y-%m-%d %H:%M:%S", gmtime())),
                                      self.config.search_engine_tot_resources)
                        c.execute("""INSERT into HORUS_TERM_SEARCH(term, id_term, id_search_engine, id_search_type,
                                                 search_engine_features, query_date, query_tot_resource)
                                                 VALUES(?, ?, ?, ?, ?, ?, ?)""", values)
                        id_term_search = c.lastrowid
                        item.extend([id_term_search])  # updating matrix

                        seq = 0
                        for web_result in result_txts:
                            seq+=1
                            row = (id_term_search, 0, web_result['id'], seq, web_result['displayUrl'], web_result['name'],
                                   web_result['snippet'], '')

                            c.execute("""INSERT INTO HORUS_SEARCH_RESULT_TEXT (id_term_search, id_ner_type,
                                         search_engine_resource_id, result_seq, result_url, result_title,
                                         result_description, result_html_text) VALUES(?,?,?,?,?,?,?,?)""", row)

                            c.execute("""UPDATE HORUS_TERM_SEARCH SET metaquery = '%s' 
                                         WHERE id = %s""" % (metaquery, id_term_search))

                        # term has not returned a result
                        if seq == 0:
                            row = (id_term_search, 0, '', seq, '', '', '', '')
                            c.execute("""INSERT INTO HORUS_SEARCH_RESULT_TEXT (id_term_search, id_ner_type,
                                         search_engine_resource_id, result_seq, result_url, result_title,
                                         result_description, result_html_text) VALUES(?,?,?,?,?,?,?,?)""", row)

                            c.execute("""UPDATE HORUS_TERM_SEARCH SET metaquery = '%s' 
                                         WHERE id = %s""" % (metaquery, id_term_search))

                        '''
                        -------------------------------------
                        Images
                        -------------------------------------
                        '''
                        self.sys.log.info(':: [%s] caching - image' % term)
                        values = (term, ret_id_term.lastrowid if type(ret_id_term) is not int else ret_id_term, self.config.search_engine_api,
                                  2, self.config.search_engine_features_img, str(strftime("%Y-%m-%d %H:%M:%S", gmtime())),
                                  self.config.search_engine_tot_resources)

                        sql = """INSERT into HORUS_TERM_SEARCH(term, id_term, id_search_engine, id_search_type,
                                 search_engine_features, query_date, query_tot_resource) VALUES(?,?,?,?,?,?,?)"""
                        c.execute(sql, values)
                        id_term_img = c.lastrowid
                        item.extend([id_term_img])  # updating matrix
                        seq = 0
                        for web_img_result in result_imgs:
                            self.sys.log.debug(':: downloading image [%s]' % (web_img_result['Title']))
                            seq += 1
                            auxtype = self.download_image_local(web_img_result['contentUrl'],
                                      web_img_result['encodingFormat'], web_img_result['thumbnailUrl'],
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

                            c.execute("""UPDATE HORUS_TERM_SEARCH SET metaquery = '%s'
                                         WHERE id = %s""" % (metaquery, id_term_img))

                            self.sys.log.debug(':: term [%s] cached (img)!' % term)
                    # done caching
                    self.conn.commit()
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
        auxi = 0
        toti = len(self.horus_matrix)
        for item in self.horus_matrix:
            auxi += 1
            if (item[5] in definitions.POS_NOUN_TAGS) or item[7] == 1:

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
                                                 result_description_en, processed, text_1_klass, text_2_klass,
                                                 text_3_klass, text_4_klass, text_5_klass
                                          FROM HORUS_SEARCH_RESULT_TEXT
                                          WHERE id_term_search = %s AND id_ner_type = %s""" % (id_term_txt, id_ner_type))
                    rows = cursor.fetchall()
                    tot_err = 0
                    for row in rows:
                        if row[6] == 0 or row[6] is None:
                            ret = self.detect_text_klass(row[2], row[3], row[0], row[4], row[5])
                            y.append(ret)
                            sql = """UPDATE HORUS_SEARCH_RESULT_TEXT
                                     SET processed = 1,
                                         text_1_klass = %s,
                                         text_2_klass = %s,
                                         text_3_klass = %s,
                                         text_4_klass = %s,
                                         text_5_klass = %s
                                     WHERE id = %s""" % (ret[0], ret[1], ret[2], ret[3], ret[4], row[0])
                            #self.sys.log.debug(':: ' + sql)
                            cursor.execute(sql)
                            if ret[0] == -1 or ret[1] == -1 or ret[2] == -1 or ret[3] == -1 or ret[4] == -1:
                                tot_err += 1
                        else:
                            y.append(row[7:12])

                    self.conn.commit()

                    yy = numpy.array(y)
                    gp = [numpy.count_nonzero(yy == 1), numpy.count_nonzero(yy == 2), numpy.count_nonzero(yy == 3)]
                    horus_tx_ner = gp.index(max(gp)) + 1

                    self.sys.log.debug(':: final textual checking statistics for term [%s] '
                                    '(1-LOC = %s, 2-ORG = %s and 3-PER = %s)' % (term, str(gp[0]),
                                                                                           str(gp[1]), str(gp[2])))
                    # 7 to 11
                    metadata.append(len(rows))
                    metadata.append(gp[0])
                    metadata.append(gp[1])
                    metadata.append(gp[2])
                    metadata.append(float(tot_err))

                    maxs_tx = heapq.nlargest(2, gp)
                    dist_tx_indicator = max(maxs_tx) - min(maxs_tx)

                    # 12, 13
                    metadata.append(dist_tx_indicator)
                    metadata.append(definitions.KLASSES[horus_tx_ner])

                    if len(rows) != 0:
                        self.sys.log.debug('-> TX_LOC  indicator: %f %%' % (float(gp[0]) / (len(rows)) * 5.0)) #division by current number of classifiers = 5
                        self.sys.log.debug('-> TX_ORG  indicator: %f %%' % (float(gp[1]) / (len(rows)) * 5.0))
                        self.sys.log.debug('-> TX_PER  indicator: %f %%' % (float(gp[2]) / (len(rows)) * 5.0))
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
        updates the predictions based on the compound
        #  updating components matrix
        # 0 = is_entity?,    1 = index_sent,   2 = index_word, 3 = word/term,
        # 4 = pos_universal, 5 = pos,          6 = ner       , 7 = compound? ,
        # 8 = compound_size, 9 = id_term_txt, 10 = id_term_img
        :return:
        '''
        self.sys.log.info(':: updating compounds predictions')
        i_y, i_sent, i_first_word, i_c_size = [], [], [], []
        for i in range(len(self.horus_matrix)):
            if self.horus_matrix[i][7] == 1:
                i_y.append(self.horus_matrix[i][25])
                i_sent.append(self.horus_matrix[i][1])
                i_first_word.append(self.horus_matrix[i][2])
                i_c_size.append(int(self.horus_matrix[i][8]))
            if self.horus_matrix[i][7] == 0:
                for z in range(len(i_y)):
                    if i_sent[z] == self.horus_matrix[i][1] and \
                                    i_first_word[z] == self.horus_matrix[i][2]:
                        for k in range(i_c_size[z]):
                            self.horus_matrix[i+k][25] = i_y[z]
        self.sys.log.info(':: alles klar!')

    def process_input_text(self, text):
        self.sys.log.info(':: text: ' + text)
        self.sys.log.info(':: tokenizing sentences ...')
        sent_tokenize_list = sent_tokenize(text)
        self.sys.log.info(':: processing ' + str(len(sent_tokenize_list)) + ' sentence(s).')
        horus = []
        sentences = []
        w = ''
        ner = ''
        pos = ''
        hasNER = -1
        isent = 0
        itoken = 0
        for sentence in sent_tokenize_list:
            isent+=1
            tokens, pos_taggers, pos_universal = self.tokenize_and_pos(sentence, self.config.models_pos_tag_lib)
            compounds = self.get_compounds(pos_taggers)
            chunked = nltk.ne_chunk(pos_taggers)
            if len(chunked) != len(tokens):
                raise Exception("err: this should never happen! token arrays size mismatching")
            if len(compounds) > 0:
                for comp in compounds:
                    horus.append([-1, isent, comp[0], comp[1], '', '', '', 1, comp[2]])
            for ch in chunked:
                itoken+=1
                if type(ch) is nltk.Tree:
                    w = ch[0][0]
                    pos = ch[0][1]
                    if ch._label == 'GPE' or ch._label == 'LOCATION':
                        ner = 'LOC'
                    elif ch._label == 'PERSON':
                        ner = 'PER'
                    elif ch._label == 'ORGANIZATION':
                        ner = 'ORG'
                    else:
                        ner = 'O'
                else:
                    w = ch[0]
                    pos = ch[1]
                    ner = 'O'

                horus.append([-1, isent, itoken, tokens[itoken-1], pos_universal[itoken-1][1], pos_taggers[itoken-1][1], ner, 0, 0])
                w = ''
                pos = ''
                ner = ''

            #_pos = list(zip(*pos_universal)[1])
            #sentences.append(compounds)
            #sentences.append([hasNER, sentence, tokens, ner, pos, _pos])
            #pos = []
            #ner = []
            #w = []
            compounds = None
        return horus

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
            c = self.conn.execute("""SELECT sentence_has_NER,
                              sentence, same_tokenization_nltk, same_tokenization_stanford, same_tokenization_tweetNLP,
                              corpus_tokens, annotator_nltk_tokens, annotator_stanford_tokens, annotator_tweetNLP_tokens,
                              corpus_ner_y, annotator_nltk_ner, annotator_stanford_ner, annotator_tweetNLP_ner,
                              corpus_pos_y, annotator_nltk_pos, annotator_stanford_pos, annotator_tweetNLP_pos,
                              corpus_pos_uni_y, annotator_nltk_pos_universal, annotator_stanford_pos_universal, annotator_tweetNLP_pos_universal,
                              annotator_nltk_compounds, annotator_stanford_compounds, annotator_tweetNLP_compounds
                                     FROM HORUS_SENTENCES
                                     WHERE corpus_name = ? and sentence = ?""", (corpus, sentence))
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
            with open(dspath) as f:
                for line in f:
                    if line.strip() != '':
                        token = line.split('\t')[0]
                        ner = line.split('\t')[1].replace('\r','').replace('\n','')
                    if line.strip() == '':
                        if len(tokens) != 0:
                            # that' a sentence, check if cached!
                            cache_sent = self.sentence_cached_before(dataset_name, s)
                            if len(cache_sent) != 0:
                                sentences.append(cache_sent)
                            else:
                                self.sys.log.debug(s)

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
                                _same_tok_nltk = (len(_tokens_nltk) == len(tokens))

                                # stanford tok has the same length of corpus tok?
                                _same_tok_stanf = (len(_tokens_st) == len(tokens))

                                # tweetNLP tok has the same length of corpus tok?
                                _same_tok_tweet = (len(_tokens_twe) == len(tokens))

                                # NLTK NER
                                nernltktags = self.tools.annotate_ner_nltk(_pos_nltk)

                                # stanford NER
                                nerstantags = self.tools.annotate_ner_stanford(s)
                                nerstantags = numpy.array(nerstantags)

                                self.sys.log.debug(':: chunking pattern ...')
                                comp_nltk = self.get_compounds(_pos_nltk)
                                comp_st = self.get_compounds(_pos_st)
                                comp_twe = self.get_compounds(_pos_twe)

                                # saving to database (pos_uni_sta not implemented yet)
                                sent = [has3NER,
                                        [s, 1 if _same_tok_nltk else 0, 1 if _same_tok_stanf else 0, 1 if _same_tok_tweet else 0],
                                        [tokens, _tokens_nltk, _tokens_st, _tokens_twe],
                                        [tags_ner_y, nernltktags, nerstantags[:, 1].tolist(), []],
                                        [[], _pos_nltk[:, 1].tolist(), _pos_st[:, 1].tolist(), _pos_twe[:, 1].tolist()],
                                        [[], _pos_uni_nltk[:, 1].tolist(), [], _pos_uni_twe[:, 1].tolist()],
                                        [[], comp_nltk, comp_st, comp_twe]
                                        ]
                                #TODO: is it necessary?
                                #self.horus_matrix.extend(sent)

                                self.db_save_sentence(sent, dataset_name)
                                sentences.append(sent)

                            tokens = []
                            tags_ner_y = []
                            s = ''
                            has3NER = -1
                    else:
                        s += token + ' '
                        tokens.append(token)
                        tags_ner_y.append(ner)
                        if ner in definitions.NER_RITTER:
                            has3NER = 1

            #just in case of tweetNLP
            #if self.config.models_pos_tag_lib == 2:
            #    #Communicating once the shell process is opened rather than closing comms it's definitely more sensible, so I have done it...
            #    list_shell = []
            #    for item in sentences:
            #        list_shell.append(item[1])
            #    self.sys.log.info(':: opening shell once to tweetNLP pos tagger ...')
            #    tokens, pos, pos_uni = self.tokenize_and_pos(list_shell)
            #    cache_sentences = []
            #    index = 0
            #    for item in sentences:
            #        _pos = zip(*pos[index])[1]
            #        _pos_uni = zip(*pos_uni[index])[1]
            #        cache_sentences.append([item[0], item[1], item[2], item[3], list(_pos), list(_pos_uni), item[6]])
            #        index +=1
            #    return cache_sentences
            #else:
            #    return sentences
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



