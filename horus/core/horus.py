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
import logging

from horus.core.config import HorusConfig
from horus.core.util.systemlog import SystemLog


class Horus(object):

    def __init__(self):
        self.sys = SystemLog("horus.log", logging.DEBUG, logging.DEBUG)
        self.config = HorusConfig()

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
                special_char = 1 if len(
                    re.findall('(http://\S+|\S*[^\w\s]\S*)', self.horus_matrix[index][3])) > 0 else 0
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
                                 int(self.horus_matrix[index][16]),  # int(self.horus_matrix[index][17])
                                 int(self.horus_matrix[index][20]), int(self.horus_matrix[index][21]),
                                 int(self.horus_matrix[index][22]), float(self.horus_matrix[index][23]),
                                 int(self.horus_matrix[index][24]),  # int(self.horus_matrix[index][25])
                                 one_char_token, special_char, first_capitalized, capitalized))

                features = numpy.array(features)
                features[0][0] = self.final_encoder.transform(features[0][0])
                features[0][1] = self.final_encoder.transform(features[0][1])
                features[0][2] = self.final_encoder.transform(features[0][2])
                self.horus_matrix[index][40] = definitions.KLASSES[self.final.predict(features)[0]]

        except Exception as error:
            raise error