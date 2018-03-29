import json
import logging
import ntpath
import sqlite3

import numpy

import json
import logging
import ntpath
import re
import nltk


from horus.core.util.definitions_sql import SQL_SENTENCE_SAVE
from horus.core.util.nlp_tools import NLPTools
from horus.core.util.sqlite_helper import SQLiteHelper, HorusDB
from horus.core.search_engines import query_bing, query_flickr, query_wikipedia
from horus.core.util import definitions
from horus.core.util.nlp_tools import NLPTools
from horus.core.util.sqlite_helper import SQLiteHelper, HorusDB
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from time import gmtime, strftime
import requests
from horus.core.search_engines import query_bing, query_flickr, query_wikipedia
from horus.core.util.sqlite_helper import SQLiteHelper, HorusDB
import string
import unicodedata
import re
import langdetect
from horus.core.util.systemlog import SysLogger


class Util(object):
    def __init__(self, config):
        #self.sys = SystemLog("horus.log", logging.DEBUG, logging.DEBUG)
        self.tools = NLPTools()
        self.html_escape_table = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&apos;",
            ">": "&gt;",
            "<": "&lt;",
        }
        self.config = config
        self.logger = SysLogger().getLog()
        self.conn = sqlite3.connect(self.config.database_db)

    def translate_old(self, t1, t2, id, t1en, t2en):
        from translate import Translator


        try:

            ret_error = [-1, -1, -1, -1, -1]

            if isinstance(t1, str):
                t1 = unicode(t1, "utf-8")
            if isinstance(t2, str):
                t2 = unicode(t2, "utf-8")

            # print t1.encode("utf-8")
            # print t2.encode("utf-8")

            # t1 = t1.decode('utf-8')
            # t2 = t2.decode('utf-8')

            # content = unicode(t1.strip(codecs.BOM_UTF8), 'utf-8')

            # print self.remove_accents(t1)
            # t1 = self.remove_non_ascii(t1)
            # t2 = self.remove_non_ascii(t2)

            t1final = t1
            t2final = t1

            # https://pypi.python.org/pypi/translate (alternative 1000 per day)
            # https://www.microsoft.com/en-us/translator/getstarted.aspx
            # https://github.com/openlabs/Microsoft-Translator-Python-API

            c = self.conn.cursor()

            # need to save to components db
            if t1en is None:
                lt1 = langdetect.detect(t1)
                if lt1 != 'en':
                    try:
                        t1final = self.translator.translate(t1, 'en')
                    except Exception as e1:
                        self.logger.error(':: Error, trying another service: ' + str(e1))
                        try:
                            translator2 = Translator(from_lang=lt1, to_lang="en")
                            t1final = translator2.translate(t1)
                        except Exception as e2:
                            self.logger.error(':: Error at service 2: ' + str(e2))
                            return ret_error
                            # updating

                sql = """UPDATE HORUS_SEARCH_RESULT_TEXT SET result_title_en = ? WHERE id = ?"""
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
                        self.logger.error(':: Error, trying another service: ' + str(e1))
                        try:
                            translator2 = Translator(from_lang=lt2, to_lang="en")
                            t2final = translator2.translate(t2)
                        except Exception as e2:
                            self.logger.error(':: Error at service 2: ' + str(e2))
                            return ret_error
                            # updating

                sql = """UPDATE HORUS_SEARCH_RESULT_TEXT SET result_description_en = ? WHERE id = ?"""
                c.execute(sql, (t2final.encode("utf-8"), id))
                self.conn.commit()
            else:
                t2final = t2en

            c.close()
        except Exception as e:
            self.logger.error(':: Error: ' + str(e))
            return False, ret_error

        return t1final, t2final

    def translate_old2(self, text):
        try:

            t1final, t2final = translate(t1, t2, id, t1en, t2en)
            if t1final is False:
                return t2final  # error vector
            else:

                if config.config.text_classification_type == 0:  # TFIDF
                    predictions = [self.text_checking_model_1.predict(docs)[0],
                                   self.text_checking_model_2.predict(docs)[0],
                                   self.text_checking_model_3.predict(docs)[0],
                                   self.text_checking_model_4.predict(docs)[0],
                                   self.text_checking_model_5.predict(docs)[0]]
                elif self.config.text_classification_type == 1:  # TopicModeling
                    dict = self.classifier_tm.score(docs)
                    predictions = []
                    predictions.append(dict.get('loc'))
                    predictions.append(dict.get('per'))
                    predictions.append(dict.get('org'))
                    predictions.append(0)
                    predictions.append(0)
                else:
                    raise Exception('parameter value not implemented: ' + str(self.config.object_detection_type))

        except Exception as e:
            self.logger.error(':: Error: ' + str(e))
            predictions = [-1, -1, -1, -1, -1]

    def __get_compounds(self, tokens):
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

    def html_escape(self, text):
        return "".join(self.html_escape_table.get(c, c) for c in text)

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
            # if i >= 14:
            #    start = i - 14
            # elif i >= 13:
            #    start = i - 13
            # elif i >= 12:
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
                        p = (term + x[ix + 1])
                    if (y[z] == term or y[z] == p) and (
                            fine1 or fine2 or fine3 or fine4 or fine5 or fine6 or fine7 or fine8):
                        #  ok, is the correct one by value and position
                        index_token = y.index(y[z])
                        q = False
                        break
                except Exception:
                    continue
            # start to merge stuff and try to locate it
            merged = ''
            print '-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-'
            ntimes = len(x) - start
            print 'ntimes = %s' % (ntimes)
            while q is True:
                for slide in range(ntimes):
                    print 'slide = %s' % (slide)
                    merged = ''
                    if q is False:
                        break
                    for m in range(start, len(x)):  # start, len(x)
                        xm = x[m].replace(u'``', u'"').replace('\'\'', u'"')
                        merged = merged + xm
                        print 'm = %s, xm = %s, merged = %s' % (m, xm, merged)
                        try:
                            index_token = y.index(merged)
                            af = (x[ix + 1] if ix + 1 < len(x) else '')
                            bf = (x[ix - 1] if ix > 0 else '')

                            af = af.replace(u'``', u'"')  # .replace('\'\'', u'"')
                            bf = bf.replace(u'``', u'"')

                            print 'af = %s, bf = %s' % (af, bf)

                            if term in merged and (
                                    af in merged or bf in merged):  # if it is merged, at least 2 MUST be included
                                q = False
                                break
                        except Exception:
                            continue
                    start += 1
                if q is True:
                    return None

        except Exception as e:
            print(':: error on get ner: %s' % e)
            exit(-1)

        return index_token

    def get_ner_mapping2(self, y, x, t, i):
        if i + 1 < len(y):
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
                for itermy in range(len(listy) - 1):
                    if listy[itermy] == termx and (listy[itermy - 1] == listx[itermx - 1]
                                                   or listy[itermy + 1] == listx[itermx + 1]):
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
            print(':: error on get ner: %s' % error)

        return index_ner_y

    def get_ner_mapping_slice(y, x, ix):

        try:
            for i in range(len(x)):
                x[i] = x[i].replace('``', u'"')
                # x[i] = x[i].replace("''", u'"')

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
            merged_bef_1_aft_6 = x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[ix + 5] + x[
                ix + 6] \
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
            merged_bef_6_aft_1 = x[ix - 6] + x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[
                ix + 1] \
                if (ix + 1 < len(x) and ix >= 6) else ''
            merged_bef_5_aft_2 = x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[
                ix + 2] \
                if (ix + 2 < len(x) and ix >= 5) else ''

            # -2 d +3..5
            merged_bef_2_aft_3 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] \
                if (ix + 3 < len(x) and ix >= 2) else ''
            merged_bef_2_aft_4 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] \
                if (ix + 4 < len(x) and ix >= 2) else ''
            merged_bef_2_aft_5 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[
                ix + 5] \
                if (ix + 5 < len(x) and ix >= 2) else ''

            # -3..4 d +2
            merged_bef_3_aft_2 = x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] \
                if (ix + 2 < len(x) and ix >= 3) else ''
            merged_bef_3_aft_4 = x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[
                ix + 4] \
                if (ix + 4 < len(x) and ix >= 3) else ''
            merged_bef_4_aft_2 = x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] \
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

                xbefore2 = x[ix + s[1] - 1] + x[ix + s[1]] \
                    if (ix + s[1] - 1) >= 0 else ''

                xbefore3 = x[ix + s[1] - 2] + x[ix + s[1] - 1] + x[ix + s[1]] \
                    if (ix + s[1] - 2) >= 0 else ''

                xbefore4 = x[ix + s[1] - 3] + x[ix + s[1] - 2] + x[ix + s[1] - 1] + x[ix + s[1]] \
                    if (ix + s[1] - 3) >= 0 else ''

                xbefore5 = x[ix + s[1] - 4] + x[ix + s[1] - 3] + x[ix + s[1] - 2] + x[ix + s[1] - 1] + x[ix + s[1]] \
                    if (ix + s[1] - 4) >= 0 else ''

                xafter4 = x[ix + s[2]] + x[ix + s[2] + 1] + x[ix + s[2] + 2] + x[ix + s[2] + 3] \
                    if (ix + s[2] + 3) < len(x) else ''

                xafter3 = x[ix + s[2]] + x[ix + s[2] + 1] + x[ix + s[2] + 2] \
                    if (ix + s[2] + 2) < len(x) else ''

                xafter2 = x[ix + s[2]] + x[ix + s[2] + 1] \
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

    def remove_non_ascii(self, t):
        import string
        printable = set(string.printable)
        temp = filter(lambda x: x in printable, t)
        return temp
        # return "".join(i for i in temp if ord(i) < 128)

    def convert_unicode(self, s):
        # u'abc'.encode('utf-8') -> unicode to str
        # 'abc'.decode('utf-8') -> str to unicode
        if isinstance(s, str):
            return s.decode('utf8')  # unicode(s, 'utf8 )
        elif isinstance(s, unicode):
            return s
        else:
            raise Exception("that's not a string!")

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
                    # if u'&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;lt' == x[i]:
                    #    term = term.replace(u'&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;lt', u'&lt;')
                    # elif u'&' in x[i]:
                    #    term = term.replace(u'&', u'&amp;')
                    # elif u'<' in x[i]:
                    #    term = term.replace(u'<', u'&lt;')
                    # elif u'>' in x[i]:
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
                        term = re.sub("&apos", "'", term)  # trick
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

    def populate_matrix_new_columns(self):
        temp = []  # receives 0=8
        temp.extend([0] * 9)  # 9-17
        temp.extend([definitions.KLASSES[4]])  # 18
        temp.extend([0] * 7)  # 19-25
        temp.extend([definitions.KLASSES[4]])  # 26
        temp.extend([0] * 9)  # 27-35
        temp.extend([definitions.KLASSES[4]] * 15)  # 36-50
        return temp

    def sentence_to_horus_matrix(self, sentences):
        '''
        converts the list to horus_matrix
        :param sentences
        :return: horus_matrix
        '''
        self.logger.info(':: starting conversion to horus_matrix based on system parameters')
        converted = []
        sent_index = 0
        try:
            for sent in sentences:
                sent_index += 1
                ipositionstartterm = 0
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
                        ind_ner_real = self.get_ner_mapping_simple(sent[2][0], sent[2][self.config.models_pos_tag_lib],
                                                              i, starty)
                        starty = ind_ner_real
                        # ind_ner = self.get_ner_mapping_slice(sent[2][0], sent[2][self.config.models_pos_tag_lib], i)
                        # ind_ner = self.get_ner_mapping2(sent[2][0], sent[2][self.config.models_pos_tag_lib], term, i)
                        is_entity = 1 if sent[3][0][ind_ner_real] in definitions.NER_TAGS else 0
                    else:
                        is_entity = -1
                    tag_ner = sent[3][self.config.models_pos_tag_lib][i] if len(
                        sent[3][self.config.models_pos_tag_lib]) > 0 else ''
                    tag_pos = sent[4][self.config.models_pos_tag_lib][i] if len(
                        sent[4][self.config.models_pos_tag_lib]) > 0 else ''
                    tag_pos_uni = sent[5][self.config.models_pos_tag_lib][i] if len(
                        sent[5][self.config.models_pos_tag_lib]) > 0 else ''
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

                    temp = [is_entity, sent_index, word_index, term, tag_pos_uni, tag_pos, tag_ner, 0, 0]  # 0-8
                    temp.extend(self.populate_matrix_new_columns())
                    temp.extend([tag_ner_y])
                    ## that is a hack to integrate to GERBIL
                    # if ipositionstartterm >= len(sent[1][0]):
                    #    ipositionstartterm-=1
                    # if sent[1][0][ipositionstartterm] == term[0]:
                    #    if sent[1][0][ipositionstartterm:ipositionstartterm+len(term)] != term:
                    #        raise Exception("GERBIL integration: error 1!")
                    # else:
                    #    ipositionstartterm-=1
                    #    if sent[1][0][ipositionstartterm] == term[0]:
                    #        if sent[1][0][ipositionstartterm:ipositionstartterm+len(term)] != term:
                    #            raise Exception("GERBIL integration: error 2!")
                    #    else:
                    #        raise Exception("GERBIL integration: error 3!")

                    temp[27] = ipositionstartterm
                    converted.append(temp)
                    ipositionstartterm += (len(term) + 1)

        except Exception as error:
            self.logger.error(':: Erro! %s' % str(error))
            exit(-1)

        return converted

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

    def sentence_cached_before(self, corpus, sentence):
        """This method caches the structure of HORUS in db
        in order to speed things up. The average pre-processing time is about to 4-5sec
        for EACH sentence due to attached components (eg.: stanford tools). If the sentence
        has already been cached, we load and convert strings to list, appending the content
        directly to the matrix, thus optimizing a lot this phase.
        """
        sent = []
        try:
            with SQLiteHelper(self.config.database_db) as sqlcon:
                t = HorusDB(sqlcon)
                t.conn.text_factory = str

                sSql = """SELECT sentence_has_NER, 
                sentence, same_tokenization_nltk, same_tokenization_stanford, same_tokenization_tweetNLP,
                corpus_tokens, annotator_nltk_tokens, annotator_stanford_tokens, annotator_tweetNLP_tokens,
                corpus_ner_y, annotator_nltk_ner, annotator_stanford_ner, annotator_tweetNLP_ner,
                corpus_pos_y, annotator_nltk_pos, annotator_stanford_pos, annotator_tweetNLP_pos,
                corpus_pos_uni_y, annotator_nltk_pos_universal, annotator_stanford_pos_universal, annotator_tweetNLP_pos_universal,
                annotator_nltk_compounds, annotator_stanford_compounds, annotator_tweetNLP_compounds
                FROM HORUS_SENTENCES
                WHERE sentence = ? and corpus_name = ?"""
                c = t.conn.execute(sSql, (sentence, corpus))
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
            self.logger.error(':: an error has occurred: ', e)
            raise
        return sent

    def process_and_save_sentence(self, hasNER, s, dataset_name='', tokens_gold_standard=[], ner_gold_standard=[]):
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

            comp_nltk = self.__get_compounds(_pos_nltk)
            comp_st = self.__get_compounds(_pos_st)
            comp_twe = self.__get_compounds(_pos_twe)

            # saving to database (pos_uni_sta not implemented yet)
            sent = [hasNER,
                    [s, 1 if _same_tok_nltk else 0, 1 if _same_tok_stanf else 0, 1 if _same_tok_tweet else 0],
                    [tokens_gold_standard, _tokens_nltk, _tokens_st, _tokens_twe],
                    [ner_gold_standard, nernltktags, nerstantags[:, 1].tolist(), []],
                    [[], _pos_nltk[:, 1].tolist(), _pos_st[:, 1].tolist(), _pos_twe[:, 1].tolist()],
                    [[], _pos_uni_nltk[:, 1].tolist(), [], _pos_uni_twe[:, 1].tolist()],
                    [[], comp_nltk, comp_st, comp_twe]
                    ]

            self.__db_save_sentence(sent, dataset_name)
            return sent

    def process_ds_conll_format(self, dspath, dataset_name, token_index, ner_index, separator='\t'):
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
            self.logger.info(':: processing sentences...')

            # hack to find problems in CONLL file
            # linenr = 0
            # with open(dspath) as f:
            #    for line in f:
            #        linenr+=1
            #        if line.strip() != '':
            #            if len(line.split()) != 2:
            #                print(linenr)
            # exit(0)

            with open(dspath) as f:
                docstart = False
                for line in f:
                    if line.strip() != '':
                        if separator == '': separator = None
                        token = line.split(separator)[token_index]
                        ner = line.split(separator)[ner_index].replace('\r', '').replace('\n', '')
                    if line.strip() == '':
                        if docstart is False:
                            if len(tokens) != 0:
                                self.logger.debug(':: processing sentence %s' % str(tot_sentences))
                                sentences.append(
                                    self.process_and_save_sentence(has3NER, s, dataset_name, tokens, tags_ner_y))
                                tokens = []
                                tags_ner_y = []
                                s = ''
                                has3NER = -1
                                tot_sentences += 1
                        else:
                            docstart = False
                    else:
                        if token != '-DOCSTART-':
                            s += token + ' '
                            tokens.append(token)
                            tags_ner_y.append(ner)
                            if ner in definitions.NER_RITTER:
                                has3NER = 1
                        else:
                            docstart = True

            self.logger.info(':: %s sentences processed successfully' % str(len(sentences)))
            return sentences
        except Exception as error:
            self.logger.error('caught this error: ' + repr(error))

    def print_annotated_sentence(self, horus_matrix):
        '''
        reads the components matrix and prints the annotated sentences
        :: param horus_matrix:
        :: return: output of annotated sentence
        '''
        x1, x2, x3, x4, x5 = '', '', '', '', ''
        id_sent_aux = horus_matrix[0][1]
        for token in horus_matrix:
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

        self.logger.info(':: sentence annotated :: ')
        self.logger.info(':: KLASS 1 -->: ' + x1)
        self.logger.info(':: KLASS 2 -->: ' + x2)
        self.logger.info(':: KLASS 3 -->: ' + x3)
        self.logger.info(':: KLASS 4 -->: ' + x4)
        self.logger.info(':: KLASS 5 -->: ' + x5)

    def download_image_local(self, image_url, image_type, thumbs_url, thumbs_type, term_id, id_ner_type, seq):
        val = URLValidator()
        auxtype = None
        try:
            val(thumbs_url)
            try:
                img_data = requests.get(thumbs_url).content
                with open('%s%s_%s_%s.%s' % (self.config.cache_img_folder, term_id, id_ner_type, seq, thumbs_type),
                          'wb') as handler:
                    handler.write(img_data)
                    auxtype = thumbs_type
            except Exception as error:
                print('-> error: ' + repr(error))
        except ValidationError, e:
            self.logger.error('No thumbs img here...', e)
            try:
                img_data = requests.get(image_url).content
                with open('%s%s_%s_%s.%s' % (self.config.cache_img_folder, term_id, id_ner_type, seq, image_type),
                          'wb') as handler:
                    auxtype = image_type
                    handler.write(img_data)
            except Exception as error:
                print('-> error: ' + repr(error))
        return auxtype

    def download_and_cache_results(self, matrix):
        try:
            self.logger.info(':: caching results...')
            auxc = 1
            horus_matrix = matrix
            with SQLiteHelper(self.config.database_db) as sqlcon:
                t = HorusDB(sqlcon)
                for index in range(len(horus_matrix)):
                    term = horus_matrix[index][3]
                    if (horus_matrix[index][5] in definitions.POS_NOUN_TAGS) or horus_matrix[index][7] == 1:
                        if auxc%1000==0:
                            self.logger.debug(':: processing term %s - %s [%s]' % (str(auxc), str(len(horus_matrix)), term))
                        res = t.term_cached(term, self.config.search_engine_api, self.config.search_engine_features_text)
                        if res is None or len(res) == 0:
                            '''
                            --------------------------------------------------------------------------
                            Downloading resources...
                            --------------------------------------------------------------------------
                            '''
                            self.logger.info(':: not cached, querying -> [%s]' % term)

                            # Microsoft Bing
                            if int(self.config.search_engine_api) == 1:
                                metaquery, result_txts, result_imgs = query_bing(term,
                                                                                 key=self.config.search_engine_key,
                                                                                 top=self.config.search_engine_tot_resources)
                            # Flickr
                            elif (self.config.search_engine_api) == 3:
                                metaquery, result_imgs = query_flickr(term)
                                metaquery, result_txts = query_wikipedia(term)

                            '''
                            --------------------------------------------------------------------------
                            Caching Documents (Texts)
                            --------------------------------------------------------------------------
                            '''
                            self.logger.debug(':: caching (web sites) -> [%s]' % term)
                            id_term_search = t.save_term(term, self.config.search_engine_tot_resources,
                                                         len(result_txts), self.config.search_engine_api,
                                                         1, self.config.search_engine_features_text,
                                                         str(strftime("%Y-%m-%d %H:%M:%S", gmtime())), metaquery)
                            horus_matrix[index][9] = id_term_search
                            seq = 0
                            for web_result_txt in result_txts:
                                self.logger.info(':: caching (web site) -> [%s]' % web_result_txt['displayUrl'])
                                seq += 1
                                t.save_website_data(id_term_search, seq, web_result_txt['id'], web_result_txt['displayUrl'],
                                                    web_result_txt['name'], web_result_txt['snippet'])
                            '''
                            --------------------------------------------------------------------------
                            Caching Documents (Images)
                            --------------------------------------------------------------------------
                            '''
                            self.logger.info(':: caching (web images) -> [%s]' % term)
                            id_term_img = t.save_term(term, self.config.search_engine_tot_resources,
                                                      len(result_imgs), self.config.search_engine_api,
                                                      2, self.config.search_engine_features_img,
                                                      str(strftime("%Y-%m-%d %H:%M:%S", gmtime())), metaquery)
                            horus_matrix[index][10] = id_term_img
                            seq = 0
                            for web_result_img in result_imgs:
                                self.logger.debug(':: downloading image [%s]' % (web_result_img['name']))
                                seq += 1
                                auxtype = self.download_image_local(web_result_img['contentUrl'],
                                                                    web_result_img['encodingFormat'],
                                                                    web_result_img['thumbnailUrl'],
                                                                    web_result_img['encodingFormat'], id_term_img, 0,
                                                                    seq)
                                self.logger.debug(':: caching image  ...')
                                t.save_image_data(id_term_img, seq, web_result_img['contentUrl'],
                                                  web_result_img['name'],
                                                  web_result_img['encodingFormat'], web_result_img['height'],
                                                  web_result_img['width'], web_result_img['thumbnailUrl'], str(auxtype))

                            t.commit()
                        else:
                            if (len(res) != 2):
                                raise Exception("that should not happen!")
                            if ((1 or 2) not in [row[1] for row in res]):
                                raise Exception("that should not happen auch!")
                            horus_matrix[index][9] = res[0][0]
                            horus_matrix[index][10] = res[1][0]

                    auxc += 1
            #eturn horus_matrix

        except Exception as e:
            self.logger.error(':: an error has occurred: ', e)
            raise e

    def path_leaf(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    def get_cv_annotation(self):
        x = numpy.array(self.horus_matrix)
        return x[:, [3, 4, 12, 13, 14, 15, 16, 17]]

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
                aux += 1
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
                self.horus_matrix.append(
                    [is_entity, i_sent, i_word, sent[2][k], sent[5][k], sent[4][k], sent[3][k], 0, 0])
                i_word += 1

            i_sent += 1

        # commit updates (compounds)
        self.conn.commit()

    def remove_accents(self, data):
        return ' '.join(x for x in unicodedata.normalize('NFKD', data) if x in string.ascii_letters).lower()

    def __cache_sentence_ritter(self, sentence_list):
        self.logger.debug(':: caching Ritter dataset...:')
        i_sent, i_word = 1, 1
        compound, prev_tag = '', ''
        sent_with_ner = 0
        token_ok = 0
        compound_ok = 0
        for sent in sentence_list:

            self.logger.info(':: processing sentence: ' + sent[1])
            if int(sent[1])==29:
                aaa=1

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
                    compound_ok += 1
                    self.horus_matrix.append([1, i_sent, i_word - len(compound.split(' ')), compound, '', '', '', 1,
                                              len(compound.split(' '))])
                    compound = ''
                prev_tag = ''
                prev_word = ''

            # processing tokens

            #  transforming to components matrix
            # 0 = is_entity?,    1 = index_sent, 2 = index_word, 3 = word/term,
            # 4 = pos_universal, 5 = pos,        6 = ner       , 7 = compound? ,
            # 8 = compound_size

            i_word = 1
            for k in range(len(sent[2])):  # list of NER tags
                is_entity = 1 if sent[3] in definitions.NER_RITTER else 0
                self.horus_matrix.append(
                    [is_entity, i_sent, i_word, sent[2][k], sent[5][k], sent[4][k], sent[3][k], 0, 0])
                i_word += 1
                if is_entity:
                    token_ok += 1

            i_sent += 1
            i_word = 1

        self.logger.debug(':: done! total of sentences = %s, tokens = %s and compounds = %s'
                           % (str(sent_with_ner), str(token_ok), str(compound_ok)))

    def __cache_sentence_conll(self, sentence_list):
        self.logger.debug(':: caching coNLL 2003 dataset...:')
        i_sent, i_word = 1, 1
        compound, prev_tag = '', ''
        sent_with_ner = 0
        token_ok = 0
        compound_ok = 0
        for sent in sentence_list:

            self.logger.info(':: processing sentence: ' + sent[1])
            if int(sent[1])==29:
                aaa=1

            # processing compounds
            if sent[0] == 1:
                sent_with_ner += 1
                for chunck_tag in sent[6]:  # list of chunck tags
                    word = sent[2][i_word - 1]
                    if chunck_tag in "I-NP":  # only NP chunck
                        if prev_tag.replace('I-NP', 'NP').replace('B-NP', 'NP') == chunck_tag.replace('I-NP',
                                                                                                      'NP').replace(
                                'B-NP', 'NP'):
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
                    compound_ok += 1
                    self.horus_matrix.append([1, i_sent, i_word - len(compound.split(' ')), compound, '', '', '', 1,
                                              len(compound.split(' '))])
                    compound = ''
                prev_tag = ''
                prev_word = ''

            # processing tokens

            #  transforming to components matrix
            # 0 = is_entity?,    1 = index_sent, 2 = index_word, 3 = word/term,
            # 4 = pos_universal, 5 = pos,        6 = ner       , 7 = compound? ,
            # 8 = compound_size

            i_word = 1
            for k in range(len(sent[2])):  # list of NER tags
                is_entity = 1 if sent[3] in definitions.NER_CONLL else 0

                self.horus_matrix.append(
                    [is_entity, i_sent, i_word, sent[2][k], sent[5][k], sent[4][k], sent[3][k], 0, 0])
                i_word += 1
                if is_entity:
                    token_ok += 1

            self.__db_save_sentence(sent[1], '-', '-', str(sent[3]))
            i_sent += 1
            i_word = 1

        self.logger.debug(':: done! total of sentences = %s, tokens = %s and compounds = %s'
                           % (str(sent_with_ner), str(token_ok), str(compound_ok)))

    def __cache_sentence(self, sentence_format, sentence_list):
        if sentence_format == 1:
            self.__cache_sentence_ritter(sentence_list)
        elif sentence_format == 2:
            self.__cache_sentence_conll(sentence_list)

    def __deletar_depois(self):
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
            # self.conn.commit() -> ja fiz o que tinha que fazer...
        except Exception as e:
            print e
            self.conn.rollback()

    def __db_save_sentence(self, sent, corpus):
        try:
            c = self.conn.cursor()
            self.conn.text_factory = str
            sentence = [corpus, sent[0], sent[1][0], sent[1][1], sent[1][2], sent[1][3],
                        json.dumps(sent[2][0]), json.dumps(sent[2][1]), json.dumps(sent[2][2]), json.dumps(sent[2][3]),
                        json.dumps(sent[3][0]), json.dumps(sent[3][1]), json.dumps(sent[3][2]), json.dumps(sent[3][3]),
                        json.dumps(sent[4][0]), json.dumps(sent[4][1]), json.dumps(sent[4][2]), json.dumps(sent[4][3]),
                        json.dumps(sent[5][0]), json.dumps(sent[5][1]), json.dumps(sent[5][2]), json.dumps(sent[5][3]),
                        json.dumps(sent[6][1]), json.dumps(sent[6][2]), json.dumps(sent[6][3])]
            id = c.execute(SQL_SENTENCE_SAVE, sentence)
            self.conn.commit()
            return id.lastrowid

        except Exception as e:
            self.logger.error(':: an error has occurred: ', e)
            raise

    def __processing_conll_ds(self, dspath):
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