import sqlite3
from src.core.util.definitions_sql import *


class SQLiteHelper(object):

    def __init__(self, path):
        self.DB_PATH = path

    def __enter__(self):
        try:
            self.conn = sqlite3.connect(self.DB_PATH)
            self.conn.text_factory = str
            return self.conn
        except Exception as error:
            print(error)
            exit(-1)

    def __exit__(self, *args):
        self.conn.close()


class HorusDB(object):

    def __init__(self, conn):
        self.conn = conn

    def commit(self):
        self.conn.commit()

    def __exists_record(self, sql, param):
        try:
            c = self.conn.execute(sql, param)
            res = c.fetchall()
            if res is None or len(res) == 0:
                return False
            else:
                return res
        except Exception as e:
            raise e

    def term_cached(self, term, search_engine, search_engine_features = ''):
        values = (term, search_engine, search_engine_features)
        ret = self.__exists_record(SQL_TERM_SEARCH_SEL, values)
        if ret is not False:
            return ret
        else:
            return None

    def save_term(self, term, query_tot_resource, tot_results_returned, id_searchengine, id_searchtype, search_engine_features, query_date , metaquery):
        try:
            values = (term, id_searchengine, id_searchtype, search_engine_features, query_date, query_tot_resource, tot_results_returned, metaquery)
            id = self.conn.cursor().execute(SQL_TERM_SEARCH_INS, values)
            return id.lastrowid
        except Exception as e:
            raise e

    def save_website_data(self, id_term_search, seq, web_id, web_display_url, web_name, web_snippet):
        try:
            values = (id_term_search, 0, web_id, seq, web_display_url, web_name, web_snippet, '')
            id = self.conn.cursor().execute(SQL_HORUS_SEARCH_RESULT_TEXT_INS, values)
            return id.lastrowid
        except:
            raise

    def save_image_data(self, id_term_img, seq, contentURL, name, encoding_format, height, width, thumbnailUrl, auxtype):
        try:
            fname = ('%s_%s_%s.%s' % (str(id_term_img), str(0), str(seq), str(auxtype)))
            values = (id_term_img, 0, seq, seq, contentURL, name, encoding_format, height, width, thumbnailUrl, encoding_format, fname)
            sql = SQL_HORUS_SEARCH_RESULT_IMG_INS
            id = self.conn.cursor().execute(sql, values)
            return id.lastrowid
        except:
            raise
