import sqlite3

class SQLiteHelper(object):

    TYPE_QUERY_TEXT = 0
    TYPE_QUERY_TRIPLE = 1
    TYPE_QUERY_CLAIM = 2
    NO_CLAIM_ID = 0
    DB_PATH = ''

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
        sel_sql = """SELECT id, id_search_type, tot_results_returned 
                             FROM HORUS_TERM_SEARCH 
                             WHERE upper(term) = ? AND id_search_engine = ? AND search_engine_features = ? 
                             ORDER BY term, id_search_type ASC LIMIT 2"""
        values = (term.upper(), search_engine, search_engine_features)
        ret = self.__exists_record(sel_sql, values)
        if ret is not False:
            return ret
        else:
            return None

    def save_term(self, term, query_tot_resource, tot_results_returned, id_searchengine, id_searchtype, search_engine_features, query_date , metaquery):
        try:
            values = (term, query_tot_resource, tot_results_returned, id_searchengine, id_searchtype, search_engine_features, query_date, metaquery)
            sql = """INSERT into HORUS_TERM_SEARCH(term, id_search_engine, id_search_type,
                     search_engine_features, query_date, query_tot_resource, tot_results_returned, metaquery)
                     VALUES(?,?,?,?,?,?,?,?)"""
            id = self.conn.cursor().execute(sql, values)
            return id.lastrowid
        except Exception as e:
            raise e

    def save_website_data(self, id_term_search, seq, web_id, web_display_url, web_name, web_snippet):
        try:
            values = (id_term_search, 0, web_id, seq, web_display_url, web_name, web_snippet, '')
            sql = """INSERT INTO HORUS_SEARCH_RESULT_TEXT (id_term_search, id_ner_type,
                                         search_engine_resource_id, result_seq, result_url, result_title,
                                         result_description, result_html_text) VALUES(?,?,?,?,?,?,?,?)"""
            id = self.conn.cursor().execute(sql, values)
            return id.lastrowid
        except Exception as e:
            raise e

    def save_image_data(self, id_term_img, seq, contentURL, name, encoding_format, height, width, thumbnailUrl, auxtype):
        try:
            fname = ('%s_%s_%s.%s' % (str(id_term_img), str(0), str(seq), str(auxtype)))
            values = (id_term_img, 0, seq, seq, contentURL, name, encoding_format, height, width, thumbnailUrl, encoding_format, fname)
            sql = """INSERT INTO HORUS_SEARCH_RESULT_IMG (id_term_search, id_ner_type, 
                     search_engine_resource_id, result_seq, result_media_url, result_media_title,
                     result_media_content_type, result_media_height, result_media_width, 
                     result_media_thumb_media_url, result_media_thumb_media_content_type, filename)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"""
            id = self.conn.cursor().execute(sql, values)
            return id.lastrowid
        except Exception as e:
            raise e
