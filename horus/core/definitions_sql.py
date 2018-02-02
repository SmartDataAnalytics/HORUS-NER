SQL_TERM_SEARCH_INS = """INSERT into HORUS_TERM_SEARCH(term, id_search_engine, id_search_type,
                     search_engine_features, query_date, query_tot_resource, tot_results_returned, metaquery)
                     VALUES(?,?,?,?,?,?,?,?)"""

SQL_TERM_SEARCH_SEL = """SELECT id, id_search_type, tot_results_returned 
                             FROM HORUS_TERM_SEARCH 
                             WHERE upper(term) = upper(?) AND id_search_engine = ? AND search_engine_features = ? 
                             ORDER BY term, id_search_type ASC LIMIT 2"""

SQL_HORUS_SEARCH_RESULT_TEXT_INS = """INSERT INTO HORUS_SEARCH_RESULT_TEXT (id_term_search, id_ner_type,
                                         search_engine_resource_id, result_seq, result_url, result_title,
                                         result_description, result_html_text) VALUES(?,?,?,?,?,?,?,?)"""


SQL_HORUS_SEARCH_RESULT_IMG_INS = """INSERT INTO HORUS_SEARCH_RESULT_IMG (id_term_search, id_ner_type, 
                     search_engine_resource_id, result_seq, result_media_url, result_media_title,
                     result_media_content_type, result_media_height, result_media_width, 
                     result_media_thumb_media_url, result_media_thumb_media_content_type, filename)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"""