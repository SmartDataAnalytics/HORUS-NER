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

#0-5 | 6-10 | 11-15 | 16-20 | 21-25 | 26-27
SQL_OBJECT_DETECTION_SEL = """SELECT filename, id, processed, nr_faces, nr_logos, nr_place_1, 
                                     nr_place_2, nr_place_3, nr_place_4, nr_place_5, nr_place_6, 
                                     nr_place_7, nr_place_8, nr_place_9, nr_place_10, nr_faces_cnn, 
                                     nr_logos_cnn, nr_place_1_cnn, nr_place_2_cnn, nr_place_3_cnn, nr_place_4_cnn, 
                                     nr_place_5_cnn, nr_place_6_cnn, nr_place_7_cnn, nr_place_8_cnn, nr_place_9_cnn, 
                                     nr_place_10_cnn, nr_faces_dlib_cnn  
                    FROM HORUS_SEARCH_RESULT_IMG WHERE id_term_search = %s AND id_ner_type = %s """

SQL_OBJECT_DETECTION_UPD = """UPDATE HORUS_SEARCH_RESULT_IMG 
                      SET nr_faces = ?, nr_logos = ?, nr_place_1 = ?, nr_place_2 = ?, nr_place_3 = ?, nr_place_4 = ?, 
                          nr_place_5 = ?, nr_place_6 = ?, nr_place_7 = ?, nr_place_8 = ?, nr_place_9 = ?, nr_place_10 = ?, 
                          nr_faces_cnn = ?, nr_logos_cnn = ?, nr_place_1_cnn= ?, nr_place_2_cnn= ?, nr_place_3_cnn= ?, nr_place_4_cnn= ?, 
                          nr_place_5_cnn= ?, nr_place_6_cnn= ?, nr_place_7_cnn= ?, nr_place_8_cnn= ?, nr_place_9_cnn= ?, nr_place_10_cnn= ?, nr_faces_dlib_cnn=?, 
                          processed = 1
                      WHERE id = ?"""

SQL_TEXT_CLASS_SEL   = """SELECT id, result_seq, result_title, result_description, result_title_en, result_description_en, 
                             processed, 
                             text_1_klass, text_2_klass, text_3_klass, text_4_klass, text_5_klass,
                             text_1_klass_cnn, text_2_klass_cnn, text_3_klass_cnn, 0, 0, tot_union_emb_per, tot_union_emb_loc, tot_union_emb_org   
                      FROM HORUS_SEARCH_RESULT_TEXT WHERE id_term_search = %s AND id_ner_type = %s"""

SQL_TEXT_CLASS_UPD   = """UPDATE HORUS_SEARCH_RESULT_TEXT SET processed = 1, 
                           text_1_klass = %s, text_2_klass = %s, text_3_klass = %s, text_4_klass = %s, text_5_klass = %s, 
                           text_1_klass_cnn = %s, text_2_klass_cnn = %s, text_3_klass_cnn = %s, text_4_klass_cnn = %s, text_5_klass_cnn = %s,
                           tot_union_emb_per = %s, tot_union_emb_loc = %s, tot_union_emb_org = %s
                           WHERE id = %s"""


SQL_SENTENCE_SAVE = """INSERT INTO HORUS_SENTENCES(corpus_name, sentence_has_NER, sentence,
                            same_tokenization_nltk, same_tokenization_stanford, same_tokenization_tweetNLP,
                            corpus_tokens, annotator_nltk_tokens, annotator_stanford_tokens, annotator_tweetNLP_tokens,
                            corpus_ner_y, annotator_nltk_ner, annotator_stanford_ner, annotator_tweetNLP_ner,
                            corpus_pos_y, annotator_nltk_pos, annotator_stanford_pos, annotator_tweetNLP_pos,
                            corpus_pos_uni_y, annotator_nltk_pos_universal, annotator_stanford_pos_universal, annotator_tweetNLP_pos_universal,
                            annotator_nltk_compounds, annotator_stanford_compounds, annotator_tweetNLP_compounds)
                          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""