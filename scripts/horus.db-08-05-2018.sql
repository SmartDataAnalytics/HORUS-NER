BEGIN TRANSACTION;
DROP TABLE IF EXISTS `HORUS_TERM_SEARCH`;
CREATE TABLE IF NOT EXISTS `HORUS_TERM_SEARCH` (
	`id`	integer NOT NULL,
	`term`	text,
	`language`	text DEFAULT 'en',
	`id_search_engine`	NUMERIC DEFAULT 1,
	`search_engine_features`	TEXT,
	`id_search_type`	INTEGER,
	`metaquery`	TEXT,
	`query_date`	TEXT,
	`query_tot_resource`	INTEGER,
	`id_term`	INTEGER,
	`tot_results_returned`	INTEGER DEFAULT 0,
	`ignore`	INTEGER NOT NULL DEFAULT 0,
	PRIMARY KEY(`id`)
);
DROP TABLE IF EXISTS `HORUS_TERM`;
CREATE TABLE IF NOT EXISTS `HORUS_TERM` (
	`id`	INTEGER NOT NULL,
	`term`	TEXT NOT NULL,
	PRIMARY KEY(`id`)
);
DROP TABLE IF EXISTS `HORUS_SENTENCES`;
CREATE TABLE IF NOT EXISTS `HORUS_SENTENCES` (
	`id`	INTEGER NOT NULL,
	`sentence_has_NER`	INTEGER,
	`sentence`	TEXT NOT NULL,
	`same_tokenization_nltk`	TEXT,
	`same_tokenization_stanford`	INTEGER,
	`same_tokenization_tweetNLP`	INTEGER,
	`corpus_name`	TEXT,
	`corpus_tokens`	TEXT,
	`corpus_ner_y`	TEXT,
	`corpus_pos_y`	TEXT,
	`corpus_pos_uni_y`	TEXT,
	`annotator_nltk_tokens`	TEXT,
	`annotator_nltk_ner`	TEXT,
	`annotator_nltk_pos`	TEXT,
	`annotator_nltk_pos_universal`	TEXT,
	`annotator_nltk_compounds`	TEXT,
	`annotator_stanford_tokens`	TEXT,
	`annotator_stanford_ner`	TEXT,
	`annotator_stanford_pos`	TEXT,
	`annotator_stanford_pos_universal`	TEXT,
	`annotator_stanford_compounds`	TEXT,
	`annotator_tweetNLP_tokens`	TEXT,
	`annotator_tweetNLP_ner`	TEXT,
	`annotator_tweetNLP_pos`	TEXT,
	`annotator_tweetNLP_pos_universal`	TEXT,
	`annotator_tweetNLP_compounds`	TEXT,
	PRIMARY KEY(`id`)
);
DROP TABLE IF EXISTS `HORUS_SEARCH_TYPES`;
CREATE TABLE IF NOT EXISTS `HORUS_SEARCH_TYPES` (
	`id`	INTEGER NOT NULL,
	`desc`	TEXT NOT NULL,
	PRIMARY KEY(`id`)
);
DROP TABLE IF EXISTS `HORUS_SEARCH_RESULT_TEXT`;
CREATE TABLE IF NOT EXISTS `HORUS_SEARCH_RESULT_TEXT` (
	`id`	integer NOT NULL,
	`id_term_search`	integer NOT NULL,
	`id_ner_type`	integer,
	`search_engine_resource_id`	text,
	`result_seq`	integer,
	`result_url`	text,
	`result_title`	text,
	`result_description`	text,
	`result_html_text`	text,
	`text_1_klass`	INTEGER,
	`text_2_klass`	INTEGER,
	`text_3_klass`	INTEGER,
	`text_4_klass`	INTEGER,
	`text_5_klass`	INTEGER,
	`result_title_en`	TEXT,
	`result_description_en`	TEXT,
	`processed`	INTEGER DEFAULT 0,
	`text_1_klass_cnn`	REAL,
	`text_2_klass_cnn`	REAL,
	`text_3_klass_cnn`	REAL,
	`text_4_klass_cnn`	REAL,
	`text_5_klass_cnn`	REAL,
	`error`	INTEGER NOT NULL DEFAULT 0,
	`error_desc`	TEXT,
	`tot_union_emb_per`	REAL,
	`tot_union_emb_loc`	REAL,
	`tot_union_emb_org`	REAL,
	`tot_union_emb_none`	REAL,
	PRIMARY KEY(`id`)
);
DROP TABLE IF EXISTS `HORUS_SEARCH_RESULT_IMG`;
CREATE TABLE IF NOT EXISTS `HORUS_SEARCH_RESULT_IMG` (
	`id`	integer NOT NULL,
	`id_term_search`	INTEGER NOT NULL,
	`id_ner_type`	integer,
	`search_engine_resource_id`	text,
	`result_seq`	integer,
	`result_media_url`	text,
	`result_media_title`	text,
	`result_media_content_type`	text,
	`result_media_height`	text,
	`result_media_width`	text,
	`result_media_thumb_media_url`	text,
	`result_media_thumb_media_content_type`	text,
	`nr_faces`	INTEGER NOT NULL DEFAULT 0,
	`nr_logos`	INTEGER NOT NULL DEFAULT 0,
	`filename`	TEXT DEFAULT 0,
	`nr_place_1`	INTEGER NOT NULL DEFAULT 0,
	`nr_place_2`	INTEGER NOT NULL DEFAULT 0,
	`nr_place_3`	INTEGER NOT NULL DEFAULT 0,
	`nr_place_4`	INTEGER NOT NULL DEFAULT 0,
	`nr_place_5`	INTEGER NOT NULL DEFAULT 0,
	`nr_place_6`	INTEGER NOT NULL DEFAULT 0,
	`nr_place_7`	INTEGER NOT NULL DEFAULT 0,
	`nr_place_8`	INTEGER NOT NULL DEFAULT 0,
	`nr_place_9`	INTEGER NOT NULL DEFAULT 0,
	`nr_place_10`	INTEGER DEFAULT 0,
	`processed`	INTEGER DEFAULT 0,
	`nr_faces_cnn`	INTEGER DEFAULT 0,
	`nr_logos_cnn`	INTEGER DEFAULT 0,
	`nr_place_1_cnn`	REAL DEFAULT 0,
	`nr_place_2_cnn`	REAL DEFAULT 0,
	`nr_place_3_cnn`	REAL DEFAULT 0,
	`nr_place_4_cnn`	REAL DEFAULT 0,
	`nr_place_5_cnn`	REAL DEFAULT 0,
	`nr_place_6_cnn`	REAL DEFAULT 0,
	`nr_place_7_cnn`	REAL DEFAULT 0,
	`nr_place_8_cnn`	REAL DEFAULT 0,
	`nr_place_9_cnn`	REAL DEFAULT 0,
	`nr_place_10_cnn`	REAL DEFAULT 0,
	`error`	INTEGER DEFAULT 0,
	`error_desc`	TEXT,
	PRIMARY KEY(`id`)
);
DROP TABLE IF EXISTS `HORUS_SEARCH_ENGINE`;
CREATE TABLE IF NOT EXISTS `HORUS_SEARCH_ENGINE` (
	`id`	integer,
	`name`	text,
	PRIMARY KEY(`id`)
);
DROP TABLE IF EXISTS `HORUS_NER_TYPES`;
CREATE TABLE IF NOT EXISTS `HORUS_NER_TYPES` (
	`id`	integer,
	`type`	text,
	`desc`	text,
	PRIMARY KEY(`id`)
);
DROP INDEX IF EXISTS `UK_TERM_VS_IDSEARCH_TYPE`;
CREATE UNIQUE INDEX IF NOT EXISTS `UK_TERM_VS_IDSEARCH_TYPE` ON `HORUS_TERM_SEARCH` (
	`term`,
	`id_search_type`
);
DROP INDEX IF EXISTS `HORUS_TERM_SEARCH_TERM_INDEX`;
CREATE INDEX IF NOT EXISTS `HORUS_TERM_SEARCH_TERM_INDEX` ON `HORUS_TERM_SEARCH` (
	`term`	ASC
);
DROP INDEX IF EXISTS `HORUS_SEARCH_RESULT_TEXT_TERM`;
CREATE INDEX IF NOT EXISTS `HORUS_SEARCH_RESULT_TEXT_TERM` ON `HORUS_SEARCH_RESULT_TEXT` (
	`id_term_search`	ASC
);
DROP INDEX IF EXISTS `HORUS_SEARCH_RESULTS_IMG_TERM`;
CREATE INDEX IF NOT EXISTS `HORUS_SEARCH_RESULTS_IMG_TERM` ON `HORUS_SEARCH_RESULT_IMG` (
	`id_term_search`	ASC
);
COMMIT;
