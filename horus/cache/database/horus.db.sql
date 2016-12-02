BEGIN TRANSACTION;
CREATE TABLE horus_types
                          (id integer PRIMARY KEY,
                           type text,
                           desc text);
CREATE TABLE "HORUS_TERM_SEARCH" (
	`id`	integer,
	`term`	text,
	`language`	text DEFAULT 'en',
	`id_search_engine`	NUMERIC DEFAULT 1,
	`search_engine_features`	TEXT,
	`id_search_type`	INTEGER,
	`metaquery`	TEXT,
	`query_date`	TEXT,
	PRIMARY KEY(`id`)
);
CREATE TABLE "HORUS_SEARCH_TYPES" (
	`id`	INTEGER NOT NULL,
	`desc`	TEXT NOT NULL,
	PRIMARY KEY(`id`)
);
CREATE TABLE "HORUS_SEARCH_RESULT_TEXT" (
	`id`	integer,
	`id_term`	integer,
	`id_ner_type`	integer,
	`search_engine_resource_id`	text,
	`result_seq`	integer,
	`result_url`	text,
	`result_title`	text,
	`result_description`	text,
	`result_html_text`	text,
	PRIMARY KEY(`id`)
);
CREATE TABLE "HORUS_SEARCH_RESULT_IMG" (
	`id`	integer,
	`id_term`	text,
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
	PRIMARY KEY(`id`)
);
CREATE TABLE HORUS_SEARCH_ENGINE
                          (id integer PRIMARY KEY,
                           name text);
CREATE TABLE HORUS_NER_TYPES
                          (id integer PRIMARY KEY,
                           type text,
                           desc text);
COMMIT;
