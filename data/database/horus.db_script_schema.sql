BEGIN TRANSACTION;
CREATE TABLE "HORUS_TERM_SEARCH" (
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
	PRIMARY KEY(`id`)
);
CREATE TABLE `HORUS_TERM` (
	`id`	INTEGER NOT NULL,
	`term`	TEXT NOT NULL,
	PRIMARY KEY(`id`)
);
CREATE TABLE `HORUS_SENTENCES` (
	`id`	INTEGER NOT NULL,
	`sentence`	TEXT NOT NULL,
	`tagged`	TEXT NOT NULL,
	`compounds`	TEXT,
	`tokens`	TEXT,
	PRIMARY KEY(`id`)
);
CREATE TABLE "HORUS_SEARCH_TYPES" (
	`id`	INTEGER NOT NULL,
	`desc`	TEXT NOT NULL,
	PRIMARY KEY(`id`)
);
CREATE TABLE "HORUS_SEARCH_RESULT_TEXT" (
	`id`	integer NOT NULL,
	`id_term_search`	integer NOT NULL,
	`id_ner_type`	integer,
	`search_engine_resource_id`	text,
	`result_seq`	integer,
	`result_url`	text,
	`result_title`	text,
	`result_description`	text,
	`result_html_text`	text,
	`text_klass`	INTEGER,
	`result_title_en`	TEXT,
	`result_description_en`	TEXT,
	`processed`	INTEGER DEFAULT 0,
	PRIMARY KEY(`id`)
);
CREATE TABLE "HORUS_SEARCH_RESULT_IMG" (
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
