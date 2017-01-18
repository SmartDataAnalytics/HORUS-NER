import sqlite3

from src.components.config import HorusCore

print sqlite3.version
print sqlite3.sqlite_version

CONST_CREATE_DB_ZERO = True

horus = HorusCore('../components.ini')
print 'connecting to file ' + horus.database_db
conn = sqlite3.connect(horus.database_db)  # or use :memory: to put it in RAM
cursor = conn.cursor()

ner_types = (
    (0, 'ALL', 'No type specification applied'),
    (1, 'LOC', 'Location'),
    (2, 'ORG', 'Organisation'),
    (3, 'PER', 'Person')
)

search_engines = (
    (1, 'Microsoft BING'),
    (2, 'Google')
)


def create_horus_database():

    cursor.execute("""DROP TABLE IF EXISTS HORUS_SEARCH_ENGINE""")
    cursor.execute("""DROP TABLE IF EXISTS HORUS_NER_TYPES""")
    cursor.execute("""DROP TABLE IF EXISTS HORUS_SEARCH_RESULT_TEXT""")
    cursor.execute("""DROP TABLE IF EXISTS HORUS_SEARCH_RESULT_IMG""")
    cursor.execute("""DROP TABLE IF EXISTS HORUS_TERM""")

    cursor.execute("""CREATE TABLE HORUS_SEARCH_ENGINE
                          (id integer PRIMARY KEY,
                           name text)
    """)
    cursor.executemany("""INSERT INTO HORUS_SEARCH_ENGINE VALUES (?, ?)""", search_engines)
    print "- table 'HORUS_SEARCH_ENGINE' ok..."

    cursor.execute("""CREATE TABLE HORUS_NER_TYPES
                          (id integer PRIMARY KEY,
                           type text,
                           desc text)
    """)
    cursor.executemany("""INSERT INTO HORUS_NER_TYPES VALUES(?, ?, ?)""", ner_types)
    print "- table 'HORUS_NER_TYPES' ok..."

    cursor.execute("""CREATE TABLE HORUS_TERM
                          (id integer PRIMARY KEY,
                           term text,
                           language text default 'en')
    """)
    print "- table 'HORUS_TERM' ok..."

    cursor.execute("""CREATE TABLE HORUS_SEARCH_RESULT_TEXT
                          (id integer PRIMARY KEY,
                           id_term integer,
                           id_type integer,
                           id_search_engine integer,
                           search_engine_features text,
                           search_engine_resource_id text,
                           query_date text,
                           result_seq integer,
                           result_url text,
                           result_title text,
                           result_description text,
                           result_html_text text)
    """)
    print "- table 'HORUS_SEARCH_RESULT_TEXT' ok..."

    cursor.execute("""CREATE TABLE HORUS_SEARCH_RESULT_IMG
                          (id integer PRIMARY KEY,
                           id_term text,
                           id_type integer,
                           id_search_engine integer,
                           search_engine_features text,
                           search_engine_resource_id text,
                           query_date text,
                           result_seq integer,
                           result_media_url text,
                           result_media_title text,
                           result_media_content_type text,
                           result_media_height text,
                           result_media_width text,
                           result_media_thumb_media_url text,
                           result_media_thumb_media_content_type text)
       """)
    print "- table 'HORUS_SEARCH_RESULT_IMG' ok..."

    conn.commit()

if CONST_CREATE_DB_ZERO:
    create_horus_database()

print 'testing...'
cur = conn.execute('select * from HORUS_SEARCH_ENGINE')
res = [dict(id=row[0], name=row[1]) for row in cur.fetchall()]
print res
conn.close()

