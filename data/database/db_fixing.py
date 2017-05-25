import sqlite3
from horus.components.config import HorusConfig

# got some database records issues, thus I had to fix it to avoid
# hammer-solutions in the code
# now each token should have exactly 2 records (text and image)
# even though it may not return anything from the web for one of those

config = HorusConfig()
conn = sqlite3.connect(config.database_db)
c = conn.cursor()
c2  = conn.cursor()

c.execute("""select * from horus_term_search where id = 382748""")
res = c.fetchall()


#SQL = """SELECT ID, TERM, LANGUAGE, ID_SEARCH_ENGINE, SEARCH_ENGINE_FEATURES,
#                ID_SEARCH_TYPE, METAQUERY, QUERY_DATE, QUERY_TOT_RESOURCE,
#                ID_TERM, TOT_RESULTS_RETURNED
#         FROM HORUS_TERM_SEARCH
#         GROUP BY TERM HAVING COUNT(1) =1"""
#c.execute(SQL)
#for row in c:
#    values = (row[1], row[2], row[3], row[4], 2, row[6], '2017-05-25 18:20:00', row[8], row[9], 0)
#    SQLINS = """INSERT INTO HORUS_TERM_SEARCH (TERM, LANGUAGE, ID_SEARCH_ENGINE, SEARCH_ENGINE_FEATURES,
#                ID_SEARCH_TYPE, METAQUERY, QUERY_DATE, QUERY_TOT_RESOURCE,
#                ID_TERM, TOT_RESULTS_RETURNED)
#              VALUES (?,?,?,?,?,?,?,?,?,?)"""
#    c2.execute(SQLINS, values)
#conn.commit()