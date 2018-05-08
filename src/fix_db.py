import json

from src.config import HorusConfig
from src.core.util.sqlite_helper import SQLiteHelper, HorusDB



sql0 = "select distinct lower(term) as term  from HORUS_TERM_SEARCH where ignore=0 and id_search_type=2 group by lower(term) having count(lower(term)) > 1"
sql1 = "select id from HORUS_TERM_SEARCH where ignore=0 and id_search_type=2 and lower(term) = ? order by id"
#sql2 = "select count(1) from HORUS_SEARCH_RESULT_TEXT where id_term_search = ?"
sql2 = "select count(1) from HORUS_SEARCH_RESULT_IMG where id_term_search = ?"
upd0 = "update horus_term_search set ignore=1 where id=?"

#assert 1==2
'''
ATTENTION: CHECK BEFORE IF THE PARAMETERS ARE CORRECT! 
id_search_type=1 => HORUS_SEARCH_RESULT_TEXT
id_search_type=2 => HORUS_SEARCH_RESULT_IMG
'''
try:
    config = HorusConfig()
    with SQLiteHelper(config.database_db) as sqlcon:
        try:
            # connection
            #t = HorusDB(sqlcon)

            c0=sqlcon.cursor()
            c1=sqlcon.cursor()
            c2=sqlcon.cursor()
            c3=sqlcon.cursor()
            sqlcon.text_factory = str
            c0.execute(sql0)
            res0 = c0.fetchall()
            if not res0 is None:
                for reg0 in res0:
                    token = reg0[0]
                    c1.execute(sql1, [token])
                    res1 = c1.fetchall()
                    counts = []
                    for reg1 in res1:
                        id = reg1[0]
                        c2.execute(sql2, [id])
                        counts.append(c2.fetchone())
                    if (len(set(counts)) <= 1) is True:
                        print(token, 'has same size', str(counts))
                        # all have same number of retrieved records, just use anyone and ignore others
                        for i in range(len(res1)-1):
                            id = res1[i][0]
                            c3.execute(upd0, (id,))
                            print(' -- marked to ignore at pos', str(i))
                    else:
                        print(token, 'has different sizes', str(counts))
                        # different retrieved records, get max
                        ok=counts.index(max(counts))
                        for i in range(len(res1)):
                            if i==ok: continue
                            id = res1[i][0]
                            c3.execute(upd0, (id,))
                            print(' -- marked to ignore at pos', str(i))
            sqlcon.commit()
            print('OK!')
        except Exception as e:
            sqlcon.rollback()
            print(e)
except Exception as e:
    print e