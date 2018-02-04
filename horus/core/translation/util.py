import logging

import langdetect

from horus.core.util.systemlog import SystemLog


def translate(self, t1, t2, id, t1en, t2en):
    from translate import Translator
    self.sys = SystemLog("horus.log", logging.DEBUG, logging.DEBUG)

    try:

        ret_error = [-1, -1, -1, -1, -1]

        if isinstance(t1, str):
            t1 = unicode(t1, "utf-8")
        if isinstance(t2, str):
            t2 = unicode(t2, "utf-8")

        # print t1.encode("utf-8")
        # print t2.encode("utf-8")

        # t1 = t1.decode('utf-8')
        # t2 = t2.decode('utf-8')

        # content = unicode(t1.strip(codecs.BOM_UTF8), 'utf-8')

        # print self.remove_accents(t1)
        # t1 = self.remove_non_ascii(t1)
        # t2 = self.remove_non_ascii(t2)

        t1final = t1
        t2final = t1

        # https://pypi.python.org/pypi/translate (alternative 1000 per day)
        # https://www.microsoft.com/en-us/translator/getstarted.aspx
        # https://github.com/openlabs/Microsoft-Translator-Python-API

        c = self.conn.cursor()

        # need to save to components db
        if t1en is None:
            lt1 = langdetect.detect(t1)
            if lt1 != 'en':
                try:
                    t1final = self.translator.translate(t1, 'en')
                except Exception as e1:
                    self.sys.log.error(':: Error, trying another service: ' + str(e1))
                    try:
                        translator2 = Translator(from_lang=lt1, to_lang="en")
                        t1final = translator2.translate(t1)
                    except Exception as e2:
                        self.sys.log.error(':: Error at service 2: ' + str(e2))
                        return ret_error
                        # updating

            sql = """UPDATE HORUS_SEARCH_RESULT_TEXT
                             SET result_title_en = ? WHERE id = ?"""
            c.execute(sql, (t1final.encode("utf-8"), id))
            self.conn.commit()
        else:
            t1final = t1en

        if t2en is None:
            lt2 = langdetect.detect(t2)
            if lt2 != 'en':
                try:
                    t2final = self.translator.translate(t2, 'en')
                except Exception as e1:
                    self.sys.log.error(':: Error, trying another service: ' + str(e1))
                    try:
                        translator2 = Translator(from_lang=lt2, to_lang="en")
                        t2final = translator2.translate(t2)
                    except Exception as e2:
                        self.sys.log.error(':: Error at service 2: ' + str(e2))
                        return ret_error
                        # updating

            sql = """UPDATE HORUS_SEARCH_RESULT_TEXT
                                    SET result_description_en = ? WHERE id = ?"""
            c.execute(sql, (t2final.encode("utf-8"), id))
            self.conn.commit()
        else:
            t2final = t2en

        c.close()
    except Exception as e:
        self.sys.log.error(':: Error: ' + str(e))
        return False, ret_error

    return t1final, t2final