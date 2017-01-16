#!/usr/bin/env python
import json
import web
import horus.main

urls = (
    # '/horus', 'horus_info',
    '/horus/(.*)', 'AnnotateSentence'
)

app = web.application(urls, globals())

class AnnotateSentence:
    def GET(self, sentence):
        ret = horus.main.annotate(sentence,'',0,'','json')
        return json.dumps(ret)

if __name__ == "__main__":
    app.run()
