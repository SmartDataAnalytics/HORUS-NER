#!/usr/bin/env python
import json

import web

from src.components import Core

urls = (
    '/components/(.*)', 'annotate'
)

app = web.application(urls, globals())


class annotate:
    def GET(self, sentence):
        horus = Core(5)
        ret = horus.annotate(sentence,'',0,'','json')
        return json.dumps(ret)


if __name__ == "__main__":
    app.run()
