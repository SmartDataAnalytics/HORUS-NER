#!/usr/bin/env python
import web
import horus.main

urls = (
    # '/horus', 'horus_info',
    '/horus/(.*)', 'AnnotateSentence'
)

app = web.application(urls, globals())


class AnnotateSentence:
    def GET(self, user):
    horus.main.
	for child in root:
		if child.attrib['id'] == user:
		    return str(child.attrib)

if __name__ == "__main__":
    app.run()