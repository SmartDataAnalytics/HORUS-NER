#!/usr/bin/env python
import json
import web
import horus.main

import xml.etree.ElementTree as ET

tree = ET.parse('/Users/esteves/Github/horus-models/horus/webservice/user_data.xml')
root = tree.getroot()

urls = (
    '/horus', 'list_users',
    '/horus/(.*)', 'annotate'
)

app = web.application(urls, globals())

class list_users:
    def GET(self):
	output = 'users:[';
	for child in root:
                print 'child', child.tag, child.attrib
                output += str(child.attrib) + ','
	output += ']';
        return output

class get_user:
    def GET(self, user):
        for child in root:
            if child.attrib['id'] == user:
                return str(child.attrib)

class annotate:
    def GET(self, sentence):
        ret = horus.main.annotate(sentence,'',0,'','json')
        return json.dumps(ret)


if __name__ == "__main__":
    app.run()
