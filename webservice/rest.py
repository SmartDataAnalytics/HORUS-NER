#!/usr/bin/env python
import urllib2

from flask import Flask, request, jsonify
import os

#urls = (
#    '/components/(.*)', 'annotate'
#)

#app = web.application(urls, globals())

from src.config import HorusConfig
from src.core.feature_extraction.features import FeatureExtraction

global config
global extractor

config = HorusConfig()
extractor = FeatureExtraction(config, config.mod_image_sift_active, config.mod_text_tfidf_active,
                              config.mod_image_cnn_active, config.mod_text_topic_active)

print(config.mod_image_sift_active, config.mod_text_tfidf_active,
                              config.mod_image_cnn_active, config.mod_text_topic_active)

app = Flask(__name__)

#with app.app_context():
#    # within this block, current_app points to app.

#TODO: implement conll processing - eg.: http://www.patricksoftwareblog.com/receiving-files-with-a-flask-rest-api/

@app.route('/')
@app.route('/index')
def index():
    global config
    _html = "<b>HORUS Framework</b><br>"
    _html += config.description + "<br>"
    _html += "version " + config.version + "<br>"
    _html += "SIFT=" + str(config.mod_image_sift_active) + ", TF-IDF=" + str(config.mod_text_tfidf_active) + \
             ", CNN=" + str(config.mod_image_cnn_active) + ", TopicModeling=" + str(config.mod_text_topic_active) + "<br>"
    _html += "<br>SDA Research<br>"
    _html += "more info: <a href='http://horus-ner.org/'>horus-ner.org</a>"

    return _html

@app.route('/annotate', methods=['GET'])
def annotate():
    global config
    global extractor
    #ext = getattr(flask.g, 'extractor', FeatureExtraction(config, True, True, False, False))
    text = request.args.get('text', '')
    text = urllib2.unquote(text)
    error = ''
    out = ''
    try:
        #ext = flask.g.get('extractor', None)
        out = extractor.extract_features_text(text)
    except BaseException as inst:
        error = str(type(inst).__name__) + ' ' + str(inst)
    return jsonify(text=text, output=out, error=error)


# start the webserver
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.debug = True
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)
