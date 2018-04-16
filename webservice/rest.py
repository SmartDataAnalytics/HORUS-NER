#!/usr/bin/env python
import json
import urllib2

import flask
import web
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os

#urls = (
#    '/components/(.*)', 'annotate'
#)

#app = web.application(urls, globals())

from config import HorusConfig
from src.core.feature_extraction.features import FeatureExtraction

global config
global extractor

config = HorusConfig()
extractor = FeatureExtraction(config, True, True, False, False)

app = Flask(__name__)

#with app.app_context():
#    # within this block, current_app points to app.


@app.route('/')
@app.route('/index')
def index():
    return "HORUS framework"

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
